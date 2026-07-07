# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import multiprocessing
import os
import pathlib
import sys
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, distribution

import pytest

import cuda.core
from cuda.bindings import driver
from cuda.core import (
    Device,
    DeviceMemoryResource,
    DeviceMemoryResourceOptions,
    ManagedMemoryResource,
    ManagedMemoryResourceOptions,
    PinnedMemoryResource,
    PinnedMemoryResourceOptions,
    _device,
)
from cuda.core._utils.cuda_utils import CUDAError, handle_return
from cuda.pathfinder import get_cuda_path_or_home

try:
    from cuda.bindings._test_helpers.mempool import xfail_if_mempool_oom
except ModuleNotFoundError:
    # Older cuda.bindings artifacts (for example 12.9.x backports) do not ship
    # this helper yet. Keep the fallback local so tests against published
    # bindings still xfail the known Windows MCDM mempool setup issue.
    #
    # Keep in sync with cuda_bindings/cuda/bindings/_test_helpers/mempool.py.
    # This copy is intentionally simpler because it only handles cuda_core
    # CUDAError exceptions when the shared helper is absent.
    def _is_windows_mcdm_device(device=0):
        if sys.platform != "win32":
            return False
        import cuda.bindings.nvml as nvml

        device_id = int(getattr(device, "device_id", device))
        (err,) = driver.cuInit(0)
        if err != driver.CUresult.CUDA_SUCCESS:
            return False
        err, pci_bus_id = driver.cuDeviceGetPCIBusId(13, device_id)
        if err != driver.CUresult.CUDA_SUCCESS:
            return False
        pci_bus_id = pci_bus_id.split(b"\x00", 1)[0].decode("ascii")
        nvml.init_v2()
        try:
            handle = nvml.device_get_handle_by_pci_bus_id_v2(pci_bus_id)
            current, _ = nvml.device_get_driver_model_v2(handle)
            return current == nvml.DriverModel.DRIVER_MCDM
        finally:
            nvml.shutdown()

    def xfail_if_mempool_oom(err_or_exc, api_name=None, device=0):
        if api_name is not None and not isinstance(api_name, str):
            device = api_name
            api_name = None

        if "CUDA_ERROR_OUT_OF_MEMORY" not in str(err_or_exc):
            return
        try:
            is_windows_mcdm = _is_windows_mcdm_device(device)
        except Exception:
            # If MCDM detection fails, leave the primary test failure visible.
            return
        if not is_windows_mcdm:
            return

        api_context = f"{api_name} " if api_name else ""
        pytest.xfail(f"{api_context}could not reserve VA for mempool operations on Windows MCDM")


# Import shared test helpers for tests across subprojects.
# PLEASE KEEP IN SYNC with copies in other conftest.py in this repo.
_test_helpers_root = pathlib.Path(__file__).resolve().parents[2] / "cuda_python_test_helpers"
try:
    distribution("cuda-python-test-helpers")
except PackageNotFoundError as exc:
    if not _test_helpers_root.is_dir():
        raise RuntimeError(
            f"cuda-python-test-helpers not installed; expected checkout path {_test_helpers_root}"
        ) from exc

    test_helpers_root = str(_test_helpers_root)
    if test_helpers_root not in sys.path:
        sys.path.insert(0, test_helpers_root)


def pytest_configure(config):
    # When using `parallel-threads` set up mini-plugin to ensure each thread has a CUDA context
    parallel_threads = getattr(config.option, "parallel_threads", 0)
    if parallel_threads == "auto" or int(parallel_threads) > 1:
        config.pluginmanager.register(_CudaCoreParallelPlugin(), name="_cuda_core_parallel_plugin")


@contextmanager
def _init_cuda_context():
    # TODO: rename this to e.g. init_context
    device = Device(0)
    device.set_current()

    # Set option to avoid spin-waiting on synchronization.
    if int(os.environ.get("CUDA_CORE_TEST_BLOCKING_SYNC", 0)) != 0:
        handle_return(
            driver.cuDevicePrimaryCtxSetFlags(device.device_id, driver.CUctx_flags.CU_CTX_SCHED_BLOCKING_SYNC)
        )

    try:
        yield device
    finally:
        _ = _device_unset_current()


def _wrap_worker_cuda_test(func):
    if getattr(func, "_cuda_core_worker_cuda_wrapped", False):
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs = dict(kwargs)  # copy before mutating
        with _init_cuda_context() as device:
            if "init_cuda" in kwargs:
                kwargs["init_cuda"] = device
            if "mempool_device_x2" in kwargs:
                kwargs["mempool_device_x2"] = _mempool_device_impl(2)
            if "mempool_device_x3" in kwargs:
                kwargs["mempool_device_x3"] = _mempool_device_impl(3)
            return func(*args, **kwargs)

    wrapper._cuda_core_worker_cuda_wrapped = True
    return wrapper


def _item_uses_init_cuda(item):
    return "init_cuda" in getattr(item, "fixturenames", ())


class _CudaCoreParallelPlugin:
    """A mini pytest plugin used only for pytest-run-parallel testing.
    pytest-run-parallel spawns new threads for each test and we need to
    initialize and pass the correct CUDA context for each these.

    This plugin looks for context specific fixtures and replaces them
    new context specific fixtures may have to be added.

    This plugin approach is not ideal, it would be nicer to introduce hooks
    into pytest-run-parallel.  Once that issue is closed this would be good
    to refactor: https://github.com/Quansight-Labs/pytest-run-parallel/issues/189
    """

    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_modifyitems(self, config, items):
        for item in items:
            if _item_uses_init_cuda(item):
                item.obj = _wrap_worker_cuda_test(item.obj)


def skip_if_pinned_memory_unsupported(device):
    try:
        if not device.properties.host_memory_pools_supported:
            pytest.skip("Device does not support host mempool operations")
    except AttributeError:
        pytest.skip("PinnedMemoryResource requires CUDA 13.0 or later")


def skip_if_managed_memory_unsupported(device):
    try:
        if not device.properties.memory_pools_supported or not device.properties.concurrent_managed_access:
            pytest.skip("Device does not support managed memory pool operations")
    except AttributeError:
        pytest.skip("ManagedMemoryResource requires CUDA 13.0 or later")
    try:
        ManagedMemoryResource()
    except CUDAError as e:
        xfail_if_mempool_oom(e, device)
        raise
    except RuntimeError as e:
        if "requires CUDA 13.0" in str(e):
            pytest.skip("ManagedMemoryResource requires CUDA 13.0 or later")
        raise


def create_managed_memory_resource_or_skip(*args, xfail_device=None, **kwargs):
    # Keep the established "skip" helper name for call-site readability, even though
    # Windows MCDM mempool OOM setup failures are xfailed instead of skipped.
    try:
        return ManagedMemoryResource(*args, **kwargs)
    except CUDAError as e:
        xfail_if_mempool_oom(e, _device_id_from_resource_options(xfail_device, args, kwargs))
        if "CUDA_ERROR_NOT_SUPPORTED" in str(e):
            pytest.skip("ManagedMemoryResource is not supported on this platform/device")
        raise
    except RuntimeError as e:
        if "requires CUDA 13.0" in str(e):
            pytest.skip("ManagedMemoryResource requires CUDA 13.0 or later")
        raise


def create_pinned_memory_resource_or_xfail(*args, xfail_device=None, **kwargs):
    try:
        return PinnedMemoryResource(*args, **kwargs)
    except CUDAError as e:
        xfail_if_mempool_oom(e, xfail_device)
        raise


@contextmanager
def xfail_on_graph_mempool_oom(device=0):
    try:
        yield
    except CUDAError as e:
        xfail_if_mempool_oom(e, "cuGraphAddMemAllocNode", device)
        raise


def _device_id_from_resource_options(device, args, kwargs):
    if device is not None:
        return device
    options = kwargs.get("options")
    if options is None and args:
        options = args[0]
    if options is None:
        return 0
    if isinstance(options, dict):
        preferred_location = options.get("preferred_location")
        preferred_location_type = options.get("preferred_location_type")
    else:
        preferred_location = getattr(options, "preferred_location", None)
        preferred_location_type = getattr(options, "preferred_location_type", None)
    if preferred_location_type in (None, "device") and isinstance(preferred_location, int) and preferred_location >= 0:
        return preferred_location
    return 0


def _require_ipc_mempool_devices(devices):
    """Return devices if they all support IPC-enabled mempools, otherwise skip."""
    from helpers import IS_WSL, supports_ipc_mempool

    checked_devices = tuple(devices)

    if not all(device.properties.handle_type_posix_file_descriptor_supported for device in checked_devices):
        pytest.skip("Device does not support IPC")

    if IS_WSL or not all(supports_ipc_mempool(device) for device in checked_devices):
        pytest.skip("Driver rejects IPC-enabled mempool creation on this platform")

    return devices


@pytest.fixture(scope="session", autouse=True)
def session_setup():
    # Always init CUDA.
    handle_return(driver.cuInit(0))

    # Never fork processes.
    multiprocessing.set_start_method("spawn", force=True)


@pytest.fixture
def init_cuda():
    with _init_cuda_context() as device:
        yield device


def _device_unset_current() -> bool:
    """Pop current CUDA context.

    Returns True if context was popped, False it the stack was empty.
    """
    ctx = handle_return(driver.cuCtxGetCurrent())
    if int(ctx) == 0:
        # no active context, do nothing
        return False
    handle_return(driver.cuCtxPopCurrent())
    if hasattr(_device._tls, "devices"):
        del _device._tls.devices
    return True


@pytest.fixture
def deinit_cuda():
    # TODO: rename this to e.g. deinit_context
    yield
    _ = _device_unset_current()


@pytest.fixture
def deinit_all_contexts_function():
    def pop_all_contexts():
        max_iters = 256
        for _ in range(max_iters):
            if _device_unset_current():
                # context was popped, continue until stack is empty
                continue
            # no active context, we are ready
            break
        else:
            raise RuntimeError(f"Number of iterations popping current CUDA contexts, exceded {max_iters}")

    return pop_all_contexts


@pytest.fixture
def ipc_device(init_cuda):
    """Obtains a device suitable for IPC-enabled mempool tests, or skips.

    The fixture also tracks every ``multiprocessing.Process`` spawned during
    the test and kills any survivors at teardown. This prevents a stuck child
    (e.g., compute-sanitizer wedged during IPC teardown -- see issue #2004)
    from blocking ``ipc_memory_resource``'s ``mr.close()`` for hours.
    """
    from helpers.child_processes import track_child_processes

    device = init_cuda

    if not device.properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")

    device = _require_ipc_mempool_devices((device,))[0]
    with track_child_processes():
        yield device


@pytest.fixture(
    params=[
        pytest.param("device", id="DeviceMR"),
        pytest.param("pinned", id="PinnedMR"),
    ]
)
def ipc_memory_resource(request, ipc_device):
    """Provides IPC-enabled memory resource (either Device or Pinned)."""
    POOL_SIZE = 2097152
    mr_type = request.param

    if mr_type == "device":
        options = DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True)
        mr = DeviceMemoryResource(ipc_device, options=options)
    else:  # pinned
        skip_if_pinned_memory_unsupported(ipc_device)
        options = PinnedMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True)
        mr = PinnedMemoryResource(options=options)

    assert mr.is_ipc_enabled
    yield mr
    mr.close()
    # TODO(seberg): Make sure the `mr` and it's buffers are fully torn down.
    # May be unnecessary as `mr.close()` is not parallel with other work.
    ipc_device.sync()


@pytest.fixture
def mempool_device(init_cuda):
    """Obtains a device suitable for mempool tests, or skips."""
    device = init_cuda

    if not device.properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")

    return device


def _mempool_device_impl(num):
    num_devices = len(cuda.core.Device.get_all_devices())
    if num_devices < num:
        pytest.skip(f"Test requires at least {num} GPUs")

    devs = [Device(i) for i in range(num)]
    for i in reversed(range(num)):
        devs[i].set_current()  # ends with device 0 current

    if not all(devs[i].can_access_peer(j) for i in range(num) for j in range(num)):
        pytest.skip("Test requires GPUs with peer access")

    if not all(devs[i].properties.memory_pools_supported for i in range(num)):
        pytest.skip("Device does not support mempool operations")

    return devs


@pytest.fixture
def mempool_device_x2(init_cuda):
    """Fixture that provides two devices if available, otherwise skips test."""
    return _mempool_device_impl(2)


@pytest.fixture
def mempool_device_x3(init_cuda):
    """Fixture that provides three devices if available, otherwise skips test."""
    return _mempool_device_impl(3)


@pytest.fixture
def ipc_mempool_device_x2(mempool_device_x2):
    """Fixture that provides two IPC-capable mempool devices, or skips.

    Also tracks/kills any leftover ``multiprocessing.Process`` children at
    teardown for the same reasons documented on :func:`ipc_device`.
    """
    from helpers.child_processes import track_child_processes

    devices = _require_ipc_mempool_devices(mempool_device_x2)
    with track_child_processes():
        yield devices


@pytest.fixture(
    params=[
        pytest.param((DeviceMemoryResource, DeviceMemoryResourceOptions), id="DeviceMR"),
        pytest.param((PinnedMemoryResource, PinnedMemoryResourceOptions), id="PinnedMR"),
        pytest.param((ManagedMemoryResource, ManagedMemoryResourceOptions), id="ManagedMR"),
    ]
)
def memory_resource_factory(request, init_cuda):
    """Parametrized fixture providing memory resource types.

    Returns a 2-tuple of (MRClass, MROptionClass).

    Usage:
        def test_something(memory_resource_factory):
            MRClass, MROptions = memory_resource_factory
            device = Device()
            if MRClass is DeviceMemoryResource:
                mr = MRClass(device)
            elif MRClass is PinnedMemoryResource:
                mr = MRClass()
            elif MRClass is ManagedMemoryResource:
                mr = MRClass()
    """
    return request.param


# Please keep in sync with the copy in the top-level conftest.py.
def _cuda_headers_available() -> bool:
    """Return True if CUDA headers are available, False if no CUDA path is set.

    Raises AssertionError if a CUDA path is set but has no include/ subdirectory.
    """
    cuda_path = get_cuda_path_or_home()
    if cuda_path is None:
        return False
    assert os.path.isdir(os.path.join(cuda_path, "include")), (
        f"CUDA path {cuda_path} does not contain an 'include' subdirectory"
    )
    return True


skipif_need_cuda_headers = pytest.mark.skipif(
    not _cuda_headers_available(),
    reason="need CUDA header",
)
