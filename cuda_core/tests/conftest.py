# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import os
import pathlib
import sys

import pytest

try:
    from cuda.bindings import driver
except ImportError:
    from cuda import cuda as driver

import cuda.core
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
from cuda.core._utils.cuda_utils import handle_return

# Import shared test helpers for tests across subprojects.
_test_helpers_root = pathlib.Path(__file__).resolve().parents[2] / "cuda_python_test_helpers"
if _test_helpers_root.is_dir():
    test_helpers_root = str(_test_helpers_root)
    if test_helpers_root not in sys.path:
        sys.path.insert(0, test_helpers_root)


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


def create_managed_memory_resource_or_skip(*args, **kwargs):
    try:
        return ManagedMemoryResource(*args, **kwargs)
    except RuntimeError as e:
        if "requires CUDA 13.0" in str(e):
            pytest.skip("ManagedMemoryResource requires CUDA 13.0 or later")
        raise


@pytest.fixture(scope="session", autouse=True)
def session_setup():
    # Always init CUDA.
    handle_return(driver.cuInit(0))

    # Never fork processes.
    multiprocessing.set_start_method("spawn", force=True)


@pytest.fixture(scope="function")
def init_cuda():
    # TODO: rename this to e.g. init_context
    device = Device(0)
    device.set_current()

    # Set option to avoid spin-waiting on synchronization.
    if int(os.environ.get("CUDA_CORE_TEST_BLOCKING_SYNC", 0)) != 0:
        handle_return(
            driver.cuDevicePrimaryCtxSetFlags(device.device_id, driver.CUctx_flags.CU_CTX_SCHED_BLOCKING_SYNC)
        )

    yield device
    _ = _device_unset_current()


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


@pytest.fixture(scope="function")
def deinit_cuda():
    # TODO: rename this to e.g. deinit_context
    yield
    _ = _device_unset_current()


@pytest.fixture(scope="function")
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
def ipc_device():
    """Obtains a device suitable for IPC-enabled mempool tests, or skips."""
    # Check if IPC is supported on this platform/device
    device = Device(0)
    device.set_current()

    if not device.properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")

    # Note: Linux specific. Once Windows support for IPC is implemented, this
    # test should be updated.
    if not device.properties.handle_type_posix_file_descriptor_supported:
        pytest.skip("Device does not support IPC")

    # Skip on WSL or if driver rejects IPC-enabled mempool creation on this platform/device
    from helpers import IS_WSL, supports_ipc_mempool

    if IS_WSL or not supports_ipc_mempool(device):
        pytest.skip("Driver rejects IPC-enabled mempool creation on this platform")

    return device


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
    return mr


@pytest.fixture
def mempool_device():
    """Obtains a device suitable for mempool tests, or skips."""
    device = Device(0)
    device.set_current()

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
def mempool_device_x2():
    """Fixture that provides two devices if available, otherwise skips test."""
    return _mempool_device_impl(2)


@pytest.fixture
def mempool_device_x3():
    """Fixture that provides three devices if available, otherwise skips test."""
    return _mempool_device_impl(3)


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


skipif_need_cuda_headers = pytest.mark.skipif(
    not os.path.isdir(os.path.join(os.environ.get("CUDA_PATH", ""), "include")),
    reason="need CUDA header",
)
