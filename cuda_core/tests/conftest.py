# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import gc
import importlib
import multiprocessing
import os
import pathlib
import sys
import warnings

import pytest

# Keep in sync with cuda_bindings/tests/conftest.py.
try:
    import cuda_python_test_helpers._pytest_plugin  # noqa: F401
except ImportError as e:
    # Don't call .resolve(): resolving symlinks can make parents[2] point
    # somewhere other than the monorepo root if a sub-directory is symlinked.
    _test_helpers_root = pathlib.Path(__file__).parents[2] / "cuda_python_test_helpers"
    if not _test_helpers_root.is_dir():
        raise RuntimeError(f"cuda-python-test-helpers not installed and not found at {_test_helpers_root}") from e
    for _k in list(sys.modules):
        if _k == "cuda_python_test_helpers" or _k.startswith("cuda_python_test_helpers."):
            del sys.modules[_k]
    sys.path.insert(0, str(_test_helpers_root))
    importlib.invalidate_caches()

pytest_plugins = ["cuda_python_test_helpers._pytest_plugin"]

from helpers.constants import POOL_SIZE
from helpers.memory import (
    BaselineMemoryResourceReleasedWarning,
    UnreleasedMemoryResourceWarning,
    skip_if_pinned_memory_unsupported,
)

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
from cuda.core._memory._buffer import _live_memory_resource_count
from cuda.core._utils.cuda_utils import handle_return


def pytest_collection_finish(session):
    # The per-test gc.collect() in init_cuda (below) must not re-traverse the
    # collected items and the imported modules on every teardown: freeze them
    # once. Objects created after this point are collected as before (#2381).
    gc.collect()
    gc.freeze()


def pytest_configure(config):
    # When using `parallel-threads` set up mini-plugin to ensure each thread has a CUDA context
    parallel_threads = getattr(config.option, "parallel_threads", 0)
    if parallel_threads == "auto" or int(parallel_threads) > 1:
        config.pluginmanager.register(_CudaCoreParallelPlugin(), name="_cuda_core_parallel_plugin")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    # Runs the OOM reason checker on the first CUDA OOM of a session; see
    # issue #2381 and helpers/oom_diagnostics.py for why this is latched and
    # what it checks (host VA exhaustion vs. physical device memory).
    report = yield
    from helpers import oom_diagnostics

    oom_diagnostics.record_if_oom(item, call, report)
    return report


def pytest_terminal_summary(terminalreporter):
    from helpers import oom_diagnostics

    oom_diagnostics.report_terminal_summary(terminalreporter)


def _wrap_worker_cuda_test(func):
    if getattr(func, "_cuda_core_worker_cuda_wrapped", False):
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs = dict(kwargs)  # copy before mutating
        device = Device(0)
        device.set_current()
        try:
            if "init_cuda" in kwargs:
                kwargs["init_cuda"] = device
            if "mempool_device_x2" in kwargs:
                kwargs["mempool_device_x2"] = _mempool_device_impl(2)
            if "mempool_device_x3" in kwargs:
                kwargs["mempool_device_x3"] = _mempool_device_impl(3)
            if "device_x2" in kwargs:
                kwargs["device_x2"] = _device_x2_impl()

            # These are used by test_green_context.py.  The original fixtures include
            # pytest.skip() but that should have correctly fired by this time.
            if "sm_resource" in kwargs:
                kwargs["sm_resource"] = device.resources.sm
            if "wq_resource" in kwargs:
                kwargs["wq_resource"] = device.resources.workqueue
            if "green_ctx" in kwargs:
                from cuda.core import ContextOptions, SMResourceOptions

                groups, _ = device.resources.sm.split(SMResourceOptions(count=None))
                kwargs["green_ctx"] = device.create_context(ContextOptions(resources=[groups[0]]))
            return func(*args, **kwargs)
        finally:
            # Unlike the `init_cuda` fixture we do not synchronize here
            # to avoid doing so while other workers are still running.
            # (E.g. for stream capture). The fixture cleanup is still run
            # even with pytest-run-parallel after worker join.
            _ = _device_unset_current()

    wrapper._cuda_core_worker_cuda_wrapped = True
    return wrapper


def _item_uses_init_cuda(item):
    return "init_cuda" in getattr(item, "fixturenames", ())


# init_cuda collects at teardown only if the process-wide live MemoryResource
# count differs from its value at setup. That is sound only if no older
# resource is released during a test (it could cancel a new leak), so tests
# that construct resources must use init_cuda, and the leftover check below
# reports resources that outlive a test without it.

# Live count after the last init_cuda test; None until one has run.
_mr_expected_live = None
# Live count at the start of the current item, before its fixtures set up.
_mr_live_at_item_start = 0
_mr_last_init_cuda_nodeid = None


def _device_default_memory_resource_count():
    """Number of main-thread Devices that currently cache a default MR.

    Reaches into Device thread-local state (``_device._tls``) on purpose; update
    this and :func:`_release_device_default_memory_resources` if that moves.
    """
    return sum(d._memory_resource is not None for d in getattr(_device._tls, "devices", ()))


def _release_device_default_memory_resources():
    """Drop each main-thread Device's lazily created default memory resource.

    Otherwise they are still counted when init_cuda compares counts at
    teardown and force a collect in every test that allocates through the
    device default. The setter rejects None,
    hence the private access.
    """
    for d in getattr(_device._tls, "devices", ()):
        d._memory_resource = None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    # Leftover check: compare with the count after the previous init_cuda test,
    # not the session start, because the count legitimately rises over a run
    # (interned GraphMemoryResource, pytest-retained class instances).
    global _mr_expected_live, _mr_live_at_item_start, _mr_last_init_cuda_nodeid
    _mr_live_at_item_start = _live_memory_resource_count()
    try:
        return (yield)
    finally:
        live = _live_memory_resource_count()
        if _item_uses_init_cuda(item):
            _mr_expected_live = live
            _mr_last_init_cuda_nodeid = item.nodeid
        elif _mr_expected_live is not None:
            # Only lower it: a released resource must not hide a later leak.
            _mr_expected_live = min(_mr_expected_live, live)


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


def _require_ipc_mempool_devices(devices):
    """Return devices if they all support IPC-enabled mempools, otherwise skip."""
    from helpers import supports_ipc_mempool

    from cuda_python_test_helpers import IS_WSL

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
def init_cuda(request):
    # TODO: rename this to e.g. init_context
    device = Device(0)
    device.set_current()

    # Set option to avoid spin-waiting on synchronization.
    if int(os.environ.get("CUDA_CORE_TEST_BLOCKING_SYNC", 0)) != 0:
        handle_return(
            driver.cuDevicePrimaryCtxSetFlags(device.device_id, driver.CUctx_flags.CU_CTX_SCHED_BLOCKING_SYNC)
        )

    # Leftover check: growth since the previous init_cuda test means something
    # in between left a resource alive. Collect first to sweep a cyclic
    # leftover, and before the count at setup is recorded so it is reflected there.
    if _mr_expected_live is not None and _mr_live_at_item_start > _mr_expected_live:
        gc.collect()
        held = _live_memory_resource_count() - _mr_expected_live
        seed = getattr(request.config.option, "randomly_seed", None)
        seed_msg = f" (pytest-randomly seed: {seed})" if seed is not None else ""
        warnings.warn(
            f"{_mr_live_at_item_start - _mr_expected_live} memory resource(s) were left alive by "
            f"tests that ran since {_mr_last_init_cuda_nodeid} ({max(held, 0)} still alive after a "
            f"collect). A test or fixture that constructs a memory resource must request "
            f"init_cuda.{seed_msg}",
            UnreleasedMemoryResourceWarning,
            stacklevel=2,
        )

    # Live count at setup, excluding cached Device default resources (they are
    # released at teardown before the count is compared).
    live_at_setup = _live_memory_resource_count() - _device_default_memory_resource_count()
    try:
        yield device
    finally:
        # pytest holds fixture values in item.funcargs (a pytest-internal dict)
        # until all teardown has run; we deliberately drop them so already
        # torn-down fixtures are not counted below. Revisit if pytest changes
        # how it retains fixture values.
        request.node.funcargs.clear()
        # Only a resource in a reference cycle outlives the test frame, so
        # collect only if the count differs from its value at setup. A lower
        # count means an older resource was released, which could mask a leak
        # in this test: collect in that case too.
        _release_device_default_memory_resources()
        live_after_release = _live_memory_resource_count()
        if live_after_release != live_at_setup:
            gc.collect()
        # Force any pool/allocation whose only remaining reference was a local
        # in this test's frame to actually get destroyed now, then drain the
        # context so the stream-ordered frees that destruction enqueues retire
        # before the next test runs (issue #2381). gc.collect() above must run
        # first: cuCtxSynchronize alone cannot drain frees that were never
        # enqueued because their owning object had not been collected yet.
        # With pytest-run-parallel this runs after worker join.
        driver.cuCtxSynchronize()
        _ = _device_unset_current()
        # After the cleanup, so an escalated warning cannot skip it.
        if live_after_release < live_at_setup:
            warnings.warn(
                f"{request.node.nodeid} released {live_at_setup - live_after_release} memory resource(s) "
                f"that existed before it started (count {live_after_release}, snapshot {live_at_setup}). "
                "A leak in this test could have been masked by that release. Find what held the "
                "released resource.",
                BaselineMemoryResourceReleasedWarning,
                stacklevel=2,
            )


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


def _device_x2_impl():
    devices = Device.get_all_devices()
    if len(devices) < 2:
        pytest.skip("Test requires at least 2 CUDA devices")
    return devices[:2]


@pytest.fixture
def device_x2(init_cuda):
    """Provide two CUDA devices, or skip when fewer are available.

    Depends on ``init_cuda`` so that, under pytest-run-parallel, the test is
    wrapped by ``_wrap_worker_cuda_test`` and the devices are re-fetched on the
    worker thread (Device objects are thread-local).
    """
    return _device_x2_impl()


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
