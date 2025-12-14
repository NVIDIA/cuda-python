# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import os

import helpers
import pytest

try:
    from cuda.bindings import driver
except ImportError:
    from cuda import cuda as driver

import cuda.core.experimental
from cuda.core.experimental import Device, DeviceMemoryResource, DeviceMemoryResourceOptions, _device
from cuda.core.experimental._utils.cuda_utils import handle_return


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

    yield
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
    if helpers.IS_WSL or not helpers.supports_ipc_mempool(device):
        pytest.skip("Driver rejects IPC-enabled mempool creation on this platform")

    return device


@pytest.fixture
def ipc_memory_resource(ipc_device):
    POOL_SIZE = 2097152
    options = DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True)
    mr = DeviceMemoryResource(ipc_device, options=options)
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
    num_devices = len(cuda.core.experimental.system.devices)
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


skipif_need_cuda_headers = pytest.mark.skipif(helpers.CUDA_INCLUDE_PATH is None, reason="need CUDA header")
