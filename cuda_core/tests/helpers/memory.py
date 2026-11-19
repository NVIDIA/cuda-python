# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Memory-related test helpers (skip guards and resource factories)."""

import pytest

from cuda.core import Device, DeviceMemoryResource, ManagedMemoryResource, PinnedMemoryResource
from cuda.core._utils.cuda_utils import CUDAError, driver, handle_return
from cuda_python_test_helpers import IS_WINDOWS

# Shareable handle type for VirtualMemoryResource allocations on this platform.
VMM_HANDLE_TYPE = "win32_kmt" if IS_WINDOWS else "posix_fd"


class UnreleasedMemoryResourceWarning(Warning):
    """A test or fixture left an owned memory pool or VA reservation alive without ``init_cuda``."""


class BaselineMemoryResourceReleasedWarning(Warning):
    """An ``init_cuda`` test released an owned memory pool or VA reservation that existed before it started."""


def skip_if_mempool_unsupported(device=None):
    """Skip unless ``device`` (default: device 0) supports memory pools."""
    if device is None:
        device = Device(0)
    if not device.properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")


def skip_if_managed_mempool_unsupported(device=None):
    """Skip unless ``device`` (default: device 0) supports managed memory pools.

    This is a property check only; it never constructs a ``ManagedMemoryResource``,
    because looking up the device's default managed pool reserves virtual address
    space for the lifetime of the process. It does not check the CUDA major version
    of the ``cuda.core`` build: ``create_managed_memory_resource_or_skip`` skips
    builds without ``ManagedMemoryResource``.
    """
    if device is None:
        device = Device(0)
    skip_if_mempool_unsupported(device)
    if not device.properties.concurrent_managed_access:
        pytest.skip("Device does not support managed memory pool operations")


def is_pool_resource(mr):
    """Whether ``mr`` has the pool half of the IPC protocol (allocation handle, registry, uuid)."""
    return isinstance(mr, (DeviceMemoryResource, PinnedMemoryResource))


def import_pool_resource(cls, device, alloc_handle):
    """Import the pool shared by ``alloc_handle`` as ``cls``, the class of the exporting resource.

    The class must match the exporter: importing a pinned pool as a
    ``DeviceMemoryResource`` (or the reverse) currently succeeds and yields a
    mislabeled resource (#3502).
    """
    if cls is PinnedMemoryResource:
        return PinnedMemoryResource.from_allocation_handle(alloc_handle)
    assert cls is DeviceMemoryResource, cls
    return DeviceMemoryResource.from_allocation_handle(device, alloc_handle)


def pool_access_flags(mr, device):
    """The ``CUmemAccess_flags`` that ``device`` has on the memory pool of ``mr`` in this process."""
    location = driver.CUmemLocation()
    location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    location.id = device.device_id
    return handle_return(driver.cuMemPoolGetAccess(driver.CUmemoryPool(int(mr.handle)), location))


def skip_if_pinned_memory_unsupported(device):
    try:
        if not device.properties.host_memory_pools_supported:
            pytest.skip("Device does not support host mempool operations")
    except AttributeError:
        pytest.skip("PinnedMemoryResource requires CUDA 13.0 or later")


def skip_if_managed_allocation_unsupported(device):
    """Skip unless ``device`` supports raw managed allocations (``cuMemAllocManaged``).

    Unlike ``skip_if_managed_mempool_unsupported``, this does not require managed
    memory pools.
    """
    if not device.properties.managed_memory:
        pytest.skip("Device does not support managed memory operations")


def create_managed_memory_resource_or_skip(*args, xfail_device=None, **kwargs):
    # Keep the established "skip" helper name for call-site readability.
    if not args and kwargs.get("options") is None and not Device().properties.concurrent_managed_access:
        # Without options this looks up the device's default managed pool, which
        # supports managed allocations only with concurrent managed access. Check
        # the property instead of the lookup's error: the driver returns
        # CUDA_ERROR_NOT_SUPPORTED on some machines and CUDA_ERROR_OUT_OF_MEMORY
        # on others. Dedicated pools (options given) work without it.
        pytest.skip("Device does not support concurrent managed memory access")
    try:
        return ManagedMemoryResource(*args, **kwargs)
    except CUDAError as e:
        if "CUDA_ERROR_NOT_SUPPORTED" in str(e):
            pytest.skip("ManagedMemoryResource is not supported on this platform/device")
        raise
    except RuntimeError as e:
        if "requires CUDA 13.0" in str(e):
            pytest.skip("ManagedMemoryResource requires CUDA 13.0 or later")
        if "concurrent managed access is not available" in str(e).lower():
            pytest.skip("Device does not support concurrent managed memory access")
        raise


def create_pinned_memory_resource_or_xfail(*args, xfail_device=None, **kwargs):
    return PinnedMemoryResource(*args, **kwargs)
