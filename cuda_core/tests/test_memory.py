# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Dummy change
try:
    from cuda.bindings import driver
except ImportError:
    from cuda import cuda as driver

import ctypes

import pytest

from cuda.core.experimental import (
    Buffer,
    Device,
    DeviceMemoryResource,
    MemoryResource,
    VMMAllocatedMemoryResource,
    VMMConfig,
)
from cuda.core.experimental._memory import DLDeviceType
from cuda.core.experimental._utils.cuda_utils import handle_return


class DummyDeviceMemoryResource(MemoryResource):
    def __init__(self, device):
        self.device = device

    def allocate(self, size, stream=None) -> Buffer:
        ptr = handle_return(driver.cuMemAlloc(size))
        return Buffer.from_handle(ptr=ptr, size=size, mr=self)

    def deallocate(self, ptr, size, stream=None):
        handle_return(driver.cuMemFree(ptr))

    @property
    def is_device_accessible(self) -> bool:
        return True

    @property
    def is_host_accessible(self) -> bool:
        return False

    @property
    def device_id(self) -> int:
        return 0


class DummyHostMemoryResource(MemoryResource):
    def __init__(self):
        pass

    def allocate(self, size, stream=None) -> Buffer:
        # Allocate a ctypes buffer of size `size`
        ptr = (ctypes.c_byte * size)()
        self._ptr = ptr
        return Buffer.from_handle(ptr=ctypes.addressof(ptr), size=size, mr=self)

    def deallocate(self, ptr, size, stream=None):
        del self._ptr

    @property
    def is_device_accessible(self) -> bool:
        return False

    @property
    def is_host_accessible(self) -> bool:
        return True

    @property
    def device_id(self) -> int:
        raise RuntimeError("the pinned memory resource is not bound to any GPU")


class DummyUnifiedMemoryResource(MemoryResource):
    def __init__(self, device):
        self.device = device

    def allocate(self, size, stream=None) -> Buffer:
        ptr = handle_return(driver.cuMemAllocManaged(size, driver.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL.value))
        return Buffer.from_handle(ptr=ptr, size=size, mr=self)

    def deallocate(self, ptr, size, stream=None):
        handle_return(driver.cuMemFree(ptr))

    @property
    def is_device_accessible(self) -> bool:
        return True

    @property
    def is_host_accessible(self) -> bool:
        return True

    @property
    def device_id(self) -> int:
        return 0


class DummyPinnedMemoryResource(MemoryResource):
    def __init__(self, device):
        self.device = device

    def allocate(self, size, stream=None) -> Buffer:
        ptr = handle_return(driver.cuMemAllocHost(size))
        return Buffer.from_handle(ptr=ptr, size=size, mr=self)

    def deallocate(self, ptr, size, stream=None):
        handle_return(driver.cuMemFreeHost(ptr))

    @property
    def is_device_accessible(self) -> bool:
        return True

    @property
    def is_host_accessible(self) -> bool:
        return True

    @property
    def device_id(self) -> int:
        raise RuntimeError("the pinned memory resource is not bound to any GPU")


class NullMemoryResource(DummyHostMemoryResource):
    @property
    def is_host_accessible(self) -> bool:
        return False


def buffer_initialization(dummy_mr: MemoryResource):
    buffer = dummy_mr.allocate(size=1024)
    assert buffer.handle != 0
    assert buffer.size == 1024
    assert buffer.memory_resource == dummy_mr
    assert buffer.is_device_accessible == dummy_mr.is_device_accessible
    assert buffer.is_host_accessible == dummy_mr.is_host_accessible
    buffer.close()


def test_buffer_initialization():
    device = Device()
    device.set_current()
    buffer_initialization(DummyDeviceMemoryResource(device))
    buffer_initialization(DummyHostMemoryResource())
    buffer_initialization(DummyUnifiedMemoryResource(device))
    buffer_initialization(DummyPinnedMemoryResource(device))


def buffer_copy_to(dummy_mr: MemoryResource, device: Device, check=False):
    src_buffer = dummy_mr.allocate(size=1024)
    dst_buffer = dummy_mr.allocate(size=1024)
    stream = device.create_stream()

    if check:
        src_ptr = ctypes.cast(src_buffer.handle, ctypes.POINTER(ctypes.c_byte))
        for i in range(1024):
            src_ptr[i] = ctypes.c_byte(i)

    src_buffer.copy_to(dst_buffer, stream=stream)
    device.sync()

    if check:
        dst_ptr = ctypes.cast(dst_buffer.handle, ctypes.POINTER(ctypes.c_byte))

        for i in range(10):
            assert dst_ptr[i] == src_ptr[i]

    dst_buffer.close()
    src_buffer.close()


def test_buffer_copy_to():
    device = Device()
    device.set_current()
    buffer_copy_to(DummyDeviceMemoryResource(device), device)
    buffer_copy_to(DummyUnifiedMemoryResource(device), device)
    buffer_copy_to(DummyPinnedMemoryResource(device), device, check=True)


def buffer_copy_from(dummy_mr: MemoryResource, device, check=False):
    src_buffer = dummy_mr.allocate(size=1024)
    dst_buffer = dummy_mr.allocate(size=1024)
    stream = device.create_stream()

    if check:
        src_ptr = ctypes.cast(src_buffer.handle, ctypes.POINTER(ctypes.c_byte))
        for i in range(1024):
            src_ptr[i] = ctypes.c_byte(i)

    dst_buffer.copy_from(src_buffer, stream=stream)
    device.sync()

    if check:
        dst_ptr = ctypes.cast(dst_buffer.handle, ctypes.POINTER(ctypes.c_byte))

        for i in range(10):
            assert dst_ptr[i] == src_ptr[i]

    dst_buffer.close()
    src_buffer.close()


def test_buffer_copy_from():
    device = Device()
    device.set_current()
    buffer_copy_from(DummyDeviceMemoryResource(device), device)
    buffer_copy_from(DummyUnifiedMemoryResource(device), device)
    buffer_copy_from(DummyPinnedMemoryResource(device), device, check=True)


def buffer_close(dummy_mr: MemoryResource):
    buffer = dummy_mr.allocate(size=1024)
    buffer.close()
    assert buffer.handle is None
    assert buffer.memory_resource is None


def test_buffer_close():
    device = Device()
    device.set_current()
    buffer_close(DummyDeviceMemoryResource(device))
    buffer_close(DummyHostMemoryResource())
    buffer_close(DummyUnifiedMemoryResource(device))
    buffer_close(DummyPinnedMemoryResource(device))


def test_buffer_dunder_dlpack():
    device = Device()
    device.set_current()
    dummy_mr = DummyDeviceMemoryResource(device)
    buffer = dummy_mr.allocate(size=1024)
    capsule = buffer.__dlpack__()
    assert "dltensor" in repr(capsule)
    capsule = buffer.__dlpack__(max_version=(1, 0))
    assert "dltensor" in repr(capsule)
    with pytest.raises(BufferError, match=r"^Sorry, not supported: dl_device other than None$"):
        buffer.__dlpack__(dl_device=())
    with pytest.raises(BufferError, match=r"^Sorry, not supported: copy=True$"):
        buffer.__dlpack__(copy=True)
    with pytest.raises(BufferError, match=r"^Expected max_version tuple\[int, int\], got \(\)$"):
        buffer.__dlpack__(max_version=())
    with pytest.raises(BufferError, match=r"^Expected max_version tuple\[int, int\], got \(9, 8, 7\)$"):
        buffer.__dlpack__(max_version=(9, 8, 7))


@pytest.mark.parametrize(
    ("DummyMR", "expected"),
    [
        (DummyDeviceMemoryResource, (DLDeviceType.kDLCUDA, 0)),
        (DummyHostMemoryResource, (DLDeviceType.kDLCPU, 0)),
        (DummyUnifiedMemoryResource, (DLDeviceType.kDLCUDAHost, 0)),
        (DummyPinnedMemoryResource, (DLDeviceType.kDLCUDAHost, 0)),
    ],
)
def test_buffer_dunder_dlpack_device_success(DummyMR, expected):
    device = Device()
    device.set_current()
    dummy_mr = DummyMR() if DummyMR is DummyHostMemoryResource else DummyMR(device)
    buffer = dummy_mr.allocate(size=1024)
    assert buffer.__dlpack_device__() == expected


def test_buffer_dunder_dlpack_device_failure():
    dummy_mr = NullMemoryResource()
    buffer = dummy_mr.allocate(size=1024)
    with pytest.raises(BufferError, match=r"^buffer is neither device-accessible nor host-accessible$"):
        buffer.__dlpack_device__()


def test_device_memory_resource_initialization():
    """Test that DeviceMemoryResource can be initialized successfully.

    This test verifies that the DeviceMemoryResource initializes properly,
    including the release threshold configuration for performance optimization.
    """
    device = Device()
    if not device.properties.memory_pools_supported:
        pytest.skip("memory pools not supported")
    device.set_current()

    # This should succeed and configure the memory pool release threshold
    mr = DeviceMemoryResource(device.device_id)

    # Verify basic properties
    assert mr.device_id == device.device_id
    assert mr.is_device_accessible is True
    assert mr.is_host_accessible is False

    # Test allocation/deallocation works
    buffer = mr.allocate(1024)
    assert buffer.size == 1024
    assert buffer.device_id == device.device_id
    buffer.close()


def test_vmm_allocator_basic_allocation():
    """Test basic VMM allocation functionality.

    This test verifies that VMMAllocatedMemoryResource can allocate memory
    using CUDA VMM APIs with default configuration.
    """
    device = Device()
    device.set_current()

    # Create VMM allocator with default config
    vmm_mr = VMMAllocatedMemoryResource(device)

    # Test basic allocation
    buffer = vmm_mr.allocate(4096)
    assert buffer.size >= 4096  # May be aligned up
    assert buffer.device_id == device.device_id
    assert buffer.memory_resource == vmm_mr

    # Test deallocation
    buffer.close()

    # Test multiple allocations
    buffers = []
    for i in range(5):
        buf = vmm_mr.allocate(1024 * (i + 1))
        buffers.append(buf)
        assert buf.size >= 1024 * (i + 1)

    # Clean up
    for buf in buffers:
        buf.close()


def test_vmm_allocator_policy_configuration():
    """Test VMM allocator with different policy configurations.

    This test verifies that VMMAllocatedMemoryResource can be configured
    with different allocation policies and that the configuration affects
    the allocation behavior.
    """
    device = Device()
    device.set_current()

    # Test with custom VMM config
    custom_config = VMMConfig(
        allocation_type=driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED,
        location_type=driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE,
        granularity=driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        gpu_direct_rdma=True,
        handle_type=driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        peers=(),
        self_access="rw",
        peer_access="rw",
    )

    vmm_mr = VMMAllocatedMemoryResource(device, config=custom_config)

    # Verify configuration is applied
    assert vmm_mr.config == custom_config
    assert vmm_mr.config.gpu_direct_rdma is True
    assert vmm_mr.config.granularity == driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM

    # Test allocation with custom config
    buffer = vmm_mr.allocate(8192)
    assert buffer.size >= 8192
    assert buffer.device_id == device.device_id

    # Test policy modification
    new_config = VMMConfig(
        allocation_type=driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED,
        location_type=driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE,
        granularity=driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        gpu_direct_rdma=False,
        handle_type=driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        peers=(),
        self_access="r",  # Read-only access
        peer_access="r",
    )

    # Modify allocation policy
    modified_buffer = vmm_mr.modify_allocation(buffer, 16384, config=new_config)
    assert modified_buffer.size >= 16384
    assert vmm_mr.config == new_config
    assert vmm_mr.config.self_access == "r"

    # Clean up
    modified_buffer.close()


def test_vmm_allocator_grow_allocation():
    """Test VMM allocator's ability to grow existing allocations.

    This test verifies that VMMAllocatedMemoryResource can grow existing
    allocations while preserving the base pointer when possible.
    """
    device = Device()
    device.set_current()

    vmm_mr = VMMAllocatedMemoryResource(device)

    # Create initial allocation
    buffer = vmm_mr.allocate(2 * 1024 * 1024)
    original_size = buffer.size

    # Grow the allocation
    grown_buffer = vmm_mr.modify_allocation(buffer, 4 * 1024 * 1024)

    # Verify growth
    assert grown_buffer.size >= 4 * 1024 * 1024
    assert grown_buffer.size > original_size

    # The pointer should ideally be preserved (fast path)
    # but may change if contiguous extension fails (slow path)
    assert grown_buffer.handle is not None

    # Test growing to same size (should return original buffer)
    same_buffer = vmm_mr.modify_allocation(grown_buffer, 4 * 1024 * 1024)
    assert same_buffer.size == grown_buffer.size

    # Test growing to smaller size (should return original buffer)
    smaller_buffer = vmm_mr.modify_allocation(grown_buffer, 2 * 1024 * 1024)
    assert smaller_buffer.size == grown_buffer.size

    # Clean up
    grown_buffer.close()
