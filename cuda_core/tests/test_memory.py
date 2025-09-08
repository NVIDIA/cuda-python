# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    from cuda.bindings import driver
except ImportError:
    from cuda import cuda as driver

import ctypes

import pytest

from cuda.core.experimental import Buffer, Device, DeviceMemoryResource, MemoryResource
from cuda.core.experimental._memory import DLDeviceType
from cuda.core.experimental._utils.cuda_utils import get_binding_version, handle_return


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

    # This should succeed and configure the memory pool release threshold.
    # The resource can be constructed from either a device or device ordinal.
    for device_arg in [device, device.device_id]:
        mr = DeviceMemoryResource(device_arg)

        # Verify basic properties
        assert mr.device_id == device.device_id
        assert mr.is_device_accessible
        assert not mr.is_host_accessible
        assert not mr.is_ipc_enabled

        # Test allocation/deallocation works
        buffer = mr.allocate(1024)
        assert buffer.size == 1024
        assert buffer.device_id == device.device_id
        buffer.close()


def test_mempool():
    if get_binding_version() < (12, 0):
        pytest.skip("Test requires CUDA 12 or higher")
    device = Device()
    device.set_current()

    if not device.properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")

    pool_size = 2097152  # 2MB size

    # Test basic pool creation
    mr = DeviceMemoryResource(device, dict(max_size=pool_size, ipc_enabled=False))
    assert mr.device_id == device.device_id
    assert mr.is_device_accessible
    assert not mr.is_host_accessible
    assert not mr.is_ipc_enabled

    # Test allocation and deallocation
    buffer1 = mr.allocate(1024)
    assert buffer1.handle != 0
    assert buffer1.size == 1024
    assert buffer1.memory_resource == mr
    buffer1.close()

    # Test multiple allocations
    buffer1 = mr.allocate(1024)
    buffer2 = mr.allocate(2048)
    assert buffer1.handle != buffer2.handle
    assert buffer1.size == 1024
    assert buffer2.size == 2048
    buffer1.close()
    buffer2.close()

    # Test stream-based allocation
    stream = device.create_stream()
    buffer = mr.allocate(1024, stream=stream)
    assert buffer.handle != 0
    buffer.close()

    # Test memory copying between buffers from same pool
    src_buffer = mr.allocate(64)
    dst_buffer = mr.allocate(64)
    stream = device.create_stream()
    src_buffer.copy_to(dst_buffer, stream=stream)
    device.sync()
    dst_buffer.close()
    src_buffer.close()

    # Test error cases
    # Test IPC operations are disabled
    buffer = mr.allocate(64)
    ipc_error_msg = "Memory resource is not IPC-enabled"

    with pytest.raises(RuntimeError, match=ipc_error_msg):
        mr._get_allocation_handle()

    with pytest.raises(RuntimeError, match=ipc_error_msg):
        mr.export_buffer(buffer)

    # with pytest.raises(RuntimeError, match=ipc_error_msg):
    #     mr.import_buffer(None)

    buffer.close()


@pytest.mark.parametrize(
    "property_name,expected_type",
    [
        ("reuse_follow_event_dependencies", bool),
        ("reuse_allow_opportunistic", bool),
        ("reuse_allow_internal_dependencies", bool),
        ("release_threshold", int),
        ("reserved_mem_current", int),
        ("reserved_mem_high", int),
        ("used_mem_current", int),
        ("used_mem_high", int),
    ],
)
def test_mempool_properties(property_name, expected_type):
    """Test all properties of the DeviceMemoryResource class."""
    # skip test if cuda version is less than 12
    if get_binding_version() < (12, 0):
        pytest.skip("Test requires CUDA 12 or higher")

    device = Device()
    device.set_current()

    if not device.properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")

    pool_size = 2097152  # 2MB size
    for ipc_enabled in [True, False]:
        mr = DeviceMemoryResource(device, dict(max_size=pool_size, ipc_enabled=ipc_enabled))
        assert mr.is_ipc_enabled == ipc_enabled

        try:
            # Get the property value
            value = getattr(mr, property_name)

            # Test type
            assert isinstance(value, expected_type), f"{property_name} should return {expected_type}, got {type(value)}"

            # Test value constraints
            if expected_type is int:
                assert value >= 0, f"{property_name} should be non-negative"

            # Test memory usage properties with actual allocations
            if property_name in ["reserved_mem_current", "used_mem_current"]:
                # Allocate some memory and check if values increase
                initial_value = value
                buffer = None
                try:
                    buffer = mr.allocate(1024)
                    new_value = getattr(mr, property_name)
                    assert new_value >= initial_value, f"{property_name} should increase or stay same after allocation"
                finally:
                    if buffer is not None:
                        buffer.close()

            # Test high watermark properties
            if property_name in ["reserved_mem_high", "used_mem_high"]:
                # High watermark should never be less than current
                current_prop = property_name.replace("_high", "_current")
                current_value = getattr(mr, current_prop)
                assert value >= current_value, f"{property_name} should be >= {current_prop}"

        finally:
            # Ensure we allocate and deallocate a small buffer to flush any pending operations
            flush_buffer = mr.allocate(64)
            flush_buffer.close()
