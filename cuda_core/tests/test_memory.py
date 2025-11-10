# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

try:
    from cuda.bindings import driver
except ImportError:
    from cuda import cuda as driver
try:
    import numpy as np
except ImportError:
    np = None
import ctypes
import platform

import pytest
from cuda.core.experimental import (
    Buffer,
    Device,
    DeviceMemoryResource,
    DeviceMemoryResourceOptions,
    MemoryResource,
    VirtualMemoryResource,
    VirtualMemoryResourceOptions,
)
from cuda.core.experimental._dlpack import DLDeviceType
from cuda.core.experimental._memory import IPCBufferDescriptor
from cuda.core.experimental._utils.cuda_utils import handle_return
from cuda.core.experimental.utils import StridedMemoryView
from helpers.buffers import DummyUnifiedMemoryResource

from cuda_python_test_helpers import supports_ipc_mempool

POOL_SIZE = 2097152  # 2MB size


@pytest.fixture(scope="function")
def mempool_device():
    """Obtains a device suitable for mempool tests, or skips."""
    device = Device()
    device.set_current()

    if not device.properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")

    return device


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


def test_package_contents():
    expected = [
        "Buffer",
        "MemoryResource",
        "DeviceMemoryResource",
        "DeviceMemoryResourceOptions",
        "IPCBufferDescriptor",
        "IPCAllocationHandle",
        "LegacyPinnedMemoryResource",
        "VirtualMemoryResourceOptions",
        "VirtualMemoryResource",
    ]
    d = {}
    exec("from cuda.core.experimental._memory import *", d)  # noqa: S102
    d = {k: v for k, v in d.items() if not k.startswith("__")}
    assert sorted(expected) == sorted(d.keys())


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
    assert buffer.handle == 0
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


@pytest.mark.parametrize("use_device_object", [True, False])
def test_device_memory_resource_initialization(mempool_device, use_device_object):
    """Test that DeviceMemoryResource can be initialized successfully.

    This test verifies that the DeviceMemoryResource initializes properly,
    including the release threshold configuration for performance optimization.
    """
    device = mempool_device

    # This should succeed and configure the memory pool release threshold.
    # The resource can be constructed from either a device or device ordinal.
    device_arg = device if use_device_object else device.device_id
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


def test_vmm_allocator_basic_allocation():
    """Test basic VMM allocation functionality.

    This test verifies that VirtualMemoryResource can allocate memory
    using CUDA VMM APIs with default configuration.
    """
    device = Device()
    device.set_current()

    # Skip if virtual memory management is not supported
    if not device.properties.virtual_memory_management_supported:
        pytest.skip("Virtual memory management is not supported on this device")

    options = VirtualMemoryResourceOptions()
    # Create VMM allocator with default config
    vmm_mr = VirtualMemoryResource(device, config=options)

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

    This test verifies that VirtualMemoryResource can be configured
    with different allocation policies and that the configuration affects
    the allocation behavior.
    """
    device = Device()
    device.set_current()

    # Skip if virtual memory management is not supported
    if not device.properties.virtual_memory_management_supported:
        pytest.skip("Virtual memory management is not supported on this device")

    # Skip if GPU Direct RDMA is supported (we want to test the unsupported case)
    if not device.properties.gpu_direct_rdma_supported:
        pytest.skip("This test requires a device that doesn't support GPU Direct RDMA")

    # Test with custom VMM config
    custom_config = VirtualMemoryResourceOptions(
        allocation_type="pinned",
        location_type="device",
        granularity="minimum",
        gpu_direct_rdma=True,
        handle_type="posix_fd" if platform.system() != "Windows" else "win32",
        peers=(),
        self_access="rw",
        peer_access="rw",
    )

    vmm_mr = VirtualMemoryResource(device, config=custom_config)

    # Verify configuration is applied
    assert vmm_mr.config == custom_config
    assert vmm_mr.config.gpu_direct_rdma is True
    assert vmm_mr.config.granularity == "minimum"

    # Test allocation with custom config
    buffer = vmm_mr.allocate(8192)
    assert buffer.size >= 8192
    assert buffer.device_id == device.device_id

    # Test policy modification
    new_config = VirtualMemoryResourceOptions(
        allocation_type="pinned",
        location_type="device",
        granularity="recommended",
        gpu_direct_rdma=False,
        handle_type="posix_fd",
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

    This test verifies that VirtualMemoryResource can grow existing
    allocations while preserving the base pointer when possible.
    """
    device = Device()
    device.set_current()

    # Skip if virtual memory management is not supported (we need it for VMM)
    if not device.properties.virtual_memory_management_supported:
        pytest.skip("Virtual memory management is not supported on this device")

    options = VirtualMemoryResourceOptions()

    vmm_mr = VirtualMemoryResource(device, config=options)

    # Create initial allocation
    buffer = vmm_mr.allocate(2 * 1024 * 1024)
    original_size = buffer.size

    # Grow the allocation
    grown_buffer = vmm_mr.modify_allocation(buffer, 4 * 1024 * 1024)

    # Verify growth
    assert grown_buffer.size >= 4 * 1024 * 1024
    assert grown_buffer.size > original_size
    # Because of the slow path, the pointer may change
    # We cannot assert that the new pointer is the same,
    # but we can assert that a new pointer was assigned
    assert grown_buffer.handle is not None

    # Test growing to same size (should return original buffer)
    same_buffer = vmm_mr.modify_allocation(grown_buffer, 4 * 1024 * 1024)
    assert same_buffer.size == grown_buffer.size

    # Test growing to smaller size (should return original buffer)
    smaller_buffer = vmm_mr.modify_allocation(grown_buffer, 2 * 1024 * 1024)
    assert smaller_buffer.size == grown_buffer.size

    # Clean up
    grown_buffer.close()


def test_vmm_allocator_rdma_unsupported_exception():
    """Test that VirtualMemoryResource throws an exception when RDMA is requested but device doesn't support it.

    This test verifies that the VirtualMemoryResource constructor throws a RuntimeError
    when gpu_direct_rdma=True is requested but the device doesn't support virtual memory management.
    """
    device = Device()
    device.set_current()

    # Skip if virtual memory management is not supported (we need it for VMM)
    if not device.properties.virtual_memory_management_supported:
        pytest.skip("Virtual memory management is not supported on this device")

    # Skip if GPU Direct RDMA is supported (we want to test the unsupported case)
    if device.properties.gpu_direct_rdma_supported:
        pytest.skip("This test requires a device that doesn't support GPU Direct RDMA")

    # Test that requesting RDMA on an unsupported device throws an exception
    options = VirtualMemoryResourceOptions(gpu_direct_rdma=True)
    with pytest.raises(RuntimeError, match="GPU Direct RDMA is not supported on this device"):
        VirtualMemoryResource(device, config=options)


def test_mempool(mempool_device):
    device = mempool_device

    # Test basic pool creation
    options = DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=False)
    mr = DeviceMemoryResource(device, options=options)
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
        mr.get_allocation_handle()

    with pytest.raises(RuntimeError, match=ipc_error_msg):
        buffer.get_ipc_descriptor()

    with pytest.raises(RuntimeError, match=ipc_error_msg):
        handle = IPCBufferDescriptor._init(b"", 0)
        Buffer.from_ipc_descriptor(mr, handle)

    buffer.close()


@pytest.mark.parametrize("ipc_enabled", [True, False])
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
def test_mempool_attributes(ipc_enabled, mempool_device, property_name, expected_type):
    """Test all properties of the DeviceMemoryResource class."""
    device = mempool_device
    if platform.system() == "Windows":
        return  # IPC not implemented for Windows

    if ipc_enabled and not supports_ipc_mempool(device):
        pytest.skip("Driver rejects IPC-enabled mempool creation on this platform")

    options = DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=ipc_enabled)
    mr = DeviceMemoryResource(device, options=options)
    assert mr.is_ipc_enabled == ipc_enabled

    # Get the property value
    value = getattr(mr.attributes, property_name)

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
            new_value = getattr(mr.attributes, property_name)
            assert new_value >= initial_value, f"{property_name} should increase or stay same after allocation"
        finally:
            if buffer is not None:
                buffer.close()

    # Test high watermark properties
    if property_name in ["reserved_mem_high", "used_mem_high"]:
        # High watermark should never be less than current
        current_prop = property_name.replace("_high", "_current")
        current_value = getattr(mr.attributes, current_prop)
        assert value >= current_value, f"{property_name} should be >= {current_prop}"


def test_mempool_attributes_ownership(mempool_device):
    """Ensure the attributes bundle handles references correctly."""
    device = mempool_device
    # Skip if IPC mempool is not supported on this platform/device
    if not supports_ipc_mempool(device):
        pytest.skip("Driver rejects IPC-enabled mempool creation on this platform")

    mr = DeviceMemoryResource(device, dict(max_size=POOL_SIZE))
    attributes = mr.attributes
    mr.close()
    del mr

    # After deleting the memory resource, the attributes suite is disconnected.
    with pytest.raises(RuntimeError, match="DeviceMemoryResource is expired"):
        _ = attributes.used_mem_high

    # Even when a new object is created (we found a case where the same
    # mempool handle was really reused).
    mr = DeviceMemoryResource(device, dict(max_size=POOL_SIZE))  # noqa: F841
    with pytest.raises(RuntimeError, match="DeviceMemoryResource is expired"):
        _ = attributes.used_mem_high


# Ensure that memory views dellocate their reference to dlpack tensors
@pytest.mark.skipif(np is None, reason="numpy is not installed")
def test_strided_memory_view_leak():
    arr = np.zeros(1048576, dtype=np.uint8)
    before = sys.getrefcount(arr)
    for idx in range(10):
        StridedMemoryView.from_any_interface(arr, stream_ptr=-1)
    after = sys.getrefcount(arr)
    assert before == after


def test_strided_memory_view_refcnt():
    # Use Fortran ordering so strides is used
    a = np.zeros((64, 4), dtype=np.uint8, order="F")
    av = StridedMemoryView.from_any_interface(a, stream_ptr=-1)
    # segfaults if refcnt is wrong
    assert av.shape[0] == 64
    assert sys.getrefcount(av.shape) >= 2

    assert av.strides[0] == 1
    assert av.strides[1] == 64
    assert sys.getrefcount(av.strides) >= 2
