# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import sys

try:
    from cuda.bindings import driver
except ImportError:
    from cuda import cuda as driver

try:
    import numpy as np
except ImportError:
    np = None
import platform
import re

import pytest
from cuda.core import (
    Buffer,
    Device,
    DeviceMemoryResource,
    DeviceMemoryResourceOptions,
    GraphMemoryResource,
    ManagedMemoryResource,
    ManagedMemoryResourceOptions,
    MemoryResource,
    PinnedMemoryResource,
    PinnedMemoryResourceOptions,
    VirtualMemoryResource,
    VirtualMemoryResourceOptions,
)
from cuda.core import (
    system as ccx_system,
)
from cuda.core._dlpack import DLDeviceType
from cuda.core._memory import IPCBufferDescriptor
from cuda.core._utils.cuda_utils import CUDAError, handle_return
from cuda.core.utils import StridedMemoryView
from helpers import IS_WINDOWS
from helpers.buffers import DummyUnifiedMemoryResource

from conftest import (
    create_managed_memory_resource_or_skip,
    skip_if_managed_memory_unsupported,
    skip_if_pinned_memory_unsupported,
)
from cuda_python_test_helpers import supports_ipc_mempool

POOL_SIZE = 2097152  # 2MB size


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
        "GraphMemoryResource",
        "IPCBufferDescriptor",
        "IPCAllocationHandle",
        "LegacyPinnedMemoryResource",
        "ManagedMemoryResource",
        "ManagedMemoryResourceOptions",
        "PinnedMemoryResourceOptions",
        "PinnedMemoryResource",
        "VirtualMemoryResourceOptions",
        "VirtualMemoryResource",
    ]
    d = {}
    exec("from cuda.core._memory import *", d)  # noqa: S102
    d = {k: v for k, v in d.items() if not k.startswith("__")}
    assert sorted(expected) == sorted(d.keys())


def buffer_initialization(dummy_mr: MemoryResource):
    buffer = dummy_mr.allocate(size=1024)
    assert buffer.handle != 0
    assert buffer.size == 1024
    assert buffer.memory_resource == dummy_mr
    assert buffer.is_device_accessible == dummy_mr.is_device_accessible
    assert buffer.is_host_accessible == dummy_mr.is_host_accessible
    assert not buffer.is_mapped
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


def _bytes_repeat(pattern: bytes, size: int) -> bytes:
    assert len(pattern) > 0
    assert size % len(pattern) == 0
    return pattern * (size // len(pattern))


def _pattern_bytes(value) -> bytes:
    if isinstance(value, int):
        return bytes([value])
    return bytes(memoryview(value).cast("B"))


@pytest.fixture(params=["device", "unified", "pinned"])
def fill_env(request):
    device = Device()
    device.set_current()
    if request.param == "device":
        mr = DummyDeviceMemoryResource(device)
    elif request.param == "unified":
        mr = DummyUnifiedMemoryResource(device)
    else:
        mr = DummyPinnedMemoryResource(device)
    return device, mr


_FILL_SIZE = 64  # Keep small; divisible by 1/2/4.

_FILL_CASES = [
    # int -> 1-byte pattern
    pytest.param(0x42, _FILL_SIZE, None, id="int-0x42"),
    pytest.param(-1, _FILL_SIZE, OverflowError, id="int-neg"),
    pytest.param(256, _FILL_SIZE, OverflowError, id="int-256"),
    pytest.param(1000, _FILL_SIZE, OverflowError, id="int-1000"),
    # bad type
    pytest.param("invalid", _FILL_SIZE, TypeError, id="bad-type-str"),
    # bytes-like patterns
    pytest.param(b"\x7f", _FILL_SIZE, None, id="bytes-1"),
    pytest.param(b"\x34\x12", _FILL_SIZE, None, id="bytes-2"),
    pytest.param(b"\xef\xbe\xad\xde", _FILL_SIZE, None, id="bytes-4"),
    pytest.param(b"\x34\x12", _FILL_SIZE + 1, ValueError, id="bytes-2-bad-size"),
    pytest.param(b"\xef\xbe\xad\xde", _FILL_SIZE + 2, ValueError, id="bytes-4-bad-size"),
    pytest.param(b"", _FILL_SIZE, ValueError, id="bytes-0"),
    pytest.param(b"\x01\x02\x03", _FILL_SIZE, ValueError, id="bytes-3"),
]

if np is not None:
    _FILL_CASES.extend(
        [
            # 8-bit patterns
            pytest.param(np.uint8(0), _FILL_SIZE, None, id="np-uint8-0"),
            pytest.param(np.uint8(255), _FILL_SIZE, None, id="np-uint8-255"),
            pytest.param(np.int8(-1), _FILL_SIZE, None, id="np-int8--1"),
            pytest.param(np.int8(127), _FILL_SIZE, None, id="np-int8-127"),
            pytest.param(np.int8(-128), _FILL_SIZE, None, id="np-int8--128"),
            # 16-bit patterns
            pytest.param(np.uint16(0x1234), _FILL_SIZE, None, id="np-uint16-0x1234"),
            pytest.param(np.uint16(0xFFFF), _FILL_SIZE, None, id="np-uint16-0xFFFF"),
            pytest.param(np.int16(-1), _FILL_SIZE, None, id="np-int16--1"),
            pytest.param(np.int16(32767), _FILL_SIZE, None, id="np-int16-max"),
            pytest.param(np.int16(-32768), _FILL_SIZE, None, id="np-int16-min"),
            pytest.param(np.uint16(0x1234), _FILL_SIZE + 1, ValueError, id="np-uint16-bad-size"),
            # 32-bit patterns
            pytest.param(np.uint32(0xDEADBEEF), _FILL_SIZE, None, id="np-uint32-0xDEADBEEF"),
            pytest.param(np.uint32(0xFFFFFFFF), _FILL_SIZE, None, id="np-uint32-0xFFFFFFFF"),
            pytest.param(np.int32(-1), _FILL_SIZE, None, id="np-int32--1"),
            pytest.param(np.int32(2147483647), _FILL_SIZE, None, id="np-int32-max"),
            pytest.param(np.int32(-2147483648), _FILL_SIZE, None, id="np-int32-min"),
            pytest.param(np.uint32(0xDEADBEEF), _FILL_SIZE + 2, ValueError, id="np-uint32-bad-size"),
            # float32 (bit-pattern fill)
            pytest.param(np.float32(1.0), _FILL_SIZE, None, id="np-float32-1.0"),
            # 64-bit patterns should error (8-byte pattern)
            pytest.param(np.uint64(0), _FILL_SIZE, ValueError, id="np-uint64-err"),
            pytest.param(np.int64(0), _FILL_SIZE, ValueError, id="np-int64-err"),
            pytest.param(np.float64(0), _FILL_SIZE, ValueError, id="np-float64-err"),
        ]
    )


@pytest.mark.parametrize("value,size,exc", _FILL_CASES)
def test_buffer_fill(fill_env, value, size, exc):
    device, mr = fill_env
    stream = device.create_stream()
    buffer = mr.allocate(size=size)
    try:
        if exc is not None:
            with pytest.raises(exc):
                buffer.fill(value, stream=stream)
            return

        buffer.fill(value, stream=stream)
        device.sync()

        # Verify contents only for host-accessible buffers.
        if buffer.is_host_accessible:
            pat = _pattern_bytes(value)
            got = ctypes.string_at(int(buffer.handle), size)
            assert got == _bytes_repeat(pat, size)
    finally:
        buffer.close()


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


def test_buffer_external_host():
    a = (ctypes.c_byte * 20)()
    ptr = ctypes.addressof(a)
    buffer = Buffer.from_handle(ptr, 20, owner=a)
    assert not buffer.is_device_accessible
    assert buffer.is_host_accessible
    assert buffer.device_id == -1
    buffer.close()


@pytest.mark.parametrize("change_device", [True, False])
def test_buffer_external_device(change_device):
    n = ccx_system.num_devices
    if n < 1:
        pytest.skip("No devices found")
    dev_id = n - 1
    d = Device(dev_id)
    d.set_current()
    buffer_ = d.allocate(size=32)

    if change_device:
        # let's switch to a different device if possibe
        # to make sure we get the original device id
        d = Device(0)
        d.set_current()

    buffer = Buffer.from_handle(int(buffer_.handle), 32)
    assert buffer.is_device_accessible
    assert not buffer.is_host_accessible
    assert buffer.device_id == dev_id
    buffer.close()
    buffer_.close()


@pytest.mark.parametrize("change_device", [True, False])
def test_buffer_external_pinned_alloc(change_device):
    n = ccx_system.num_devices
    if n < 1:
        pytest.skip("No devices found")
    dev_id = n - 1
    d = Device(dev_id)
    d.set_current()
    mr = DummyPinnedMemoryResource(d)
    buffer_ = mr.allocate(size=32)

    if change_device:
        # let's switch to a different device if possibe
        # to make sure we get the original device id
        d = Device(0)
        d.set_current()

    buffer = Buffer.from_handle(int(buffer_.handle), 32)
    assert buffer.is_device_accessible
    assert buffer.is_host_accessible
    assert buffer.device_id == dev_id
    buffer.close()
    buffer_.close()


@pytest.mark.parametrize("change_device", [True, False])
def test_buffer_external_pinned_registered(change_device):
    n = ccx_system.num_devices
    if n < 1:
        pytest.skip("No devices found")
    dev_id = n - 1
    d = Device(dev_id)
    d.set_current()
    a = (ctypes.c_byte * 20)()
    ptr = ctypes.addressof(a)

    buffer = Buffer.from_handle(ptr, 20, owner=ptr)
    assert not buffer.is_device_accessible
    assert buffer.is_host_accessible
    assert buffer.device_id == -1

    handle_return(driver.cuMemHostRegister(ptr, 20, 0))
    try:
        if change_device:
            # let's switch to a different device if possibe
            # to make sure we get the original device id
            d = Device(0)
            d.set_current()

        buffer = Buffer.from_handle(ptr, 20, owner=ptr)
        assert buffer.is_device_accessible
        assert buffer.is_host_accessible
        assert buffer.device_id == dev_id
        buffer.close()
    finally:
        handle_return(driver.cuMemHostUnregister(ptr))


@pytest.mark.parametrize("change_device", [True, False])
def test_buffer_external_managed(change_device):
    n = ccx_system.num_devices
    if n < 1:
        pytest.skip("No devices found")
    dev_id = n - 1
    d = Device(dev_id)
    d.set_current()
    ptr = None
    try:
        ptr = handle_return(driver.cuMemAllocManaged(32, driver.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL.value))
        if change_device:
            # let's switch to a different device if possibe
            # to make sure we get the original device id
            d = Device(0)
            d.set_current()
        buffer = Buffer.from_handle(ptr, 32)
        assert buffer.is_device_accessible
        assert buffer.is_host_accessible
        assert buffer.device_id == dev_id
    finally:
        if ptr is not None:
            handle_return(driver.cuMemFree(ptr))


def test_memory_resource_and_owner_disallowed():
    with pytest.raises(ValueError, match="cannot be both specified together"):
        a = (ctypes.c_byte * 20)()
        ptr = ctypes.addressof(a)
        Buffer.from_handle(ptr, 20, mr=DummyDeviceMemoryResource(Device()), owner=a)


def test_owner_close():
    a = (ctypes.c_byte * 20)()
    ptr = ctypes.addressof(a)
    before = sys.getrefcount(a)
    buffer = Buffer.from_handle(ptr, 20, owner=a)
    assert sys.getrefcount(a) != before
    buffer.close()
    after = sys.getrefcount(a)
    assert after == before


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


def test_buffer_dlpack_failure_clean_up():
    dummy_mr = NullMemoryResource()
    buffer = dummy_mr.allocate(size=1024)
    before = sys.getrefcount(buffer)
    with pytest.raises(BufferError, match="invalid buffer"):
        buffer.__dlpack__()
    after = sys.getrefcount(buffer)
    # we use the buffer refcount as sentinel for proper clean-up here,
    # hoping that malloc and frees did the right thing
    # as they are handled by the same deleter
    assert after == before


@pytest.mark.parametrize("use_device_object", [True, False])
def test_device_memory_resource_initialization(use_device_object):
    """Test that DeviceMemoryResource can be initialized successfully.

    This test verifies that the DeviceMemoryResource initializes properly,
    including the release threshold configuration for performance optimization.
    """
    device = Device()

    if not device.properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")

    device.set_current()

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


def test_pinned_memory_resource_initialization(init_cuda):
    device = Device()
    skip_if_pinned_memory_unsupported(device)

    device.set_current()

    mr = PinnedMemoryResource()
    assert mr.is_device_accessible
    assert mr.is_host_accessible

    # Test allocation/deallocation works
    buffer = mr.allocate(1024)
    assert buffer.size == 1024
    assert buffer.device_id == -1  # Not bound to any GPU
    assert buffer.is_host_accessible
    assert buffer.memory_resource == mr
    assert buffer.is_device_accessible
    buffer.close()


def test_managed_memory_resource_initialization(init_cuda):
    device = Device()
    skip_if_managed_memory_unsupported(device)

    device.set_current()

    mr = create_managed_memory_resource_or_skip()
    assert mr.is_device_accessible
    assert mr.is_host_accessible

    # Test allocation/deallocation works
    buffer = mr.allocate(1024)
    assert buffer.size == 1024
    assert buffer.is_host_accessible  # But accessible from host
    assert buffer.memory_resource == mr
    assert buffer.is_device_accessible
    buffer.close()


def get_handle_type():
    if IS_WINDOWS:
        return (("win32", None), ("win32_kmt", None))
    else:
        return (("posix_fd", None),)


@pytest.mark.parametrize("use_device_object", [True, False])
@pytest.mark.parametrize("handle_type", get_handle_type())
def test_vmm_allocator_basic_allocation(use_device_object, handle_type):
    """Test basic VMM allocation functionality.

    This test verifies that VirtualMemoryResource can allocate memory
    using CUDA VMM APIs with default configuration.
    """
    device = Device()
    device.set_current()

    # Skip if virtual memory management is not supported
    if not device.properties.virtual_memory_management_supported:
        pytest.skip("Virtual memory management is not supported on this device")

    handle_type, security_attribute = handle_type  # unpack
    options = VirtualMemoryResourceOptions(handle_type=handle_type)
    # Create VMM allocator with default config
    device_arg = device if use_device_object else device.device_id
    vmm_mr = VirtualMemoryResource(device_arg, config=options)

    # Test basic allocation
    try:
        buffer = vmm_mr.allocate(4096)
    except NotImplementedError:
        assert handle_type == "win32"
        return
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

    # Skip if GPU Direct RDMA is not supported
    if not device.properties.gpu_direct_rdma_supported:
        pytest.skip("This test requires a device that supports GPU Direct RDMA")

    # Test with custom VMM config
    custom_config = VirtualMemoryResourceOptions(
        allocation_type="pinned",
        location_type="device",
        granularity="minimum",
        gpu_direct_rdma=True,
        handle_type="posix_fd" if not IS_WINDOWS else "win32_kmt",
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
    try:
        buffer = vmm_mr.allocate(8192)
    except CUDAError as exc:
        msg = str(exc)
        if "CUDA_ERROR_INVALID_DEVICE" in msg:
            pytest.xfail("TODO(#1300): Failing on Jetson AGX Orin P3730")
        raise
    assert buffer.size >= 8192
    assert buffer.device_id == device.device_id

    # Test policy modification
    new_config = VirtualMemoryResourceOptions(
        allocation_type="pinned",
        location_type="device",
        granularity="recommended",
        gpu_direct_rdma=False,
        handle_type="posix_fd" if not IS_WINDOWS else "win32_kmt",
        peers=(),
        self_access="r",  # Read-only access
        peer_access="r",
    )

    # Modify allocation policy
    try:
        modified_buffer = vmm_mr.modify_allocation(buffer, 16384, config=new_config)
    except CUDAError as exc:
        msg = str(exc)
        if "CUDA_ERROR_UNKNOWN" in msg:
            pytest.xfail("TODO(#1300): Known to fail already with CTK 13.0 (Windows)")
        raise
    assert modified_buffer.size >= 16384
    assert vmm_mr.config == new_config
    assert vmm_mr.config.self_access == "r"

    # Clean up
    modified_buffer.close()


@pytest.mark.parametrize("handle_type", get_handle_type())
def test_vmm_allocator_grow_allocation(handle_type):
    """Test VMM allocator's ability to grow existing allocations.

    This test verifies that VirtualMemoryResource can grow existing
    allocations while preserving the base pointer when possible.
    """
    device = Device()
    device.set_current()

    # Skip if virtual memory management is not supported (we need it for VMM)
    if not device.properties.virtual_memory_management_supported:
        pytest.skip("Virtual memory management is not supported on this device")

    handle_type, security_attribute = handle_type  # unpack
    options = VirtualMemoryResourceOptions(handle_type=handle_type)
    vmm_mr = VirtualMemoryResource(device, config=options)

    # Create initial allocation
    try:
        buffer = vmm_mr.allocate(2 * 1024 * 1024)
    except NotImplementedError:
        assert handle_type == "win32"
        return
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


def test_device_memory_resource_with_options(init_cuda):
    device = Device()
    if not device.properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")

    device.set_current()

    # Test basic pool creation
    options = DeviceMemoryResourceOptions(max_size=POOL_SIZE)
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
    buffer.close(stream)

    # Test memory copying between buffers from same pool
    src_buffer = mr.allocate(64)
    dst_buffer = mr.allocate(64)
    stream = device.create_stream()
    src_buffer.copy_to(dst_buffer, stream=stream)
    device.sync()
    dst_buffer.close()
    src_buffer.close()


def test_pinned_memory_resource_with_options(init_cuda):
    device = Device()
    skip_if_pinned_memory_unsupported(device)

    device.set_current()

    # Test basic pool creation
    options = PinnedMemoryResourceOptions(max_size=POOL_SIZE)
    mr = PinnedMemoryResource(options)
    assert mr.device_id == -1  # Not bound to any GPU
    assert mr.is_device_accessible
    assert mr.is_host_accessible
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
    buffer.close(stream)

    # Test memory copying between buffers from same pool
    src_buffer = mr.allocate(64)
    dst_buffer = mr.allocate(64)
    stream = device.create_stream()
    src_buffer.copy_to(dst_buffer, stream=stream)
    device.sync()
    dst_buffer.close()
    src_buffer.close()


def test_managed_memory_resource_with_options(init_cuda):
    device = Device()
    skip_if_managed_memory_unsupported(device)

    device.set_current()

    # Test basic pool creation
    options = ManagedMemoryResourceOptions()
    mr = create_managed_memory_resource_or_skip(options)
    assert mr.is_device_accessible
    assert mr.is_host_accessible
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
    buffer.close(stream)

    # Test memory copying between buffers from same pool
    src_buffer = mr.allocate(64)
    dst_buffer = mr.allocate(64)
    stream = device.create_stream()
    src_buffer.copy_to(dst_buffer, stream=stream)
    device.sync()
    dst_buffer.close()
    src_buffer.close()


def test_mempool_ipc_errors(mempool_device):
    """Test error cases when IPC operations are disabled."""
    device = mempool_device
    options = DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=False)
    mr = DeviceMemoryResource(device, options=options)
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


def test_pinned_mempool_ipc_basic():
    """Test basic IPC functionality for PinnedMemoryResource."""
    device = Device()
    device.set_current()

    skip_if_pinned_memory_unsupported(device)

    if platform.system() == "Windows":
        pytest.skip("IPC not implemented for Windows")

    if not supports_ipc_mempool(device):
        pytest.skip("Driver rejects IPC-enabled mempool creation on this platform")

    # Test IPC-enabled PinnedMemoryResource creation
    options = PinnedMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True)
    mr = PinnedMemoryResource(options)
    assert mr.is_ipc_enabled
    assert mr.is_device_accessible
    assert mr.is_host_accessible
    assert mr.device_id == 0  # IPC-enabled uses location id 0

    # Test allocation handle export
    alloc_handle = mr.get_allocation_handle()
    assert alloc_handle is not None

    # Test buffer allocation
    buffer = mr.allocate(1024)
    assert buffer.size == 1024
    assert buffer.is_device_accessible
    assert buffer.is_host_accessible

    # Test IPC descriptor
    ipc_desc = buffer.get_ipc_descriptor()
    assert ipc_desc is not None
    assert ipc_desc.size == 1024

    buffer.close()
    mr.close()


def test_pinned_mempool_ipc_errors():
    """Test error cases when IPC operations are disabled for PinnedMemoryResource."""
    device = Device()
    device.set_current()

    skip_if_pinned_memory_unsupported(device)

    # Test with IPC disabled (default)
    options = PinnedMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=False)
    mr = PinnedMemoryResource(options)
    assert not mr.is_ipc_enabled
    assert mr.device_id == -1  # Non-IPC uses location id -1

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
    mr.close()


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
def test_mempool_attributes(ipc_enabled, memory_resource_factory, property_name, expected_type):
    """Test all properties of memory pool attributes for all memory resource types."""
    MR, MRops = memory_resource_factory
    device = Device()

    if MR is DeviceMemoryResource and not device.properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")
    elif MR is PinnedMemoryResource:
        skip_if_pinned_memory_unsupported(device)
    elif MR is ManagedMemoryResource:
        skip_if_managed_memory_unsupported(device)

    # ManagedMemoryResource does not support IPC
    if MR is ManagedMemoryResource and ipc_enabled:
        pytest.skip(f"{MR.__name__} does not support IPC")

    device.set_current()

    if platform.system() == "Windows":
        return  # IPC not implemented for Windows

    if ipc_enabled and not supports_ipc_mempool(device):
        pytest.skip("Driver rejects IPC-enabled mempool creation on this platform")

    if MR is DeviceMemoryResource:
        options = MRops(max_size=POOL_SIZE, ipc_enabled=ipc_enabled)
        mr = MR(device, options=options)
        assert mr.is_ipc_enabled == ipc_enabled
    elif MR is PinnedMemoryResource:
        options = MRops(max_size=POOL_SIZE, ipc_enabled=ipc_enabled)
        mr = MR(options)
        assert mr.is_ipc_enabled == ipc_enabled
    elif MR is ManagedMemoryResource:
        options = MRops()
        mr = create_managed_memory_resource_or_skip(options)
        assert not mr.is_ipc_enabled

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


def test_mempool_attributes_repr(memory_resource_factory):
    """Test the repr of memory pool attributes for all memory resource types."""
    MR, MRops = memory_resource_factory
    device = Device()

    if MR is DeviceMemoryResource and not device.properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")
    elif MR is PinnedMemoryResource:
        skip_if_pinned_memory_unsupported(device)
    elif MR is ManagedMemoryResource:
        skip_if_managed_memory_unsupported(device)

    device.set_current()

    if MR is DeviceMemoryResource:
        mr = MR(device, options={"max_size": 2048})
    elif MR is PinnedMemoryResource:
        mr = MR(options={"max_size": 2048})
    elif MR is ManagedMemoryResource:
        mr = create_managed_memory_resource_or_skip(options={})

    buffer1 = mr.allocate(64)
    buffer2 = mr.allocate(64)
    buffer1.close()
    assert re.match(
        r".*Attributes\(release_threshold=\d+, reserved_mem_current=\d+, reserved_mem_high=\d+, "
        r"reuse_allow_internal_dependencies=(True|False), reuse_allow_opportunistic=(True|False), "
        r"reuse_follow_event_dependencies=(True|False), used_mem_current=\d+, used_mem_high=\d+\)",
        str(mr.attributes),
    )
    buffer2.close()


def test_mempool_attributes_ownership(memory_resource_factory):
    """Ensure the attributes bundle handles references correctly for all memory resource types."""
    MR, MRops = memory_resource_factory
    device = Device()

    if MR is DeviceMemoryResource and not device.properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")
    elif MR is PinnedMemoryResource:
        skip_if_pinned_memory_unsupported(device)
    elif MR is ManagedMemoryResource:
        skip_if_managed_memory_unsupported(device)

    # Skip if IPC mempool is not supported on this platform/device (only relevant for DeviceMemoryResource)
    if MR is DeviceMemoryResource and not supports_ipc_mempool(device):
        pytest.skip("Driver rejects IPC-enabled mempool creation on this platform")

    device.set_current()

    if MR is DeviceMemoryResource:
        mr = MR(device, dict(max_size=POOL_SIZE))
    elif MR is PinnedMemoryResource:
        mr = MR(dict(max_size=POOL_SIZE))
    elif MR is ManagedMemoryResource:
        mr = create_managed_memory_resource_or_skip(dict())

    attributes = mr.attributes
    mr.close()
    del mr

    # After deleting the memory resource, the attributes suite is disconnected.
    with pytest.raises(RuntimeError, match="is expired"):
        _ = attributes.used_mem_high

    # Even when a new object is created (we found a case where the same
    # mempool handle was really reused).
    if MR is DeviceMemoryResource:
        mr = MR(device, dict(max_size=POOL_SIZE))  # noqa: F841
    elif MR is PinnedMemoryResource:
        mr = MR(dict(max_size=POOL_SIZE))  # noqa: F841
    elif MR is ManagedMemoryResource:
        mr = create_managed_memory_resource_or_skip(dict())  # noqa: F841

    with pytest.raises(RuntimeError, match="is expired"):
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


def test_graph_memory_resource_object(init_cuda):
    device = Device()
    gmr1 = GraphMemoryResource(device)
    gmr2 = GraphMemoryResource(device)
    gmr3 = GraphMemoryResource(device.device_id)

    # These objects are interned.
    assert gmr1 is gmr2 is gmr3
    assert gmr1 == gmr2 == gmr3
