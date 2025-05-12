# Copyright 2024 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    from cuda.bindings import driver
except ImportError:
    from cuda import cuda as driver

import ctypes

import pytest

from cuda.core.experimental import Device
from cuda.core.experimental._memory import Buffer, DLDeviceType, MemoryResource
from cuda.core.experimental._utils.cuda_utils import handle_return


class DummyDeviceMemoryResource(MemoryResource):
    def __init__(self, device):
        self.device = device

    def allocate(self, size, stream=None) -> Buffer:
        ptr = handle_return(driver.cuMemAlloc(size))
        return Buffer(ptr=ptr, size=size, mr=self)

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
        return Buffer(ptr=ptr, size=size, mr=self)

    def deallocate(self, ptr, size, stream=None):
        # the memory is deallocated per the ctypes deallocation at garbage collection time
        pass

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
        return Buffer(ptr=ptr, size=size, mr=self)

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
        return Buffer(ptr=ptr, size=size, mr=self)

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
        buffer.__dlpack__(dl_device=[])
    with pytest.raises(BufferError, match=r"^Sorry, not supported: copy=True$"):
        buffer.__dlpack__(copy=True)
    with pytest.raises(BufferError, match=r"^Expected max_version Tuple\[int, int\], got \[\]$"):
        buffer.__dlpack__(max_version=[])
    with pytest.raises(BufferError, match=r"^Expected max_version Tuple\[int, int\], got \(9, 8, 7\)$"):
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
