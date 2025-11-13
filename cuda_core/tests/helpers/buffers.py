# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes

from cuda.core.experimental import Buffer, Device, MemoryResource
from cuda.core.experimental._utils.cuda_utils import driver, handle_return

from . import libc

__all__ = [
    "compare_buffer_to_constant",
    "compare_equal_buffers",
    "DummyUnifiedMemoryResource",
    "make_scratch_buffer",
    "PatternGen",
]


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
        return self.device


class PatternGen:
    """
    Provides methods to fill a target buffer with  known test patterns and
    verify the expected values.

    If a stream is provided, operations are synchronized with respect to that
    stream.  Otherwise, they are synchronized over the device.

    The test pattern is either a fixed value or a cyclic pattern generated from
    an 8-bit seed.  Only one of `value` or `seed` should be supplied.

    Distinct test patterns are stored in private buffers called pattern
    buffers. Calls to `fill_buffer` copy from a pattern buffer to the target
    buffer. Calls to `verify_buffer` copy from the target buffer to a scratch
    buffer and then perform a comparison.
    """

    def __init__(self, device, size, stream=None):
        self.device = device
        self.size = size
        self.stream = stream if stream is not None else device.create_stream()
        self.sync_target = stream if stream is not None else device
        self.pattern_buffers = {}

    def fill_buffer(self, buffer, seed=None, value=None):
        """Fill a device buffer with a sequential test pattern using unified memory."""
        assert buffer.size == self.size
        pattern_buffer = self._get_pattern_buffer(seed, value)
        buffer.copy_from(pattern_buffer, stream=self.stream)

    def verify_buffer(self, buffer, seed=None, value=None):
        """Verify the buffer contents against a sequential pattern."""
        assert buffer.size == self.size
        scratch_buffer = DummyUnifiedMemoryResource(self.device).allocate(self.size)
        ptr_test = self._ptr(scratch_buffer)
        pattern_buffer = self._get_pattern_buffer(seed, value)
        ptr_expected = self._ptr(pattern_buffer)
        scratch_buffer.copy_from(buffer, stream=self.stream)
        self.sync_target.sync()
        assert libc.memcmp(ptr_test, ptr_expected, self.size) == 0

    @staticmethod
    def _ptr(buffer):
        """Get a pointer to the specified buffer."""
        return ctypes.cast(int(buffer.handle), ctypes.POINTER(ctypes.c_ubyte))

    def _get_pattern_buffer(self, seed, value):
        """Get a buffer holding the specified test pattern."""
        assert seed is None or value is None
        if value is None:
            seed = (0 if seed is None else seed) & 0xFF
        key = seed, value
        pattern_buffer = self.pattern_buffers.get(key, None)
        if pattern_buffer is None:
            if value is not None:
                pattern_buffer = make_scratch_buffer(self.device, value, self.size)
            else:
                pattern_buffer = DummyUnifiedMemoryResource(self.device).allocate(self.size)
                ptr = self._ptr(pattern_buffer)
                for i in range(self.size):
                    ptr[i] = (seed + i) & 0xFF
            self.pattern_buffers[key] = pattern_buffer
        return pattern_buffer


def make_scratch_buffer(device, value, nbytes):
    """Create a unified memory buffer with the specified value."""
    buffer = DummyUnifiedMemoryResource(device).allocate(nbytes)
    set_buffer(buffer, value)
    return buffer


def set_buffer(buffer, value):
    assert 0 <= int(value) < 256
    ptr = ctypes.cast(int(buffer.handle), ctypes.POINTER(ctypes.c_byte))
    ctypes.memset(ptr, value & 0xFF, buffer.size)


def compare_equal_buffers(buffer1, buffer2):
    """Compare the contents of two host-accessible buffers for bitwise equality."""
    if buffer1.size != buffer2.size:
        return False
    ptr1 = ctypes.cast(int(buffer1.handle), ctypes.POINTER(ctypes.c_byte))
    ptr2 = ctypes.cast(int(buffer2.handle), ctypes.POINTER(ctypes.c_byte))
    return libc.memcmp(ptr1, ptr2, buffer1.size) == 0


def compare_buffer_to_constant(buffer, value):
    device_id = buffer.memory_resource.device_id
    device = Device(device_id)
    stream = device.create_stream()
    expected = make_scratch_buffer(device, value, buffer.size)
    tmp = make_scratch_buffer(device, 0, buffer.size)
    tmp.copy_from(buffer, stream=stream)
    stream.sync()
    result = compare_equal_buffers(expected, tmp)
    expected.close()
    tmp.close()
    return result
