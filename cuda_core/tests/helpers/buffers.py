# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes

import pytest

from cuda.core import Buffer, Device, MemoryResource
from cuda.core._stream import Stream_accept
from cuda.core._utils.cuda_utils import driver, handle_return

from . import IS_WINDOWS, IS_WSL, libc

__all__ = [
    "DummyDeviceMemoryResource",
    "DummyHostMemoryResource",
    "DummyUnifiedMemoryResource",
    "NumpyHostMemoryResource",
    "PatternGen",
    "StubMemoryResource",
    "compare_buffer_to_constant",
    "compare_equal_buffers",
    "make_instrumented_memory_resource",
    "make_scratch_buffer",
    "thread_unsafe_on_windows",
]


def thread_unsafe_on_windows(func):
    # Tests that use these buffers and access the memory on the host are
    # thread-unsafe on windows. On windows the GPU must be fully quiescent for host
    # access to be safe and with threaded tests that would require a barrier.
    if IS_WINDOWS or IS_WSL:
        return pytest.mark.thread_unsafe(reason="windows host-access unsafe while GPU is working")(func)
    return func


class StubMemoryResource(MemoryResource):
    """Device-only memory resource for tests that supply a fake pointer."""

    def __init__(self, device):
        self.device = device

    def allocate(self, size, *, stream=None):
        raise NotImplementedError("StubMemoryResource does not allocate")

    def deallocate(self, ptr, size, *, stream=None):
        Stream_accept(stream)

    @property
    def is_device_accessible(self):
        return True

    @property
    def is_host_accessible(self):
        return False

    @property
    def device_id(self):
        return self.device.device_id


def make_instrumented_memory_resource(
    backing=StubMemoryResource,
    *,
    record_streams=False,
    track_active=False,
    deallocate_error=None,
):
    """Return an instrumented backing subclass and its shared telemetry.

    Only calls dispatched through the Python ``allocate`` and ``deallocate``
    methods are observed. Some built-in memory resources free their buffers
    directly in C++ instead (see issue #2615).
    """
    if not isinstance(backing, type) or not issubclass(backing, MemoryResource):
        raise TypeError("backing must be a MemoryResource subclass")

    telemetry = {"active": {}, "deallocations": []}

    class InstrumentedMemoryResource(backing):
        __slots__ = ()

        if track_active:

            def allocate(self, size, *, stream=None):
                buffer = super().allocate(size, stream=stream)
                telemetry["active"][int(buffer.handle)] = size
                return buffer

        if record_streams or track_active or deallocate_error is not None:

            def deallocate(self, ptr, size, *, stream=None):
                if record_streams:
                    telemetry["deallocations"].append({"ptr": int(ptr), "size": size, "stream": stream})
                if deallocate_error is not None:
                    raise deallocate_error
                super().deallocate(ptr, size, stream=stream)
                if track_active:
                    telemetry["active"].pop(int(ptr), None)

    InstrumentedMemoryResource.__name__ = f"Instrumented{backing.__name__}"
    return InstrumentedMemoryResource, telemetry


class DummyDeviceMemoryResource(MemoryResource):
    # cuMemAlloc / cuMemFree are synchronous; stream is accepted for
    # interface conformance but ignored.
    def __init__(self, device):
        self.device = device

    def allocate(self, size, *, stream=None) -> Buffer:
        ptr = handle_return(driver.cuMemAlloc(size))
        return Buffer.from_handle(ptr=ptr, size=size, mr=self)

    def deallocate(self, ptr, size, *, stream=None):
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


class DummyUnifiedMemoryResource(MemoryResource):
    # cuMemAllocManaged / cuMemFree are synchronous; stream is accepted
    # for interface conformance with stream-ordered MRs but ignored.
    def __init__(self, device):
        self.device = device

    def allocate(self, size, *, stream=None) -> Buffer:
        ptr = handle_return(driver.cuMemAllocManaged(size, driver.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL.value))
        return Buffer.from_handle(ptr=ptr, size=size, mr=self)

    def deallocate(self, ptr, size, *, stream=None):
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


class DummyHostMemoryResource(MemoryResource):
    # Pure-host ctypes allocation; stream is accepted for interface
    # conformance but ignored.
    def __init__(self):
        pass

    def allocate(self, size, *, stream=None) -> Buffer:
        # Allocate a ctypes buffer of size `size`
        ptr = (ctypes.c_byte * size)()
        self._ptr = ptr
        return Buffer.from_handle(ptr=ctypes.addressof(ptr), size=size, mr=self)

    def deallocate(self, ptr, size, *, stream=None):
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


class NumpyHostMemoryResource(MemoryResource):
    """Host-only resource backed by ``numpy.empty``, adapted from issue #2769.

    It never touches the CUDA driver, so it must work in a process that has not
    initialized CUDA. ``deallocate`` takes ``stream`` positionally, as the
    reporter's resource does.
    """

    def __init__(self):
        # Strong refs keyed by pointer; Buffer carries only the int address.
        self._held = {}

    def allocate(self, size, *, stream=None) -> Buffer:
        import numpy as np

        arr = np.empty(size, dtype=np.uint8)
        ptr = int(arr.ctypes.data)
        self._held[ptr] = arr
        return Buffer.from_handle(ptr=ptr, size=size, mr=self)

    def deallocate(self, ptr, size, stream=None):
        self._held.pop(int(ptr), None)

    @property
    def is_device_accessible(self) -> bool:
        return False

    @property
    def is_host_accessible(self) -> bool:
        return True

    @property
    def is_managed(self) -> bool:
        return False

    @property
    def device_id(self) -> int:
        return -1


class PatternGen:
    """
    Provides methods to fill a target buffer with  known test patterns and
    verify the expected values.

    Operations are submitted to the supplied stream. Verification synchronizes
    that stream before comparing results on the host.

    The test pattern is either a fixed value or a cyclic pattern generated from
    an 8-bit seed.  Only one of `value` or `seed` should be supplied.

    Distinct test patterns are stored in private buffers called pattern
    buffers. Calls to `fill_buffer` copy from a pattern buffer to the target
    buffer. Calls to `verify_buffer` copy from the target buffer to a scratch
    buffer and then perform a comparison.
    """

    def __init__(self, device, size, *, stream):
        self.device = device
        self.size = size
        self.stream = Stream_accept(stream)
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
        self.stream.sync()
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
