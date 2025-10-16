# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import sys
import time

from cuda.core.experimental import Buffer, MemoryResource
from cuda.core.experimental._utils.cuda_utils import driver, handle_return

if sys.platform.startswith("win"):
    libc = ctypes.CDLL("msvcrt.dll")
else:
    libc = ctypes.CDLL("libc.so.6")


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


class IPCBufferTestHelper:
    """
    A helper for manipulating memory buffers in IPC tests.

    Provides methods to fill a target buffer with a known test pattern and
    verify the expected values.

    If a stream is provided, operations are synchronized with respect to that
    stream.  Otherwise, they are synchronized over the device.

    The test pattern is either a fixed value or a cyclic function of the byte
    offset. The cyclic pattern is based on two arguments:

        `flipped`: whether to flip (invert) each byte.
        `starting_from`: an offset to start counting from.

    For a fixed pattern, specify `value`. This supercedes the above arguments.

    Distinct test patterns are stored in separate buffers called pattern
    buffers. Calls to `fill_buffer` copy from a pattern buffer to the target
    buffer. Calls to `verify_buffer` copy from the target buffer to a scratch
    buffer and then perform a comparison.
    """

    def __init__(self, device, buffer, stream=None):
        self.device = device
        self.buffer = buffer
        self.stream = stream if stream is not None else device.create_stream()
        self.sync_target = stream if stream is not None else device
        self.scratch_buffer = DummyUnifiedMemoryResource(self.device).allocate(self.buffer.size)
        self.pattern_buffers = {}

    @property
    def size(self):
        """The buffer size in bytes."""
        return self.buffer.size

    def fill_buffer(self, flipped=False, starting_from=0, value=None, repeat=1, sync=True):
        """Fill a device buffer with a sequential test pattern using unified memory."""
        pattern_buffer = self._get_pattern_buffer(flipped, starting_from, value)
        for _ in range(repeat):
            self.buffer.copy_from(pattern_buffer, stream=self.stream)
        if sync:
            self.sync()

    def verify_buffer(self, flipped=False, starting_from=0, value=None, repeat=1):
        """Verify the buffer contents against a sequential pattern."""
        ptr_test = self._ptr(self.scratch_buffer)
        pattern_buffer = self._get_pattern_buffer(flipped, starting_from, value)
        ptr_expected = self._ptr(pattern_buffer)
        for _ in range(repeat):
            self.scratch_buffer.copy_from(self.buffer, stream=self.stream)
            self.sync()
            assert libc.memcmp(ptr_test, ptr_expected, self.size) == 0

    def sync(self):
        """Synchronize against the sync target (a stream or device)."""
        self.sync_target.sync()

    @staticmethod
    def _ptr(buffer):
        """Get a pointer to the specified buffer."""
        return ctypes.cast(int(buffer.handle), ctypes.POINTER(ctypes.c_ubyte))

    def _get_pattern_buffer(self, flipped, starting_from, value):
        """Get a buffer holding the specified test pattern."""
        assert value is None or (not flipped and starting_from == 0)
        key = (value & 0xFF,) if value is not None else (flipped, starting_from)
        pattern_buffer = self.pattern_buffers.get(key, None)
        if pattern_buffer is None:
            if value is not None:
                pattern_buffer = make_scratch_buffer(self.device, value, self.size)
            else:
                pattern_buffer = DummyUnifiedMemoryResource(self.device).allocate(self.size)
                ptr = self._ptr(pattern_buffer)
                pattern = lambda i: (starting_from + i) & 0xFF  # noqa: E731
                if flipped:
                    for i in range(self.size):
                        ptr[i] = ~pattern(i)
                else:
                    for i in range(self.size):
                        ptr[i] = pattern(i)
            self.pattern_buffers[key] = pattern_buffer
        return pattern_buffer


def make_scratch_buffer(device, value, nbytes):
    """Create a unified memory buffer with the specified value."""
    buffer = DummyUnifiedMemoryResource(device).allocate(nbytes)
    ptr = ctypes.cast(int(buffer.handle), ctypes.POINTER(ctypes.c_byte))
    ctypes.memset(ptr, value & 0xFF, nbytes)
    return buffer


def compare_buffers(buffer1, buffer2):
    """Compare the contents of two host-accessible buffers a la memcmp."""
    assert buffer1.size == buffer2.size
    ptr1 = ctypes.cast(int(buffer1.handle), ctypes.POINTER(ctypes.c_byte))
    ptr2 = ctypes.cast(int(buffer2.handle), ctypes.POINTER(ctypes.c_byte))
    return libc.memcmp(ptr1, ptr2, buffer1.size)


class TimestampedLogger:
    """
    A logger that prefixes each output with a timestamp, containing the elapsed
    time since the logger was created.

    Example:

        import multiprocess as mp
        import time

        def main():
            log = TimestampedLogger(prefix="parent: ")
            log("begin")
            process = mp.Process(target=child_main, args=(log,))
            process.start()
            process.join()
            log("done")

        def child_main(log):
            log.prefix = " child: "
            log("begin")
            time.sleep(1)
            log("done")

        if __name__ == "__main__":
            main()

    Possible output:

        [     0.003 ms] parent: begin
        [   819.464 ms]  child: begin
        [  1819.666 ms]  child: done
        [  1882.954 ms] parent: done
    """

    def __init__(self, prefix=None, start_time=None, enabled=True):
        self.prefix = "" if prefix is None else prefix
        self.start_time = start_time if start_time is not None else time.time_ns()
        self.enabled = enabled

    def __call__(self, msg):
        if self.enabled:
            now = (time.time_ns() - self.start_time) * 1e-6
            print(f"[{now:>10.3f} ms] {self.prefix}{msg}")
