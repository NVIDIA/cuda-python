# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes

from cuda.core.experimental import Buffer, MemoryResource
from cuda.core.experimental._utils.cuda_utils import driver, handle_return


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
    """A helper for manipulating memory buffers in IPC tests.

    Provides methods to fill a buffer with one of two test patterns and verify
    the expected values.
    """

    def __init__(self, device, buffer):
        self.device = device
        self.buffer = buffer
        self.scratch_buffer = DummyUnifiedMemoryResource(self.device).allocate(self.buffer.size)
        self.stream = device.create_stream()

    def fill_buffer(self, flipped=False, starting_from=0):
        """Fill a device buffer with test pattern using unified memory."""
        ptr = ctypes.cast(int(self.scratch_buffer.handle), ctypes.POINTER(ctypes.c_byte))
        op = (lambda i: 255 - i) if flipped else (lambda i: i)
        for i in range(self.buffer.size):
            ptr[i] = ctypes.c_byte(op(starting_from + i))
        self.buffer.copy_from(self.scratch_buffer, stream=self.stream)
        self.device.sync()

    def verify_buffer(self, flipped=False, starting_from=0):
        """Verify the buffer contents."""
        self.scratch_buffer.copy_from(self.buffer, stream=self.stream)
        self.device.sync()
        ptr = ctypes.cast(int(self.scratch_buffer.handle), ctypes.POINTER(ctypes.c_byte))
        op = (lambda i: 255 - i) if flipped else (lambda i: i)
        for i in range(self.buffer.size):
            assert ctypes.c_byte(ptr[i]).value == ctypes.c_byte(op(starting_from + i)).value, (
                f"Buffer contains incorrect data at index {i}"
            )
