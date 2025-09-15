# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    from cuda.bindings import driver
except ImportError:
    from cuda import cuda as driver

import ctypes
import multiprocessing

import pytest

from cuda.core.experimental import Buffer, Device, DeviceMemoryResource, IPCChannel, MemoryResource
from cuda.core.experimental._utils.cuda_utils import handle_return

POOL_SIZE = 2097152  # 2MB size
NBYTES = 64


@pytest.fixture(scope="function")
def ipc_device():
    """Obtains a device suitable for IPC-enabled mempool tests, or skips."""
    # Check if IPC is supported on this platform/device
    device = Device()
    device.set_current()

    if not device.properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")

    # Note: Linux specific. Once Windows support for IPC is implemented, this
    # test should be updated.
    if not device.properties.handle_type_posix_file_descriptor_supported:
        pytest.skip("Device does not support IPC")

    return device


def test_ipc_mempool(ipc_device):
    # Set up the IPC-enabled memory pool and share it.
    stream = ipc_device.create_stream()
    mr = DeviceMemoryResource(ipc_device, dict(max_size=POOL_SIZE, ipc_enabled=True))
    assert mr.is_ipc_enabled
    channel = IPCChannel()
    mr.share_to_channel(channel)

    # Start the child process.
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=child_main, args=(channel, queue))
    process.start()

    # Allocate and fill memory.
    buffer = mr.allocate(NBYTES, stream=stream)
    protocol = IPCBufferTestProtocol(ipc_device, buffer, stream=stream)
    protocol.fill_buffer(flipped=False)
    stream.sync()

    # Export the buffer via IPC.
    handle = mr.export_buffer(buffer)
    queue.put(handle)

    # Wait for the child process.
    process.join(timeout=10)
    assert process.exitcode == 0

    # Verify that the buffer was modified.
    protocol.verify_buffer(flipped=True)


def child_main(channel, queue):
    device = Device()
    device.set_current()
    stream = device.create_stream()

    mr = DeviceMemoryResource.from_shared_channel(device, channel)
    handle = queue.get()  # Get exported buffer data
    buffer = mr.import_buffer(handle)

    protocol = IPCBufferTestProtocol(device, buffer, stream=stream)
    protocol.verify_buffer(flipped=False)
    protocol.fill_buffer(flipped=True)
    stream.sync()


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


class IPCBufferTestProtocol:
    """The protocol for verifying IPC.

    Provides methods to fill a buffer with one of two test patterns and verify
    the expected values.
    """

    def __init__(self, device, buffer, nbytes=NBYTES, stream=None):
        self.device = device
        self.buffer = buffer
        self.nbytes = nbytes
        self.stream = stream if stream is not None else device.create_stream()
        self.scratch_buffer = DummyUnifiedMemoryResource(self.device).allocate(self.nbytes, stream=self.stream)

    def fill_buffer(self, flipped=False):
        """Fill a device buffer with test pattern using unified memory."""
        ptr = ctypes.cast(int(self.scratch_buffer.handle), ctypes.POINTER(ctypes.c_byte))
        op = (lambda i: 255 - i) if flipped else (lambda i: i)
        for i in range(self.nbytes):
            ptr[i] = ctypes.c_byte(op(i))
        self.buffer.copy_from(self.scratch_buffer, stream=self.stream)

    def verify_buffer(self, flipped=False):
        """Verify the buffer contents."""
        self.scratch_buffer.copy_from(self.buffer, stream=self.stream)
        self.stream.sync()
        ptr = ctypes.cast(int(self.scratch_buffer.handle), ctypes.POINTER(ctypes.c_byte))
        op = (lambda i: 255 - i) if flipped else (lambda i: i)
        for i in range(self.nbytes):
            assert ctypes.c_byte(ptr[i]).value == ctypes.c_byte(op(i)).value, (
                f"Buffer contains incorrect data at index {i}"
            )
