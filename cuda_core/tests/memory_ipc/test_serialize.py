# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import multiprocessing.reduction
import os

from cuda.core.experimental import Buffer, Device, DeviceMemoryResource
from utility import IPCBufferTestHelper

CHILD_TIMEOUT_SEC = 20
NBYTES = 64
POOL_SIZE = 2097152


class TestObjectSerializationDirect:
    """
    Test the low-level interface for sharing memory resources.

    Send a memory resource over a connection via Python's `send_handle`. Reconstruct
    it on the other end and demonstrate buffer sharing.
    """

    def test_main(self, ipc_device, ipc_memory_resource):
        device = ipc_device
        mr = ipc_memory_resource

        # Start the child process.
        parent_conn, child_conn = mp.Pipe()
        process = mp.Process(target=self.child_main, args=(child_conn,))
        process.start()

        # Send a memory resource by allocation handle.
        alloc_handle = mr.get_allocation_handle()
        mp.reduction.send_handle(parent_conn, alloc_handle.handle, process.pid)

        # Send a buffer.
        buffer1 = mr.allocate(NBYTES)
        parent_conn.send(buffer1)  # directly

        buffer2 = mr.allocate(NBYTES)
        parent_conn.send(buffer2.get_ipc_descriptor())  # by descriptor

        # Wait for the child process.
        process.join(timeout=CHILD_TIMEOUT_SEC)
        assert process.exitcode == 0

        # Confirm buffers were modified.
        IPCBufferTestHelper(device, buffer1).verify_buffer(flipped=True)
        IPCBufferTestHelper(device, buffer2).verify_buffer(flipped=True)
        buffer1.close()
        buffer2.close()

    def child_main(self, conn):
        # Set up the device.
        device = Device()
        device.set_current()

        # Receive the memory resource.
        handle = mp.reduction.recv_handle(conn)
        mr = DeviceMemoryResource.from_allocation_handle(device, handle)
        os.close(handle)

        # Receive the buffers.
        buffer1 = conn.recv()  # directly
        buffer_desc = conn.recv()
        buffer2 = Buffer.from_ipc_descriptor(mr, buffer_desc)  # by descriptor

        # Modify the buffers.
        IPCBufferTestHelper(device, buffer1).fill_buffer(flipped=True)
        IPCBufferTestHelper(device, buffer2).fill_buffer(flipped=True)
        buffer1.close()
        buffer2.close()


class TestObjectSerializationWithMR:
    def test_main(self, ipc_device, ipc_memory_resource):
        """Test sending IPC memory objects to a child through a queue."""
        device = ipc_device
        mr = ipc_memory_resource

        # Start the child process. Sending the memory resource registers it so
        # that buffers can be handled automatically.
        pipe = [mp.Queue() for _ in range(2)]
        process = mp.Process(target=self.child_main, args=(pipe, mr))
        process.start()

        # Send a memory resource directly. This relies on the mr already
        # being passed when spawning the child.
        pipe[0].put(mr)
        uuid = pipe[1].get(timeout=CHILD_TIMEOUT_SEC)
        assert uuid == mr.uuid

        # Send a buffer.
        buffer = mr.allocate(NBYTES)
        pipe[0].put(buffer)

        # Wait for the child process.
        process.join(timeout=CHILD_TIMEOUT_SEC)
        assert process.exitcode == 0

        # Confirm buffer was modified.
        IPCBufferTestHelper(device, buffer).verify_buffer(flipped=True)
        buffer.close()

    def child_main(self, pipe, _):
        device = Device()
        device.set_current()

        # Memory resource.
        mr = pipe[0].get(timeout=CHILD_TIMEOUT_SEC)
        pipe[1].put(mr.uuid)

        # Buffer.
        buffer = pipe[0].get(timeout=CHILD_TIMEOUT_SEC)
        assert buffer.memory_resource.handle == mr.handle
        IPCBufferTestHelper(device, buffer).fill_buffer(flipped=True)
        buffer.close()


def test_object_passing(ipc_device, ipc_memory_resource):
    """
    Test sending objects as arguments when starting a process.

    True pickling of allocation handles and memory resources is enabled only
    when spawning a process. This is similar to the way sockets and various objects
    in multiprocessing (e.g., Queue) work.
    """

    # Define the objects.
    device = ipc_device
    mr = ipc_memory_resource
    alloc_handle = mr.get_allocation_handle()
    buffer = mr.allocate(NBYTES)
    buffer_desc = buffer.get_ipc_descriptor()

    helper = IPCBufferTestHelper(device, buffer)
    helper.fill_buffer(flipped=False)

    # Start the child process.
    process = mp.Process(target=child_main, args=(alloc_handle, mr, buffer_desc, buffer))
    process.start()
    process.join(timeout=CHILD_TIMEOUT_SEC)
    assert process.exitcode == 0

    helper.verify_buffer(flipped=True)
    buffer.close()


def child_main(alloc_handle, mr1, buffer_desc, buffer1):
    device = Device()
    device.set_current()
    mr2 = DeviceMemoryResource.from_allocation_handle(device, alloc_handle)

    # OK to build the buffer from either mr and the descriptor.
    # All buffer* objects point to the same memory.
    buffer2 = Buffer.from_ipc_descriptor(mr1, buffer_desc)
    buffer3 = Buffer.from_ipc_descriptor(mr2, buffer_desc)

    helper1 = IPCBufferTestHelper(device, buffer1)
    helper2 = IPCBufferTestHelper(device, buffer2)
    helper3 = IPCBufferTestHelper(device, buffer3)

    helper1.verify_buffer(flipped=False)
    helper2.verify_buffer(flipped=False)
    helper3.verify_buffer(flipped=False)

    # Modify 1.
    helper1.fill_buffer(flipped=True)

    helper1.verify_buffer(flipped=True)
    helper2.verify_buffer(flipped=True)
    helper3.verify_buffer(flipped=True)

    # Modify 2.
    helper2.fill_buffer(flipped=False)

    helper1.verify_buffer(flipped=False)
    helper2.verify_buffer(flipped=False)
    helper3.verify_buffer(flipped=False)

    # Modify 3.
    helper3.fill_buffer(flipped=True)

    helper1.verify_buffer(flipped=True)
    helper2.verify_buffer(flipped=True)
    helper3.verify_buffer(flipped=True)

    # Close any one buffer.
    buffer1.close()
