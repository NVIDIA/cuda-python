# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import multiprocessing.reduction
import os

import pytest
from cuda.core import Buffer, Device, DeviceMemoryResource
from helpers.buffers import PatternGen

CHILD_TIMEOUT_SEC = 30
NBYTES = 64
POOL_SIZE = 2097152


class TestObjectSerializationDirect:
    """
    Test the low-level interface for sharing memory resources.

    Send a memory resource over a connection via Python's `send_handle`. Reconstruct
    it on the other end and demonstrate buffer sharing.
    """

    @pytest.mark.flaky(reruns=2)
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
        pgen = PatternGen(device, NBYTES)
        pgen.verify_buffer(buffer1, seed=True)
        pgen.verify_buffer(buffer2, seed=True)
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
        pgen = PatternGen(device, NBYTES)
        pgen.fill_buffer(buffer1, seed=True)
        pgen.fill_buffer(buffer2, seed=True)
        buffer1.close()
        buffer2.close()


class TestObjectSerializationWithMR:
    @pytest.mark.flaky(reruns=2)
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
        pgen = PatternGen(device, NBYTES)
        pgen.verify_buffer(buffer, seed=True)
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
        pgen = PatternGen(device, NBYTES)
        pgen.fill_buffer(buffer, seed=True)
        buffer.close()


class TestObjectPassing:
    """
    Test sending objects as arguments when starting a process.

    True pickling of allocation handles and memory resources is enabled only
    when spawning a process. This is similar to the way sockets and various objects
    in multiprocessing (e.g., Queue) work.
    """

    @pytest.mark.flaky(reruns=2)
    def test_main(self, ipc_device, ipc_memory_resource):
        # Define the objects.
        device = ipc_device
        mr = ipc_memory_resource
        alloc_handle = mr.get_allocation_handle()
        buffer = mr.allocate(NBYTES)
        buffer_desc = buffer.get_ipc_descriptor()

        pgen = PatternGen(device, NBYTES)
        pgen.fill_buffer(buffer, seed=False)

        # Start the child process.
        process = mp.Process(target=self.child_main, args=(alloc_handle, mr, buffer_desc, buffer))
        process.start()
        process.join(timeout=CHILD_TIMEOUT_SEC)
        assert process.exitcode == 0

        pgen.verify_buffer(buffer, seed=True)
        buffer.close()

    def child_main(self, alloc_handle, mr1, buffer_desc, buffer):
        device = Device()
        device.set_current()
        mr2 = DeviceMemoryResource.from_allocation_handle(device, alloc_handle)  # noqa: F841
        pgen = PatternGen(device, NBYTES)

        # Verify initial content
        pgen.verify_buffer(buffer, seed=False)

        # Modify the buffer
        pgen.fill_buffer(buffer, seed=True)

        # Verify modified content
        pgen.verify_buffer(buffer, seed=True)

        # Clean up - only ONE free
        buffer.close()
