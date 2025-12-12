# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import multiprocessing.reduction
import os

import pytest
from cuda.core.experimental import Buffer, Device, DeviceMemoryResource
from helpers.buffers import PatternGen

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

    def test_main(self, ipc_device, ipc_memory_resource):
        # TODO: This test fails with PinnedMR due to CUDA_ERROR_ALREADY_MAPPED.
        # When buffer1 is passed as an argument, it's serialized and mapped into
        # the child process. Then trying to recreate it from descriptor causes
        # "already mapped" error. This might be a test design issue or a real
        # difference in how PMR vs DMR handle double-mapping. Needs investigation.
        from cuda.core.experimental import PinnedMemoryResource

        if isinstance(ipc_memory_resource, PinnedMemoryResource):
            pytest.skip(
                "TestObjectPassing temporarily skipped for PinnedMR (TODO: investigate CUDA_ERROR_ALREADY_MAPPED)"
            )

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

    def child_main(self, alloc_handle, mr1, buffer_desc, buffer1):
        device = Device()
        device.set_current()

        # Recreate MR from allocation handle using the same type as mr1
        # For DMR, we need to pass device; for PMR, we don't
        from cuda.core.experimental import DeviceMemoryResource

        if type(mr1) is DeviceMemoryResource:
            mr2 = type(mr1).from_allocation_handle(device, alloc_handle)
        else:
            mr2 = type(mr1).from_allocation_handle(alloc_handle)

        pgen = PatternGen(device, NBYTES)

        # OK to build the buffer from either mr and the descriptor.
        # All buffer* objects point to the same memory.
        buffer2 = Buffer.from_ipc_descriptor(mr1, buffer_desc)
        buffer3 = Buffer.from_ipc_descriptor(mr2, buffer_desc)

        pgen.verify_buffer(buffer1, seed=False)
        pgen.verify_buffer(buffer2, seed=False)
        pgen.verify_buffer(buffer3, seed=False)

        # Modify 1.
        pgen.fill_buffer(buffer1, seed=True)

        pgen.verify_buffer(buffer1, seed=True)
        pgen.verify_buffer(buffer2, seed=True)
        pgen.verify_buffer(buffer3, seed=True)

        # Modify 2.
        pgen.fill_buffer(buffer2, seed=False)

        pgen.verify_buffer(buffer1, seed=False)
        pgen.verify_buffer(buffer2, seed=False)
        pgen.verify_buffer(buffer3, seed=False)

        # Modify 3.
        pgen.fill_buffer(buffer3, seed=True)

        pgen.verify_buffer(buffer1, seed=True)
        pgen.verify_buffer(buffer2, seed=True)
        pgen.verify_buffer(buffer3, seed=True)

        # Close any one buffer.
        buffer1.close()
