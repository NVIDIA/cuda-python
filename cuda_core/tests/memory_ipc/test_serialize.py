# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import multiprocessing.reduction
import os

import pytest
from helpers.buffers import PatternGen
from helpers.child_processes import child_timeout_sec, kill_subprocesses

from cuda.core import Buffer, Device, DeviceMemoryResource, PinnedMemoryResource, VirtualMemoryResource

CHILD_TIMEOUT_SEC = child_timeout_sec()
NBYTES = 64

# these tests spawn new processes and files which fails for very many threads
pytestmark = pytest.mark.parallel_threads_limit(4)


def _is_pool(mr):
    """Whether ``mr`` has the pool half of the IPC protocol (allocation handle, registry, uuid)."""
    return isinstance(mr, (DeviceMemoryResource, PinnedMemoryResource))


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
        stream = device.default_stream

        # Start the child process.
        parent_conn, child_conn = mp.Pipe()
        process = mp.Process(target=self.child_main, args=(child_conn,))
        process.start()

        # Send a memory resource: a pool by its allocation handle over the raw
        # fd channel, a VirtualMemoryResource as the object itself (it holds no
        # driver object; a VMM descriptor carries its fds per allocation).
        parent_conn.send(_is_pool(mr))
        if _is_pool(mr):
            alloc_handle = mr.allocation_handle
            mp.reduction.send_handle(parent_conn, alloc_handle.handle, process.pid)
        else:
            parent_conn.send(mr)

        # Send a buffer.
        buffer1 = mr.allocate(NBYTES, stream=stream)
        parent_conn.send(buffer1)  # directly

        buffer2 = mr.allocate(NBYTES, stream=stream)
        stream.sync()
        parent_conn.send(buffer2.ipc_descriptor)  # by descriptor

        # Wait for the child process.
        process.join(timeout=CHILD_TIMEOUT_SEC)
        survivors = kill_subprocesses(process)
        assert not survivors, "child did not exit within timeout"
        assert process.exitcode == 0

        # Confirm buffers were modified.
        pgen = PatternGen(device, NBYTES, stream=stream)
        pgen.verify_buffer(buffer1, seed=True)
        pgen.verify_buffer(buffer2, seed=True)
        buffer1.close()
        buffer2.close()
        stream.sync()

    def child_main(self, conn):
        # Set up the device.
        device = Device()
        device.set_current()

        # Receive the memory resource.
        if conn.recv():  # pool
            handle = mp.reduction.recv_handle(conn)
            mr = DeviceMemoryResource.from_allocation_handle(device, handle)
            os.close(handle)
        else:
            mr = conn.recv()
            assert isinstance(mr, VirtualMemoryResource)

        # Receive the buffers.
        buffer1 = conn.recv()  # directly
        buffer_desc = conn.recv()
        stream = device.default_stream
        buffer2 = Buffer.from_ipc_descriptor(mr, buffer_desc, stream=stream)  # by descriptor

        # Modify the buffers.
        pgen = PatternGen(device, NBYTES, stream=stream)
        pgen.fill_buffer(buffer1, seed=True)
        pgen.fill_buffer(buffer2, seed=True)
        buffer1.close()
        buffer2.close()
        stream.sync()


class TestObjectSerializationWithMR:
    @pytest.mark.flaky(reruns=2)
    def test_main(self, ipc_device, ipc_memory_resource):
        """Test sending IPC memory objects to a child through a queue."""
        device = ipc_device
        mr = ipc_memory_resource
        stream = device.default_stream

        # Start the child process. Sending the memory resource registers it so
        # that buffers can be handled automatically.
        pipe = [mp.Queue() for _ in range(2)]
        process = mp.Process(target=self.child_main, args=(pipe, mr))
        process.start()

        # Send a memory resource directly. This relies on the mr already
        # being passed when spawning the child.
        pipe[0].put(mr)
        uuid = pipe[1].get(timeout=CHILD_TIMEOUT_SEC)
        assert uuid == getattr(mr, "uuid", None)

        # Send a buffer.
        buffer = mr.allocate(NBYTES, stream=stream)
        stream.sync()
        pipe[0].put(buffer)

        # Wait for the child process.
        process.join(timeout=CHILD_TIMEOUT_SEC)
        survivors = kill_subprocesses(process)
        assert not survivors, "child did not exit within timeout"
        assert process.exitcode == 0

        # Confirm buffer was modified.
        pgen = PatternGen(device, NBYTES, stream=stream)
        pgen.verify_buffer(buffer, seed=True)
        buffer.close()
        stream.sync()

    def child_main(self, pipe, _):
        device = Device()
        device.set_current()

        # Memory resource.
        mr = pipe[0].get(timeout=CHILD_TIMEOUT_SEC)
        pipe[1].put(getattr(mr, "uuid", None))

        # Buffer.
        buffer = pipe[0].get(timeout=CHILD_TIMEOUT_SEC)
        if _is_pool(mr):
            assert buffer.memory_resource.handle == mr.handle
        else:
            # A VMM resource pickles as (device, options); the buffer's rebuilt
            # resource is equivalent, not identical.
            assert buffer.memory_resource.config == mr.config
        stream = device.default_stream
        pgen = PatternGen(device, NBYTES, stream=stream)
        pgen.fill_buffer(buffer, seed=True)
        buffer.close()
        stream.sync()


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
        alloc_handle = mr.allocation_handle if _is_pool(mr) else None
        stream = device.default_stream
        buffer = mr.allocate(NBYTES, stream=stream)
        buffer_desc = buffer.ipc_descriptor

        pgen = PatternGen(device, NBYTES, stream=stream)
        pgen.fill_buffer(buffer, seed=False)
        stream.sync()

        # Start the child process.
        process = mp.Process(target=self.child_main, args=(alloc_handle, mr, buffer_desc, buffer))
        process.start()
        process.join(timeout=CHILD_TIMEOUT_SEC)
        survivors = kill_subprocesses(process)
        assert not survivors, "child did not exit within timeout"
        assert process.exitcode == 0

        pgen.verify_buffer(buffer, seed=True)
        buffer.close()
        stream.sync()

    def child_main(self, alloc_handle, mr1, buffer_desc, buffer):
        device = Device()
        device.set_current()
        if isinstance(mr1, PinnedMemoryResource):
            with pytest.raises(TypeError):
                DeviceMemoryResource.from_allocation_handle(device, alloc_handle)
            mr2 = PinnedMemoryResource.from_allocation_handle(alloc_handle)
        elif isinstance(mr1, DeviceMemoryResource):
            with pytest.raises(TypeError):
                PinnedMemoryResource.from_allocation_handle(alloc_handle)
            mr2 = DeviceMemoryResource.from_allocation_handle(device, alloc_handle)
        else:
            # A VirtualMemoryResource has no allocation handle; the resource
            # argument itself is the rebuilt resource.
            assert alloc_handle is None
            assert isinstance(mr1, VirtualMemoryResource)
            mr2 = mr1
        stream = device.default_stream
        pgen = PatternGen(device, NBYTES, stream=stream)

        # Verify initial content
        pgen.verify_buffer(buffer, seed=False)

        # Modify the buffer
        pgen.fill_buffer(buffer, seed=True)

        # Verify modified content
        pgen.verify_buffer(buffer, seed=True)

        # Clean up - only ONE free
        buffer.close()
        stream.sync()
