# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import multiprocessing as mp
import multiprocessing.reduction
import os
import pickle
from contextlib import closing

import pytest
from helpers.buffers import PatternGen
from helpers.child_processes import child_exporter, child_timeout_sec, kill_subprocesses, receive_with_timeout
from helpers.constants import POOL_SIZE
from helpers.contexts import current_context_handle, no_current_context
from helpers.memory import import_pool_resource, is_pool_resource

from cuda.core import (
    Buffer,
    Device,
    DeviceMemoryResource,
    DeviceMemoryResourceOptions,
    PinnedMemoryResource,
    VirtualMemoryResource,
)
from cuda.core._utils.cuda_utils import driver, handle_return

CHILD_TIMEOUT_SEC = child_timeout_sec()
NBYTES = 64

# these tests spawn new processes and files which fails for very many threads;
# child processes and IPC fixtures are shared across workers (#2784); cap at two threads
pytestmark = pytest.mark.parallel_threads_limit(2)


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
        parent_conn.send(type(mr) if is_pool_resource(mr) else None)
        if is_pool_resource(mr):
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
        pool_cls = conn.recv()
        if pool_cls is not None:
            handle = mp.reduction.recv_handle(conn)
            mr = import_pool_resource(pool_cls, device, handle)
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
        if is_pool_resource(mr):
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
        alloc_handle = mr.allocation_handle if is_pool_resource(mr) else None
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

        # Import the allocation again, from the descriptor through the resource
        # rebuilt from the allocation handle. Both buffers alias the same memory.
        buffer2 = Buffer.from_ipc_descriptor(mr2, buffer_desc, stream=stream)

        # Verify initial content
        pgen.verify_buffer(buffer, seed=False)
        pgen.verify_buffer(buffer2, seed=False)

        # Modify the buffer and verify through both imports
        pgen.fill_buffer(buffer, seed=True)
        pgen.verify_buffer(buffer, seed=True)
        pgen.verify_buffer(buffer2, seed=True)

        buffer2.close()
        buffer.close()
        stream.sync()


def _read_buffer(buffer):
    host = (ctypes.c_ubyte * NBYTES)()
    handle_return(driver.cuMemcpyDtoH(ctypes.addressof(host), int(buffer.handle), NBYTES))
    return bytes(host)


def _export_buffer(connection, device_id):
    device = Device(device_id)
    device.set_current()
    options = DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True)
    with (
        connection,
        closing(DeviceMemoryResource(device, options)) as mr,
        mr.allocate(NBYTES, stream=device.default_stream) as buffer,
    ):
        buffer.fill(0x35, stream=device.default_stream)
        device.default_stream.sync()
        # Transfer the MR first so ordinary Buffer pickle can find its UUID.
        connection.send(mr)
        connection.send_bytes(pickle.dumps(buffer))
        assert receive_with_timeout(connection) == "buffer-updated"
        assert _read_buffer(buffer) == b"\x6b" * NBYTES
        connection.send("update-verified")
        assert receive_with_timeout(connection) == "resource-closed"


@pytest.mark.thread_unsafe(reason="spawns an IPC exporter and changes the receiving CUDA context")
@pytest.mark.agent_authored(model="gpt-6-sol")
def test_buffer_unpickle_initializes_missing_current_context(ipc_device):
    """Unpickling without a current context initializes it and preserves shared buffer contents."""
    with child_exporter(_export_buffer, ipc_device.device_id) as connection:
        with closing(receive_with_timeout(connection)) as mapped_mr:
            payload = receive_with_timeout(connection, raw=True)
            original_context = current_context_handle()
            # Deserialize the MR beforehand so only Buffer restoration can initialize the context.
            with no_current_context():
                try:
                    assert current_context_handle() == 0
                    with closing(pickle.loads(payload)) as buffer:  # noqa: S301 - bytes from our own child
                        assert current_context_handle() != 0
                        assert buffer.memory_resource is mapped_mr
                        assert buffer.device_id == ipc_device.device_id
                        assert _read_buffer(buffer) == b"\x35" * NBYTES
                        buffer.fill(0x6B, stream=ipc_device.default_stream)
                        ipc_device.default_stream.sync()
                finally:
                    # no_current_context restores the old stack; pop only the context created here.
                    if current_context_handle():
                        handle_return(driver.cuCtxPopCurrent())
            assert current_context_handle() == original_context
            connection.send("buffer-updated")
            assert receive_with_timeout(connection) == "update-verified"
        connection.send("resource-closed")
