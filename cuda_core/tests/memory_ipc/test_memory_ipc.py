# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp

from cuda.core.experimental import Buffer, DeviceMemoryResource
from utility import IPCBufferTestHelper

CHILD_TIMEOUT_SEC = 20
NBYTES = 64
NWORKERS = 2
NTASKS = 2


class TestIpcMempool:
    def test_main(self, ipc_device, ipc_memory_resource):
        """Test IPC with memory pools."""
        # Set up the IPC-enabled memory pool and share it.
        device = ipc_device
        mr = ipc_memory_resource

        # Start the child process.
        queue = mp.Queue()
        process = mp.Process(target=self.child_main, args=(device, mr, queue))
        process.start()

        # Allocate and fill memory.
        buffer = mr.allocate(NBYTES)
        helper = IPCBufferTestHelper(device, buffer)
        helper.fill_buffer(flipped=False)

        # Export the buffer via IPC.
        queue.put(buffer)

        # Wait for the child process.
        process.join(timeout=CHILD_TIMEOUT_SEC)
        assert process.exitcode == 0

        # Verify that the buffer was modified.
        helper.verify_buffer(flipped=True)

    def child_main(self, device, mr, queue):
        device.set_current()
        buffer = queue.get(timeout=CHILD_TIMEOUT_SEC)
        helper = IPCBufferTestHelper(device, buffer)
        helper.verify_buffer(flipped=False)
        helper.fill_buffer(flipped=True)


class TestIPCMempoolMultiple:
    def test_main(self, ipc_device, ipc_memory_resource):
        """Test IPC with memory pools using multiple processes."""
        # Construct an IPC-enabled memory resource and share it with two children.
        device = ipc_device
        mr = ipc_memory_resource
        q1, q2 = (mp.Queue() for _ in range(2))

        # Allocate memory buffers and export them to each child.
        buffer1 = mr.allocate(NBYTES)
        q1.put(buffer1)
        q2.put(buffer1)
        buffer2 = mr.allocate(NBYTES)
        q1.put(buffer2)
        q2.put(buffer2)

        # Start the child processes.
        p1 = mp.Process(target=self.child_main, args=(device, mr, 1, q1))
        p2 = mp.Process(target=self.child_main, args=(device, mr, 2, q2))
        p1.start()
        p2.start()

        # Wait for the child processes.
        p1.join(timeout=CHILD_TIMEOUT_SEC)
        p2.join(timeout=CHILD_TIMEOUT_SEC)
        assert p1.exitcode == 0
        assert p2.exitcode == 0

        # Verify that the buffers were modified.
        IPCBufferTestHelper(device, buffer1).verify_buffer(flipped=False)
        IPCBufferTestHelper(device, buffer2).verify_buffer(flipped=True)

    def child_main(self, device, mr, idx, queue):
        # Note: passing the mr registers it so that buffers can be passed
        # directly.
        device.set_current()
        buffer1 = queue.get(timeout=CHILD_TIMEOUT_SEC)
        buffer2 = queue.get(timeout=CHILD_TIMEOUT_SEC)
        if idx == 1:
            IPCBufferTestHelper(device, buffer1).fill_buffer(flipped=False)
        elif idx == 2:
            IPCBufferTestHelper(device, buffer2).fill_buffer(flipped=True)


class TestIPCSharedAllocationHandleAndBufferDescriptors:
    def test_main(self, ipc_device, ipc_memory_resource):
        """
        Demonstrate that a memory pool allocation handle can be reused for IPC
        with multiple processes. Uses buffer descriptors.
        """
        # Set up the IPC-enabled memory pool and share it using one handle.
        device = ipc_device
        mr = ipc_memory_resource
        alloc_handle = mr.get_allocation_handle()

        # Start children.
        q1, q2 = (mp.Queue() for _ in range(2))
        p1 = mp.Process(target=self.child_main, args=(device, alloc_handle, 1, q1))
        p2 = mp.Process(target=self.child_main, args=(device, alloc_handle, 2, q2))
        p1.start()
        p2.start()

        # Allocate and share memory.
        buf1 = mr.allocate(NBYTES)
        buf2 = mr.allocate(NBYTES)
        q1.put(buf1.get_ipc_descriptor())
        q2.put(buf2.get_ipc_descriptor())

        # Wait for children.
        p1.join(timeout=CHILD_TIMEOUT_SEC)
        p2.join(timeout=CHILD_TIMEOUT_SEC)
        assert p1.exitcode == 0
        assert p2.exitcode == 0

        # Verify results.
        IPCBufferTestHelper(device, buf1).verify_buffer(starting_from=1)
        IPCBufferTestHelper(device, buf2).verify_buffer(starting_from=2)

    def child_main(self, device, alloc_handle, idx, queue):
        """Fills a shared memory buffer."""
        # In this case, the device needs to be set up (passing the mr does it
        # implicitly in other tests).
        device.set_current()
        mr = DeviceMemoryResource.from_allocation_handle(device, alloc_handle)
        buffer_descriptor = queue.get(timeout=CHILD_TIMEOUT_SEC)
        buffer = Buffer.from_ipc_descriptor(mr, buffer_descriptor)
        IPCBufferTestHelper(device, buffer).fill_buffer(starting_from=idx)


class TestIPCSharedAllocationHandleAndBufferObjects:
    def test_main(self, ipc_device, ipc_memory_resource):
        """
        Demonstrate that a memory pool allocation handle can be reused for IPC
        with multiple processes. Uses buffer objects (not descriptors).
        """
        device = ipc_device
        mr = ipc_memory_resource
        alloc_handle = mr.get_allocation_handle()

        # Start children.
        q1, q2 = (mp.Queue() for _ in range(2))
        p1 = mp.Process(target=self.child_main, args=(device, alloc_handle, 1, q1))
        p2 = mp.Process(target=self.child_main, args=(device, alloc_handle, 2, q2))
        p1.start()
        p2.start()

        # Allocate and share memory.
        buf1 = mr.allocate(NBYTES)
        buf2 = mr.allocate(NBYTES)
        q1.put(buf1)
        q2.put(buf2)

        # Wait for children.
        p1.join(timeout=CHILD_TIMEOUT_SEC)
        p2.join(timeout=CHILD_TIMEOUT_SEC)
        assert p1.exitcode == 0
        assert p2.exitcode == 0

        # Verify results.
        IPCBufferTestHelper(device, buf1).verify_buffer(starting_from=1)
        IPCBufferTestHelper(device, buf2).verify_buffer(starting_from=2)

    def child_main(self, device, alloc_handle, idx, queue):
        """Fills a shared memory buffer."""
        device.set_current()

        # Register the memory resource.
        DeviceMemoryResource.from_allocation_handle(device, alloc_handle)

        # Now get buffers.
        buffer = queue.get(timeout=CHILD_TIMEOUT_SEC)
        IPCBufferTestHelper(device, buffer).fill_buffer(starting_from=idx)
