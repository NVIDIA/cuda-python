# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp

import pytest
from cuda.core import Buffer, DeviceMemoryResource
from helpers.buffers import PatternGen

CHILD_TIMEOUT_SEC = 30
NBYTES = 64
NWORKERS = 2
NTASKS = 2


class TestIpcMempool:
    @pytest.mark.flaky(reruns=2)
    def test_main(self, ipc_device, ipc_memory_resource):
        """Test IPC with memory pools."""
        # Set up the IPC-enabled memory pool and share it.
        device = ipc_device
        mr = ipc_memory_resource
        assert not mr.is_mapped
        pgen = PatternGen(device, NBYTES)

        # Start the child process.
        queue = mp.Queue()
        process = mp.Process(target=self.child_main, args=(device, mr, queue))
        process.start()

        # Allocate and fill memory.
        buffer = mr.allocate(NBYTES)
        assert not buffer.is_mapped
        pgen.fill_buffer(buffer, seed=False)

        # Export the buffer via IPC.
        queue.put(buffer)

        # Wait for the child process.
        process.join(timeout=CHILD_TIMEOUT_SEC)
        assert process.exitcode == 0

        # Verify that the buffer was modified.
        pgen.verify_buffer(buffer, seed=True)
        buffer.close()

    def child_main(self, device, mr, queue):
        device.set_current()
        assert mr.is_mapped
        buffer = queue.get(timeout=CHILD_TIMEOUT_SEC)
        assert buffer.is_mapped
        pgen = PatternGen(device, NBYTES)
        pgen.verify_buffer(buffer, seed=False)
        pgen.fill_buffer(buffer, seed=True)
        buffer.close()


class TestIPCMempoolMultiple:
    @pytest.mark.flaky(reruns=2)
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
        pgen = PatternGen(device, NBYTES)
        pgen.verify_buffer(buffer1, seed=1)
        pgen.verify_buffer(buffer2, seed=2)
        buffer1.close()
        buffer2.close()

    def child_main(self, device, mr, seed, queue):
        # Note: passing the mr registers it so that buffers can be passed
        # directly.
        device.set_current()
        buffer1 = queue.get(timeout=CHILD_TIMEOUT_SEC)
        buffer2 = queue.get(timeout=CHILD_TIMEOUT_SEC)
        pgen = PatternGen(device, NBYTES)
        if seed == 1:
            pgen.fill_buffer(buffer1, seed=1)
        elif seed == 2:
            pgen.fill_buffer(buffer2, seed=2)
        buffer1.close()
        buffer2.close()


class TestIPCSharedAllocationHandleAndBufferDescriptors:
    @pytest.mark.flaky(reruns=2)
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
        p1 = mp.Process(target=self.child_main, args=(device, alloc_handle, False, q1))
        p2 = mp.Process(target=self.child_main, args=(device, alloc_handle, True, q2))
        p1.start()
        p2.start()

        # Allocate and share memory.
        buffer1 = mr.allocate(NBYTES)
        buffer2 = mr.allocate(NBYTES)
        q1.put(buffer1.get_ipc_descriptor())
        q2.put(buffer2.get_ipc_descriptor())

        # Wait for children.
        p1.join(timeout=CHILD_TIMEOUT_SEC)
        p2.join(timeout=CHILD_TIMEOUT_SEC)
        assert p1.exitcode == 0
        assert p2.exitcode == 0

        # Verify results.
        pgen = PatternGen(device, NBYTES)
        pgen.verify_buffer(buffer1, seed=False)
        pgen.verify_buffer(buffer2, seed=True)
        buffer1.close()
        buffer2.close()

    def child_main(self, device, alloc_handle, seed, queue):
        """Fills a shared memory buffer."""
        # In this case, the device needs to be set up (passing the mr does it
        # implicitly in other tests).
        device.set_current()
        mr = DeviceMemoryResource.from_allocation_handle(device, alloc_handle)
        buffer_descriptor = queue.get(timeout=CHILD_TIMEOUT_SEC)
        buffer = Buffer.from_ipc_descriptor(mr, buffer_descriptor)
        pgen = PatternGen(device, NBYTES)
        pgen.fill_buffer(buffer, seed=seed)
        buffer.close()


class TestIPCSharedAllocationHandleAndBufferObjects:
    @pytest.mark.flaky(reruns=2)
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
        p1 = mp.Process(target=self.child_main, args=(device, alloc_handle, False, q1))
        p2 = mp.Process(target=self.child_main, args=(device, alloc_handle, True, q2))
        p1.start()
        p2.start()

        # Allocate and share memory.
        buffer1 = mr.allocate(NBYTES)
        buffer2 = mr.allocate(NBYTES)
        q1.put(buffer1)
        q2.put(buffer2)

        # Wait for children.
        p1.join(timeout=CHILD_TIMEOUT_SEC)
        p2.join(timeout=CHILD_TIMEOUT_SEC)
        assert p1.exitcode == 0
        assert p2.exitcode == 0

        # Verify results.
        pgen = PatternGen(device, NBYTES)
        pgen.verify_buffer(buffer1, seed=False)
        pgen.verify_buffer(buffer2, seed=True)
        buffer1.close()
        buffer2.close()

    def child_main(self, device, alloc_handle, seed, queue):
        """Fills a shared memory buffer."""
        device.set_current()

        # Register the memory resource.
        DeviceMemoryResource.from_allocation_handle(device, alloc_handle)

        # Now get buffers.
        buffer = queue.get(timeout=CHILD_TIMEOUT_SEC)
        pgen = PatternGen(device, NBYTES)
        pgen.fill_buffer(buffer, seed=seed)
        buffer.close()
