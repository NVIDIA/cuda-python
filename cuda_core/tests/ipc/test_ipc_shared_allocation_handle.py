# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.core.experimental import Buffer, Device, DeviceMemoryResource, IPCChannel
from utility import IPCBufferTestHelper
import multiprocessing
import pytest
from itertools import cycle

CHILD_TIMEOUT_SEC = 10
NBYTES = 64
NWORKERS = 2
NTASKS = 2

def test_ipc_shared_allocation_handle(device, ipc_memory_resource):
    """Demonstrate that a memory pool allocation handle can be reused for IPC
    with multiple processes."""
    # Set up communication.
    ch1 = IPCChannel()
    ch2 = IPCChannel()
    q1 = multiprocessing.Queue()
    q2 = multiprocessing.Queue()

    # Start children.
    p1 = multiprocessing.Process(target=child_main, args=(1, ch1, q1))
    p2 = multiprocessing.Process(target=child_main, args=(2, ch2, q2))
    p1.start()
    p2.start()

    # Set up the IPC-enabled memory pool and share it using one handle.
    mr = ipc_memory_resource
    alloc_handle = mr.get_allocation_handle()
    ch1.send_allocation_handle(alloc_handle)
    ch2.send_allocation_handle(alloc_handle)

    # Allocate a share memory.
    buf1 = mr.allocate(NBYTES)
    buf2 = mr.allocate(NBYTES)
    q1.put(buf1.export())
    q2.put(buf2.export())

    # Wait for children.
    p1.join(timeout=CHILD_TIMEOUT_SEC)
    p2.join(timeout=CHILD_TIMEOUT_SEC)
    assert p1.exitcode == 0
    assert p2.exitcode == 0

    # Verify results.
    IPCBufferTestHelper(device, buf1).verify_buffer(starting_from=1)
    IPCBufferTestHelper(device, buf2).verify_buffer(starting_from=2)


def child_main(idx, channel, queue):
    """Fills a shared memory buffer."""
    device = Device()
    device.set_current()
    alloc_handle = channel.receive_allocation_handle()
    mr = DeviceMemoryResource.from_allocation_handle(device, alloc_handle)
    buffer_descriptor = queue.get()
    buffer = Buffer.import_(mr, buffer_descriptor)
    IPCBufferTestHelper(device, buffer).fill_buffer(starting_from=idx)


def test_ipc_shared_allocation_handle2(device, ipc_memory_resource):
    """Demonstrate that a memory pool allocation handle can be reused for IPC
    with multiple processes (simplified)."""
    # Set up communication.
    ch1 = IPCChannel()
    ch2 = IPCChannel()

    # Start children.
    p1 = multiprocessing.Process(target=child_main2, args=(1, ch1))
    p2 = multiprocessing.Process(target=child_main2, args=(2, ch2))
    p1.start()
    p2.start()

    # Set up the IPC-enabled memory pool and share it using one handle.
    mr = ipc_memory_resource
    alloc_handle = mr.get_allocation_handle()
    ch1.send_allocation_handle(alloc_handle)
    ch2.send_allocation_handle(alloc_handle)

    # Allocate a share memory.
    buf1 = mr.allocate(NBYTES)
    buf2 = mr.allocate(NBYTES)
    ch1.send_buffer(buf1)
    ch2.send_buffer(buf2)

    # Wait for children.
    p1.join(timeout=CHILD_TIMEOUT_SEC)
    p2.join(timeout=CHILD_TIMEOUT_SEC)
    assert p1.exitcode == 0
    assert p2.exitcode == 0

    # Verify results.
    IPCBufferTestHelper(device, buf1).verify_buffer(starting_from=1)
    IPCBufferTestHelper(device, buf2).verify_buffer(starting_from=2)


def child_main2(idx, channel):
    """Fills a shared memory buffer."""
    device = Device()
    device.set_current()
    buffer = channel.receive_buffer()
    IPCBufferTestHelper(device, buffer).fill_buffer(starting_from=idx)

