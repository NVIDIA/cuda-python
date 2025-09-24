# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing

from utility import IPCBufferTestHelper

from cuda.core.experimental import Buffer, Device, DeviceMemoryResource, IPCChannel

CHILD_TIMEOUT_SEC = 4
NBYTES = 64
NWORKERS = 2
NTASKS = 2


def test_ipc_mempool(device, ipc_memory_resource):
    """Test IPC with memory pools."""
    # Set up the IPC-enabled memory pool and share it.
    mr = ipc_memory_resource
    channel = mr.create_ipc_channel()

    # Start the child process.
    process = multiprocessing.Process(target=child_main1, args=(channel,))
    process.start()

    # Allocate and fill memory.
    buffer = mr.allocate(NBYTES)
    helper = IPCBufferTestHelper(device, buffer)
    helper.fill_buffer(flipped=False)

    # Export the buffer via IPC.
    channel.send_buffer(buffer)

    # Wait for the child process.
    process.join(timeout=CHILD_TIMEOUT_SEC)
    assert process.exitcode == 0

    # Verify that the buffer was modified.
    helper.verify_buffer(flipped=True)


def child_main1(channel):
    device = Device()
    device.set_current()
    buffer = channel.receive_buffer()
    helper = IPCBufferTestHelper(device, buffer)
    helper.verify_buffer(flipped=False)
    helper.fill_buffer(flipped=True)


def test_ipc_mempool_multiple(device, ipc_memory_resource):
    """Test IPC with memory pools using multiple processes."""
    # Construct an IPC-enabled memory resource and share it over two channels.
    mr = ipc_memory_resource
    ch1, ch2 = (mr.create_ipc_channel() for _ in range(2))

    # Allocate memory buffers and export them to each channel.
    buffer1 = mr.allocate(NBYTES)
    ch1.send_buffer(buffer1)
    ch2.send_buffer(buffer1)
    buffer2 = mr.allocate(NBYTES)
    ch1.send_buffer(buffer2)
    ch2.send_buffer(buffer2)

    # Start the child processes.
    p1 = multiprocessing.Process(target=child_main2, args=(1, ch1))
    p2 = multiprocessing.Process(target=child_main2, args=(2, ch2))
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


def child_main2(idx, channel):
    device = Device()
    device.set_current()
    buffer1 = channel.receive_buffer()  # implicitly set up the shared memory pool
    buffer2 = channel.receive_buffer()
    if idx == 1:
        IPCBufferTestHelper(device, buffer1).fill_buffer(flipped=False)
    elif idx == 2:
        IPCBufferTestHelper(device, buffer2).fill_buffer(flipped=True)


def test_ipc_shared_allocation_handle(device, ipc_memory_resource):
    """Demonstrate that a memory pool allocation handle can be reused for IPC
    with multiple processes."""
    # Set up communication.
    ch1 = IPCChannel()
    ch2 = IPCChannel()
    q1 = multiprocessing.Queue()
    q2 = multiprocessing.Queue()

    # Start children.
    p1 = multiprocessing.Process(target=child_main3, args=(1, ch1, q1))
    p2 = multiprocessing.Process(target=child_main3, args=(2, ch2, q2))
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


def child_main3(idx, channel, queue):
    """Fills a shared memory buffer."""
    device = Device()
    device.set_current()
    alloc_handle = channel.receive_allocation_handle()
    mr = DeviceMemoryResource.from_allocation_handle(device, alloc_handle)
    buffer_descriptor = queue.get(timeout=CHILD_TIMEOUT_SEC)
    buffer = Buffer.import_(mr, buffer_descriptor)
    IPCBufferTestHelper(device, buffer).fill_buffer(starting_from=idx)


def test_ipc_shared_allocation_handle2(device, ipc_memory_resource):
    """Demonstrate that a memory pool allocation handle can be reused for IPC
    with multiple processes (simplified)."""
    # Set up communication.
    ch1 = IPCChannel()
    ch2 = IPCChannel()

    # Start children.
    p1 = multiprocessing.Process(target=child_main4, args=(1, ch1))
    p2 = multiprocessing.Process(target=child_main4, args=(2, ch2))
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


def child_main4(idx, channel):
    """Fills a shared memory buffer."""
    device = Device()
    device.set_current()
    buffer = channel.receive_buffer()
    IPCBufferTestHelper(device, buffer).fill_buffer(starting_from=idx)
