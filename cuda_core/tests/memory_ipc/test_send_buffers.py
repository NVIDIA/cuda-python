# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
from itertools import cycle

from utility import IPCBufferTestHelper

from cuda.core.experimental import Device, DeviceMemoryResource

CHILD_TIMEOUT_SEC = 4
NBYTES = 64
NMRS = 3
NTASKS = 7
POOL_SIZE = 2097152


def test_ipc_send_buffers(device, ipc_memory_resource):
    """Test passing buffers directly to a child separately from a memory resource."""
    mr = ipc_memory_resource

    # Allocate and fill memory.
    buffers = [mr.allocate(NBYTES) for _ in range(NTASKS)]
    for buffer in buffers:
        helper = IPCBufferTestHelper(device, buffer)
        helper.fill_buffer(flipped=False)

    # Start the child process. Send the buffer directly.
    process = multiprocessing.Process(target=child_main, args=(buffers,))
    process.start()

    # Wait for the child process.
    process.join(timeout=CHILD_TIMEOUT_SEC)
    assert process.exitcode == 0

    # Verify that the buffers were modified.
    for buffer in buffers:
        helper = IPCBufferTestHelper(device, buffer)
        helper.verify_buffer(flipped=True)


def test_ipc_send_buffers_multi(device, ipc_memory_resource):
    """Test passing buffers sourced from multiple memory resources."""
    # Set up several IPC-enabled memory pools.
    mrs = [ipc_memory_resource] + [
        DeviceMemoryResource(device, dict(max_size=POOL_SIZE, ipc_enabled=True)) for _ in range(NMRS - 1)
    ]

    # Allocate and fill memory.
    buffers = [mr.allocate(NBYTES) for mr, _ in zip(cycle(mrs), range(NTASKS))]
    for buffer in buffers:
        helper = IPCBufferTestHelper(device, buffer)
        helper.fill_buffer(flipped=False)

    # Start the child process.
    process = multiprocessing.Process(target=child_main, args=(buffers,))
    process.start()

    # Wait for the child process.
    process.join(timeout=CHILD_TIMEOUT_SEC)
    assert process.exitcode == 0

    # Verify that the buffers were modified.
    for buffer in buffers:
        helper = IPCBufferTestHelper(device, buffer)
        helper.verify_buffer(flipped=True)


def child_main(buffers):
    device = Device()
    for buffer in buffers:
        helper = IPCBufferTestHelper(device, buffer)
        helper.verify_buffer(flipped=False)
        helper.fill_buffer(flipped=True)
