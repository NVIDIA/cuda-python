# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.core.experimental import Device
from utility import IPCBufferTestHelper
import multiprocessing
import pytest

CHILD_TIMEOUT_SEC = 10
NBYTES = 64


def test_ipc_mempool_multiple(device, ipc_memory_resource):
    """Test IPC with memory pools using multiple processes."""
    # Construct an IPC-enabled memory resource and share it over two channels.
    mr = ipc_memory_resource
    ch1, ch2 = (mr.create_ipc_channel() for _ in range(2))

    # Allocate memory buffers and export them to each channel.
    buffer1 = mr.allocate(NBYTES)
    ch1.export(buffer1)
    ch2.export(buffer1)
    buffer2 = mr.allocate(NBYTES)
    ch1.export(buffer2)
    ch2.export(buffer2)

    # Start the child processes.
    p1 = multiprocessing.Process(target=child_main, args=(1, ch1))
    p2 = multiprocessing.Process(target=child_main, args=(2, ch2))
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


def child_main(idx, channel):
    device = Device()
    device.set_current()
    buffer1 = channel.import_() # implicitly set up the shared memory pool
    buffer2 = channel.import_()
    if idx == 1:
        IPCBufferTestHelper(device, buffer1).fill_buffer(flipped=False)
    elif idx == 2:
        IPCBufferTestHelper(device, buffer2).fill_buffer(flipped=True)

