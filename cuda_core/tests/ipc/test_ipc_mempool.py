# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.core.experimental import Device
from utility import IPCBufferTestHelper
import multiprocessing
import pytest

CHILD_TIMEOUT_SEC = 10
NBYTES = 64

def test_ipc_mempool(device, ipc_memory_resource):
    """Test IPC with memory pools."""
    # Set up the IPC-enabled memory pool and share it.
    mr = ipc_memory_resource
    channel = mr.create_ipc_channel()

    # Start the child process.
    process = multiprocessing.Process(target=child_main, args=(channel,))
    process.start()

    # Allocate and fill memory.
    buffer = mr.allocate(NBYTES)
    helper = IPCBufferTestHelper(device, buffer)
    helper.fill_buffer(flipped=False)

    # Export the buffer via IPC.
    channel.export(buffer)

    # Wait for the child process.
    process.join(timeout=CHILD_TIMEOUT_SEC)
    assert process.exitcode == 0

    # Verify that the buffer was modified.
    helper.verify_buffer(flipped=True)


def child_main(channel):
    device = Device()
    device.set_current()
    buffer = channel.import_()
    helper = IPCBufferTestHelper(device, buffer)
    helper.verify_buffer(flipped=False)
    helper.fill_buffer(flipped=True)
