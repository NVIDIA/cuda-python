# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.core.experimental import Buffer, Device, DeviceMemoryResource, IPCChannel
from utility import IPCBufferTestHelper
import multiprocessing
import pytest

CHILD_TIMEOUT_SEC = 10
NBYTES = 64

def test_ipc_mempool(device, ipc_memory_resource):
    """Test IPC with memory pools."""
    # Set up the IPC-enabled memory pool and share it.
    mr = ipc_memory_resource
    channel = IPCChannel()
    mr.share_to_channel(channel)

    # Start the child process.
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=child_main, args=(channel, queue))
    process.start()

    # Allocate and fill memory.
    buffer = mr.allocate(NBYTES)
    helper = IPCBufferTestHelper(device, buffer, NBYTES)
    helper.fill_buffer(flipped=False)

    # Export the buffer via IPC.
    handle = buffer.export()
    queue.put(handle)

    # Wait for the child process.
    process.join(timeout=CHILD_TIMEOUT_SEC)
    assert process.exitcode == 0

    # Verify that the buffer was modified.
    helper.verify_buffer(flipped=True)


def child_main(channel, queue):
    device = Device()
    device.set_current()

    mr = DeviceMemoryResource.from_shared_channel(device, channel)
    handle = queue.get()  # Get exported buffer data
    buffer = Buffer.import_(mr, handle)

    helper = IPCBufferTestHelper(device, buffer, NBYTES)
    helper.verify_buffer(flipped=False)
    helper.fill_buffer(flipped=True)
