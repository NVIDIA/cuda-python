# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.core.experimental._utils.cuda_utils import CUDAError
from cuda.core.experimental import Device, DeviceMemoryResource, IPCChannel
import multiprocessing
import pytest

CHILD_TIMEOUT_SEC = 10
NBYTES = 64

def test_share_to_wrong_channel(device, ipc_memory_resource):
    mr1 = ipc_memory_resource
    mr2 = DeviceMemoryResource(device, dict(ipc_enabled=True))

    channel1 = mr1.create_ipc_channel()
    buffer1 = mr1.allocate(NBYTES)
    buffer2 = mr2.allocate(NBYTES)

    channel1.send_buffer(buffer1) # ok

    with pytest.raises(CUDAError):
        channel1.send_buffer(buffer2)


def test_ipc_child_errors(device, ipc_memory_resource):
    """Test expected errors with allocating from a shared IPC memory pool."""
    mr = ipc_memory_resource
    # Set up the IPC-enabled memory pool and share it.
    channel = IPCChannel()
    mr.share_to_channel(channel)

    # Start a child process to generate error info.
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=child_main, args=(channel, queue))
    process.start()

    # Check the errors.
    exc_type, exc_msg = queue.get(timeout=CHILD_TIMEOUT_SEC)
    assert exc_type is TypeError
    assert exc_msg == "Cannot allocate from shared memory pool imported via IPC"

    # Wait for the child process.
    process.join(timeout=CHILD_TIMEOUT_SEC)
    assert process.exitcode == 0


def child_main(channel, queue):
    """Child process that pushes IPC errors to a shared queue for testing."""
    device = Device()
    device.set_current()

    mr = DeviceMemoryResource.from_shared_channel(device, channel)

    # Allocating from an imported pool.
    try:
        mr.allocate(NBYTES)
    except Exception as e:
        exc_info = type(e), str(e)
        queue.put(exc_info)
