# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
from itertools import cycle

import pytest
from cuda.core.experimental import DeviceMemoryResource, DeviceMemoryResourceOptions
from utility import IPCBufferTestHelper

from cuda_python_test_helpers import supports_ipc_mempool

CHILD_TIMEOUT_SEC = 20
NBYTES = 64
NMRS = 3
NTASKS = 7
POOL_SIZE = 2097152


@pytest.mark.parametrize("nmrs", (1, NMRS))
def test_ipc_send_buffers(ipc_device, nmrs):
    """Test passing buffers sourced from multiple memory resources."""
    if not supports_ipc_mempool(ipc_device):
        pytest.skip("Driver rejects IPC-enabled mempool creation on this platform")
    # Set up several IPC-enabled memory pools.
    device = ipc_device
    options = DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True)
    mrs = [DeviceMemoryResource(device, options=options) for _ in range(NMRS)]

    # Allocate and fill memory.
    buffers = [mr.allocate(NBYTES) for mr, _ in zip(cycle(mrs), range(NTASKS))]
    for buffer in buffers:
        helper = IPCBufferTestHelper(device, buffer)
        helper.fill_buffer(flipped=False)

    # Start the child process.
    process = mp.Process(
        target=child_main,
        args=(
            device,
            buffers,
        ),
    )
    process.start()

    # Wait for the child process.
    process.join(timeout=CHILD_TIMEOUT_SEC)
    assert process.exitcode == 0

    # Verify that the buffers were modified.
    for buffer in buffers:
        helper = IPCBufferTestHelper(device, buffer)
        helper.verify_buffer(flipped=True)
        buffer.close()


def child_main(device, buffers):
    device.set_current()
    for buffer in buffers:
        helper = IPCBufferTestHelper(device, buffer)
        helper.verify_buffer(flipped=False)
        helper.fill_buffer(flipped=True)
        buffer.close()
