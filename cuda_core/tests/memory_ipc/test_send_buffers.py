# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
from itertools import cycle

import pytest
from cuda.core.experimental import DeviceMemoryResource, DeviceMemoryResourceOptions
from helpers.buffers import PatternGen

CHILD_TIMEOUT_SEC = 20
NBYTES = 64
NMRS = 3
NTASKS = 7
POOL_SIZE = 2097152


@pytest.mark.parametrize("nmrs", (1, NMRS))
def test_ipc_send_buffers(ipc_device, nmrs):
    """Test passing buffers sourced from multiple memory resources."""
    # Set up several IPC-enabled memory pools.
    device = ipc_device
    options = DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True)
    mrs = [DeviceMemoryResource(device, options=options) for _ in range(nmrs)]

    # Allocate and fill memory.
    buffers = [mr.allocate(NBYTES) for mr, _ in zip(cycle(mrs), range(NTASKS))]
    pgen = PatternGen(device, NBYTES)
    for buffer in buffers:
        pgen.fill_buffer(buffer, seed=False)

    # Start the child process.
    process = mp.Process(target=child_main, args=(device, buffers))
    process.start()

    # Wait for the child process.
    process.join(timeout=CHILD_TIMEOUT_SEC)
    assert process.exitcode == 0

    # Verify that the buffers were modified.
    pgen = PatternGen(device, NBYTES)
    for buffer in buffers:
        pgen.verify_buffer(buffer, seed=True)
        buffer.close()


def child_main(device, buffers):
    device.set_current()
    pgen = PatternGen(device, NBYTES)
    for buffer in buffers:
        pgen.verify_buffer(buffer, seed=False)
        pgen.fill_buffer(buffer, seed=True)
        buffer.close()
