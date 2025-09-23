# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
from itertools import cycle

from utility import IPCBufferTestHelper

from cuda.core.experimental import Device, DeviceMemoryResource

CHILD_TIMEOUT_SEC = 10
NBYTES = 64
NWORKERS = 2
NMRS = 3
NTASKS = 20
POOL_SIZE = 2097152


def test_ipc_workerpool(device, ipc_memory_resource):
    """Test IPC with a worker pool."""
    mr = ipc_memory_resource
    buffers = [mr.allocate(NBYTES) for _ in range(NTASKS)]
    with multiprocessing.Pool(processes=NWORKERS) as pool:
        pool.map(process_buffer, buffers)

    for buffer in buffers:
        helper = IPCBufferTestHelper(device, buffer)
        helper.verify_buffer(flipped=True)


def test_ipc_workerpool_multi_mr(device, ipc_memory_resource):
    """Test IPC with a worker pool using multiple memory resources."""
    mrs = [ipc_memory_resource] + [
        DeviceMemoryResource(device, dict(max_size=POOL_SIZE, ipc_enabled=True)) for _ in range(NMRS - 1)
    ]
    buffers = [mr.allocate(NBYTES) for mr, _ in zip(cycle(mrs), range(NTASKS))]
    with multiprocessing.Pool(processes=NWORKERS) as pool:
        pool.map(process_buffer, buffers)

    for buffer in buffers:
        helper = IPCBufferTestHelper(device, buffer)
        helper.verify_buffer(flipped=True)


def process_buffer(buffer):
    device = Device()
    helper = IPCBufferTestHelper(device, buffer)
    helper.fill_buffer(flipped=True)
