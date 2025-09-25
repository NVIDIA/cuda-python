# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
from itertools import cycle

from utility import IPCBufferTestHelper

from cuda.core.experimental import Buffer, Device, DeviceMemoryResource

CHILD_TIMEOUT_SEC = 4
NBYTES = 64
NWORKERS = 2
NMRS = 3
NTASKS = 20
POOL_SIZE = 2097152

# Global memory resources, set in children.
g_mrs = None


class TestIpcWorkerPoolUsingExport:
    """
    Test buffer sharing using export handles.

    The memory resources need to be passed to subprocesses at startup. Buffers
    are passed by their handles and reconstructed using the corresponding mr.
    """

    @staticmethod
    def init_worker(mrs):
        global g_mrs
        g_mrs = mrs

    def test_ipc_workerpool(self, device, ipc_memory_resource):
        """Test IPC with a worker pool."""
        mr = ipc_memory_resource
        buffers = [mr.allocate(NBYTES) for _ in range(NTASKS)]
        with multiprocessing.Pool(processes=NWORKERS, initializer=self.init_worker, initargs=([mr],)) as pool:
            pool.starmap(self.process_buffer, [(0, buffer.export()) for buffer in buffers])

        for buffer in buffers:
            IPCBufferTestHelper(device, buffer).verify_buffer(flipped=True)

    def test_ipc_workerpool_multi_mr(self, device, ipc_memory_resource):
        """Test IPC with a worker pool using multiple memory resources."""
        mrs = [ipc_memory_resource] + [
            DeviceMemoryResource(device, dict(max_size=POOL_SIZE, ipc_enabled=True)) for _ in range(NMRS - 1)
        ]
        buffers = [mr.allocate(NBYTES) for mr, _ in zip(cycle(mrs), range(NTASKS))]
        with multiprocessing.Pool(processes=NWORKERS, initializer=self.init_worker, initargs=(mrs,)) as pool:
            pool.starmap(
                self.process_buffer, [(mrs.index(buffer.memory_resource), buffer.export()) for buffer in buffers]
            )

        for buffer in buffers:
            IPCBufferTestHelper(device, buffer).verify_buffer(flipped=True)

    def process_buffer(self, mr_idx, buffer_desc):
        device = Device()
        buffer = Buffer.import_(g_mrs[mr_idx], buffer_desc)
        IPCBufferTestHelper(device, buffer).fill_buffer(flipped=True)


class TestIpcWorkerPool:
    """
    Test buffer sharing without using export handles.

    The memory resources need to be passed to subprocesses at startup. Buffers
    are serialized with the `uuid` of the corresponding mr, and the
    import/export is handled automatically.
    """

    @staticmethod
    def init_worker(mrs):
        global g_mrs
        g_mrs = mrs

    def test_ipc_workerpool(self, device, ipc_memory_resource):
        """Test IPC with a worker pool."""
        mr = ipc_memory_resource
        buffers = [mr.allocate(NBYTES) for _ in range(NTASKS)]
        with multiprocessing.Pool(processes=NWORKERS, initializer=self.init_worker, initargs=([mr],)) as pool:
            pool.map(self.process_buffer, buffers)

        for buffer in buffers:
            IPCBufferTestHelper(device, buffer).verify_buffer(flipped=True)

    def test_ipc_workerpool_multi_mr(self, device, ipc_memory_resource):
        """Test IPC with a worker pool using multiple memory resources."""
        mrs = [ipc_memory_resource] + [
            DeviceMemoryResource(device, dict(max_size=POOL_SIZE, ipc_enabled=True)) for _ in range(NMRS - 1)
        ]
        buffers = [mr.allocate(NBYTES) for mr, _ in zip(cycle(mrs), range(NTASKS))]
        with multiprocessing.Pool(processes=NWORKERS, initializer=self.init_worker, initargs=(mrs,)) as pool:
            pool.map(self.process_buffer, buffers)

        for buffer in buffers:
            IPCBufferTestHelper(device, buffer).verify_buffer(flipped=True)

    def process_buffer(self, buffer):
        device = Device()
        IPCBufferTestHelper(device, buffer).fill_buffer(flipped=True)
