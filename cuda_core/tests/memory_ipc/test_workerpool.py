# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import pickle
from itertools import cycle

import pytest
from cuda.core import Buffer, Device, DeviceMemoryResource, DeviceMemoryResourceOptions
from helpers.buffers import PatternGen

CHILD_TIMEOUT_SEC = 30
NBYTES = 64
NWORKERS = 2
NMRS = 3
NTASKS = 20
POOL_SIZE = 2097152


class TestIpcWorkerPool:
    """
    Map a function over shared buffers using a worker pool to distribute work.

    This demonstrates the simplest interface, though not the most efficient
    one.  Each buffer transfer involes a deep transfer of the associated memory
    resource (duplicates are ignored on the receiving end).
    """

    @pytest.mark.flaky(reruns=2)
    @pytest.mark.parametrize("nmrs", (1, NMRS))
    def test_main(self, ipc_device, nmrs):
        device = ipc_device
        options = DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True)
        mrs = [DeviceMemoryResource(device, options=options) for _ in range(nmrs)]
        buffers = [mr.allocate(NBYTES) for mr, _ in zip(cycle(mrs), range(NTASKS))]

        with mp.Pool(NWORKERS) as pool:
            pool.map(self.process_buffer, buffers)

        pgen = PatternGen(device, NBYTES)
        for buffer in buffers:
            pgen.verify_buffer(buffer, seed=True)
            buffer.close()

    def process_buffer(self, buffer):
        device = Device(buffer.memory_resource.device_id)
        device.set_current()
        pgen = PatternGen(device, NBYTES)
        pgen.fill_buffer(buffer, seed=True)
        buffer.close()


class TestIpcWorkerPoolUsingIPCDescriptors:
    """
    Test buffer sharing using IPC descriptors.

    The memory resources are passed to subprocesses at startup. Buffers are
    passed by their handles and reconstructed using the corresponding resource.
    """

    @staticmethod
    def init_worker(mrs):
        """Called during child process initialization to store received memory resources."""
        TestIpcWorkerPoolUsingIPCDescriptors.mrs = mrs

    @pytest.mark.flaky(reruns=2)
    @pytest.mark.parametrize("nmrs", (1, NMRS))
    def test_main(self, ipc_device, nmrs):
        device = ipc_device
        options = DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True)
        mrs = [DeviceMemoryResource(device, options=options) for _ in range(nmrs)]
        buffers = [mr.allocate(NBYTES) for mr, _ in zip(cycle(mrs), range(NTASKS))]

        with mp.Pool(NWORKERS, initializer=self.init_worker, initargs=(mrs,)) as pool:
            pool.starmap(
                self.process_buffer,
                [(mrs.index(buffer.memory_resource), buffer.get_ipc_descriptor()) for buffer in buffers],
            )

        pgen = PatternGen(device, NBYTES)
        for buffer in buffers:
            pgen.verify_buffer(buffer, seed=True)
            buffer.close()

    def process_buffer(self, mr_idx, buffer_desc):
        mr = self.mrs[mr_idx]
        device = Device(mr.device_id)
        device.set_current()
        buffer = Buffer.from_ipc_descriptor(mr, buffer_desc)
        pgen = PatternGen(device, NBYTES)
        pgen.fill_buffer(buffer, seed=True)
        buffer.close()


class TestIpcWorkerPoolUsingRegistry:
    """
    Test buffer sharing using the memory resource registry.

    The memory resources are passed to subprocesses at startup, which
    implicitly registers them. Buffers are passed via serialization and matched
    to the corresponding memory resource through the registry. This is more
    complicated than the simple example (first, above) but passes buffers more
    efficiently.
    """

    @staticmethod
    def init_worker(mrs):
        # Passing mrs implicitly registers them.
        pass

    @pytest.mark.flaky(reruns=2)
    @pytest.mark.parametrize("nmrs", (1, NMRS))
    def test_main(self, ipc_device, nmrs):
        device = ipc_device
        options = DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True)
        mrs = [DeviceMemoryResource(device, options=options) for _ in range(nmrs)]
        buffers = [mr.allocate(NBYTES) for mr, _ in zip(cycle(mrs), range(NTASKS))]

        with mp.Pool(NWORKERS, initializer=self.init_worker, initargs=(mrs,)) as pool:
            pool.starmap(self.process_buffer, [(device, pickle.dumps(buffer)) for buffer in buffers])

        pgen = PatternGen(device, NBYTES)
        for buffer in buffers:
            pgen.verify_buffer(buffer, seed=True)
            buffer.close()

    def process_buffer(self, device, buffer_s):
        device.set_current()
        buffer = pickle.loads(buffer_s)  # noqa: S301
        pgen = PatternGen(device, NBYTES)
        pgen.fill_buffer(buffer, seed=True)
        buffer.close()
