# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
from itertools import cycle

import pytest
from cuda.core import Device, DeviceMemoryResource, DeviceMemoryResourceOptions
from helpers.buffers import PatternGen

CHILD_TIMEOUT_SEC = 30
NBYTES = 64
NMRS = 3
NTASKS = 7
POOL_SIZE = 2097152


class TestIpcSendBuffers:
    @pytest.mark.flaky(reruns=2)
    @pytest.mark.parametrize("nmrs", (1, NMRS))
    def test_main(self, ipc_device, nmrs):
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
        process = mp.Process(target=self.child_main, args=(device, buffers))
        process.start()

        # Wait for the child process.
        process.join(timeout=CHILD_TIMEOUT_SEC)
        assert process.exitcode == 0

        # Verify that the buffers were modified.
        pgen = PatternGen(device, NBYTES)
        for buffer in buffers:
            pgen.verify_buffer(buffer, seed=True)
            buffer.close()

    def child_main(self, device, buffers):
        device.set_current()
        pgen = PatternGen(device, NBYTES)
        for buffer in buffers:
            pgen.verify_buffer(buffer, seed=False)
            pgen.fill_buffer(buffer, seed=True)
            buffer.close()


class TestIpcReexport:
    """
    Test re-export of an IPC-enabled memory allocation.

    Work is done by three processes as follows:

        - Process A allocates a buffer and shares it with process B.
        - Process B shares it with process C.
        - Process C receives the buffer, fills it, and signals completion.

    This test checks that a buffer allocated in A can be exported to B and then
    re-exported from B to C.
    """

    @pytest.mark.flaky(reruns=2)
    def test_main(self, ipc_device, ipc_memory_resource):
        # Set up the device.
        device = ipc_device
        device.set_current()

        # Allocate, fill a buffer.
        mr = ipc_memory_resource
        pgen = PatternGen(device, NBYTES)
        buffer = mr.allocate(NBYTES)
        pgen.fill_buffer(buffer, seed=0)

        # Set up communication.
        q_bc = mp.Queue()
        event_b, event_c = [mp.Event() for _ in range(2)]

        # Spawn B and C.
        proc_b = mp.Process(target=self.process_b_main, args=(buffer, q_bc, event_b))
        proc_c = mp.Process(target=self.process_c_main, args=(q_bc, event_c))
        proc_b.start()
        proc_c.start()

        # Wait for C to signal completion then clean up.
        event_c.wait(timeout=CHILD_TIMEOUT_SEC)
        event_b.set()  # b can finish now
        proc_b.join(timeout=CHILD_TIMEOUT_SEC)
        proc_c.join(timeout=CHILD_TIMEOUT_SEC)
        assert proc_b.exitcode == 0
        assert proc_c.exitcode == 0

        # Verify that C’s operations are visible.
        pgen.verify_buffer(buffer, seed=1)
        buffer.close()

    def process_b_main(self, buffer, q_bc, event_b):
        # Process B: receive buffer from A then forward it to C.
        device = Device()
        device.set_current()

        # Forward the buffer to C.
        q_bc.put(buffer)
        buffer.close()

        # Wait for C to receive before exiting.
        event_b.wait(timeout=CHILD_TIMEOUT_SEC)

    def process_c_main(self, q_bc, event_c):
        # Process C: receive buffer from B then fill it.
        device = Device()
        device.set_current()

        # Get the buffer and fill it.
        buffer = q_bc.get(timeout=CHILD_TIMEOUT_SEC)
        pgen = PatternGen(device, NBYTES)
        pgen.fill_buffer(buffer, seed=1)
        buffer.close()

        # Signal A that the work is complete.
        event_c.set()
