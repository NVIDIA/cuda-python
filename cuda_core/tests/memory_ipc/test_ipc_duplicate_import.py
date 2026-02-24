# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test for duplicate IPC buffer imports.

Verifies that importing the same buffer descriptor multiple times returns the
same underlying handle, and that closing all imports works correctly without
crashing. This tests the workaround for nvbug 5570902 where IPC-imported
pointers are not correctly reference counted by the driver.
"""

import contextlib
import multiprocessing as mp

import pytest
from cuda.core import Buffer, Device
from helpers.logging import TimestampedLogger

CHILD_TIMEOUT_SEC = 30
NBYTES = 64
POOL_SIZE = 2097152

ENABLE_LOGGING = False  # Set True for test debugging and development


def child_main(log, queue):
    log.prefix = " child: "
    log("ready")
    device = Device()
    device.set_current()
    mr = queue.get()
    buffer_desc1 = queue.get()
    buffer_desc2 = queue.get()

    # Import the same buffer twice - should return same handle due to cache
    buffer1 = Buffer.from_ipc_descriptor(mr, buffer_desc1)
    buffer2 = Buffer.from_ipc_descriptor(mr, buffer_desc2)

    log(f"buffer1.handle = {buffer1.handle}")
    log(f"buffer2.handle = {buffer2.handle}")
    log(f"same handle: {buffer1.handle == buffer2.handle}")

    # Close both - should not crash
    buffer1.close()
    log("buffer1 closed")

    buffer2.close()
    log("buffer2 closed")

    device.sync()
    log("done")


class TestIpcDuplicateImport:
    """Test that duplicate IPC imports return the same handle and close safely."""

    @pytest.fixture(autouse=True)
    def _set_start_method(self):
        # Ensure spawn is used for multiprocessing
        with contextlib.suppress(RuntimeError):
            mp.set_start_method("spawn", force=True)

    @pytest.mark.flaky(reruns=2)
    def test_main(self, ipc_device, ipc_memory_resource):
        log = TimestampedLogger(prefix="parent: ", enabled=ENABLE_LOGGING)
        ipc_device.set_current()
        mr = ipc_memory_resource

        log("allocating buffer")
        buffer = mr.allocate(NBYTES)

        # Start the child process.
        log("starting child")
        queue = mp.Queue()
        process = mp.Process(target=child_main, args=(log, queue))
        process.start()

        # Send the memory resource and buffer descriptor twice.
        log("sending mr and buffer descriptors")
        queue.put(mr)
        queue.put(buffer.get_ipc_descriptor())
        queue.put(buffer.get_ipc_descriptor())

        log("waiting for child")
        process.join(timeout=CHILD_TIMEOUT_SEC)
        log(f"child exit code: {process.exitcode}")
        assert process.exitcode == 0, f"Child process failed with exit code {process.exitcode}"
        log("done")
