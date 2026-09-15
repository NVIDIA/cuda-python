# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for #2840: concurrent IPC imports must not deadlock.

``deviceptr_import_ipc()`` took ``ipc_import_mutex`` while still holding the
GIL and released the GIL only inside the critical section. Because destruction
runs in reverse declaration order, the importing thread reacquired the GIL
before dropping the mutex, while a second thread blocked on the mutex holding
the GIL.

Every import here succeeds and nothing is reported, so a hang indicates the
lock ordering rather than anything on an error path. A deadlocked child is
detected by the parent's join timeout; the per-directory conftest timeout is
the final backstop.
"""

import contextlib
import multiprocessing as mp
import threading

import pytest
from helpers.child_processes import child_timeout_sec, kill_subprocesses

from cuda.core import Buffer, Device

CHILD_TIMEOUT_SEC = child_timeout_sec()
NBYTES = 64
THREADS = 4

# these tests spawn new processes and files which fails for very many threads
pytestmark = pytest.mark.parallel_threads_limit(4)


def child_main(queue):
    device = Device()
    device.set_current()
    mr = queue.get()
    descriptor = queue.get()

    # One descriptor for every thread is enough: the mutex is taken before the
    # pointer cache is consulted, so a thread that would have been a cache hit
    # still blocks on the lock while holding the GIL.
    barrier = threading.Barrier(THREADS)

    def importer():
        # A current context, so the import succeeds and nothing is reported.
        Device().set_current()
        barrier.wait()
        Buffer.from_ipc_descriptor(mr, descriptor, stream=device.default_stream).close()

    threads = [threading.Thread(target=importer) for _ in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    device.sync()


class TestIpcConcurrentImport:
    """Importing one descriptor from several threads at once must not hang."""

    @pytest.fixture(autouse=True)
    def _set_start_method(self):
        # Ensure spawn is used for multiprocessing
        with contextlib.suppress(RuntimeError):
            mp.set_start_method("spawn", force=True)

    def test_main(self, ipc_device, ipc_memory_resource):
        ipc_device.set_current()
        mr = ipc_memory_resource

        stream = ipc_device.default_stream
        buffer = mr.allocate(NBYTES, stream=stream)
        stream.sync()

        queue = mp.Queue()
        process = mp.Process(target=child_main, args=(queue,))
        process.start()
        queue.put(mr)
        queue.put(buffer.ipc_descriptor)

        process.join(timeout=CHILD_TIMEOUT_SEC)
        survivors = kill_subprocesses(process)
        assert not survivors, "concurrent importers deadlocked (see #2840)"
        assert process.exitcode == 0, f"child process failed with exit code {process.exitcode}"
