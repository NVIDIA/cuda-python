# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stress test: importing a descriptor while its last import is being released.

``deviceptr_import_ipc()`` shares one handle per descriptor, because the
driver's first free of an imported pointer unmaps every import of it
(nvbug 5570902). Releasing the last handle frees the mapping after its
reference count reaches zero, so an import can arrive while that free is still
pending. The import must wait for the free, not re-import the still-mapped
pointer and lose it to the pending free.

Threads in one child import, use and close the same descriptor in a loop, so
the last handle is released again and again while other threads import. A
lost mapping shows up as a failed copy; a missed wakeup shows up as a hang,
which the parent's join timeout catches.
"""

import contextlib
import multiprocessing as mp
import threading
import time

import pytest
from helpers.child_processes import child_timeout_sec, kill_subprocesses

from cuda.core import Buffer, Device

CHILD_TIMEOUT_SEC = child_timeout_sec()
NBYTES = 4096
THREADS = 4
DURATION_SEC = 1.0

# these tests spawn new processes and files which fails for very many threads;
# child processes and IPC fixtures are shared across workers (#2784); cap at two threads
pytestmark = pytest.mark.parallel_threads_limit(2)


def child_main(queue):
    device = Device()
    device.set_current()
    mr = queue.get()
    descriptor = queue.get()
    # A VirtualMemoryResource rounds the allocation up to its granularity.
    nbytes = queue.get()

    barrier = threading.Barrier(THREADS)
    errors = []
    iterations = [0] * THREADS

    def worker(index):
        Device().set_current()
        stream = Device().create_stream()
        local = Device().memory_resource.allocate(nbytes, stream=stream)
        barrier.wait()
        deadline = time.monotonic() + DURATION_SEC
        try:
            while time.monotonic() < deadline and not errors:
                buffer = Buffer.from_ipc_descriptor(mr, descriptor, stream=stream)
                local.copy_from(buffer, stream=stream)
                stream.sync()
                buffer.close()
                iterations[index] += 1
        except Exception as e:
            errors.append(e)
        finally:
            local.close(stream)
            stream.sync()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if errors:
        raise errors[0]
    assert all(iterations), iterations
    device.sync()


class TestIpcImportReleaseRace:
    """Imports that race the release of the last handle must stay usable and must not hang."""

    @pytest.fixture(autouse=True)
    def _set_start_method(self):
        # Ensure spawn is used for multiprocessing
        with contextlib.suppress(RuntimeError):
            mp.set_start_method("spawn", force=True)

    @pytest.mark.agent_authored(model="claude-opus-5-5")
    def test_main(self, ipc_device, ipc_memory_resource):
        ipc_device.set_current()
        mr = ipc_memory_resource

        stream = ipc_device.default_stream
        buffer = mr.allocate(NBYTES, stream=stream)
        stream.sync()

        try:
            queue = mp.Queue()
            process = mp.Process(target=child_main, args=(queue,))
            process.start()
            queue.put(mr)
            queue.put(buffer.ipc_descriptor)
            queue.put(buffer.size)

            process.join(timeout=CHILD_TIMEOUT_SEC)
            survivors = kill_subprocesses(process)
            assert not survivors, "an import racing the last release did not return"
            assert process.exitcode == 0, f"child process failed with exit code {process.exitcode}"
        finally:
            buffer.close(stream)
