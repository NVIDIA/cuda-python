# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for tests that spawn ``multiprocessing.Process`` children.

These exist primarily to defend IPC tests against a class of CI hang where a
child process spawns too slowly and the parent does not implement proper guards
for that (see issue #2004). Without intervention, a zombie child holds an IPC
memory handle and blocks the parent's ``mr.close()`` in fixture teardown,
leading to deadlock and wedging the test runner for hours.
"""

import contextlib
import multiprocessing.process
import weakref

from cuda_python_test_helpers import under_compute_sanitizer

CHILD_TIMEOUT_SEC_DEFAULT = 30
CHILD_TIMEOUT_SEC_SANITIZER = 120


def child_timeout_sec() -> int:
    """Return the per-process join/wait timeout for IPC-style tests.

    Compute-sanitizer significantly slows process startup and CUDA context
    teardown, so we use a larger budget when it is active.
    """
    return CHILD_TIMEOUT_SEC_SANITIZER if under_compute_sanitizer() else CHILD_TIMEOUT_SEC_DEFAULT


@contextlib.contextmanager
def track_child_processes():
    """Context manager that kills any ``multiprocessing.Process`` children still
    alive at exit.

    Patches ``multiprocessing.process.BaseProcess.__init__`` to record every
    ``Process`` instance constructed inside the ``with`` block. This covers
    the delegating ``mp.Process`` class as well as direct ``SpawnProcess`` /
    ``ForkProcess`` instances (including those created by ``mp.Pool``), since
    all of them inherit from ``BaseProcess``. On exit, any tracked process
    that is still alive is killed and joined.

    This protects fixture teardown (e.g. ``ipc_memory_resource``'s
    ``mr.close()``) from blocking on IPC handles held by a stuck child --
    see issue #2004.
    """
    tracked = weakref.WeakSet()
    base = multiprocessing.process.BaseProcess
    original_init = base.__init__

    def tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        tracked.add(self)

    base.__init__ = tracking_init
    try:
        yield
    finally:
        base.__init__ = original_init
        for proc in list(tracked):
            # is_alive() / kill() raise ValueError if the Process was never
            # started or has already been closed; nothing to clean up in that
            # case.
            with contextlib.suppress(ValueError):
                if proc.is_alive():
                    proc.kill()
                    proc.join()
