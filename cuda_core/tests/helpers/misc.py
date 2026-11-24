# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading

import pytest


def run_in_threads(targets, *, barrier=None, timeout=None):
    """Run each callable in ``targets`` on its own thread, and re-raise the first failure.

    An exception in a thread would otherwise only end that thread. A thread that
    raises aborts ``barrier``, if given, so the threads waiting at it raise
    ``threading.BrokenBarrierError`` instead of blocking forever; the original
    exception is re-raised, not the broken barrier. If starting a thread fails,
    the barrier is aborted the same way. Every started thread is joined (up to
    ``timeout``) before this returns or raises.

    With ``timeout``, each join waits at most that many seconds, and a thread
    still running afterwards (for example a deadlocked one) raises
    ``TimeoutError``, also when another thread failed; the failure is its
    ``__cause__``. Such a thread cannot be stopped, so callers must not
    release anything it may still be using. The barrier is aborted first, so a
    thread waiting at it (or reaching it later) raises
    ``threading.BrokenBarrierError`` and exits instead of blocking until the
    interpreter shuts down. It does not run the code after its wait.
    """
    errors = []

    def run(target):
        try:
            target()
        # pytest.fail and pytest.skip raise BaseException subclasses.
        except BaseException as e:
            errors.append(e)
            if barrier is not None:
                barrier.abort()

    threads = [threading.Thread(target=run, args=(target,)) for target in targets]
    started = []
    try:
        for thread in threads:
            thread.start()
            started.append(thread)
    except BaseException:
        if barrier is not None:
            barrier.abort()
        raise
    finally:
        for thread in started:
            thread.join(timeout)
    failure = (
        next((e for e in errors if not isinstance(e, threading.BrokenBarrierError)), errors[0]) if errors else None
    )
    running = [thread.name for thread in started if thread.is_alive()]
    if running:
        if barrier is not None:
            barrier.abort()
        raise TimeoutError(f"threads still running after {timeout} s: {running}") from failure
    if failure is not None:
        raise failure


def try_create_condition(g, default_value=1):
    """Create a GraphCondition on graph *g*, skipping the test if unsupported."""
    from cuda.core._utils.cuda_utils import CUDAError

    try:
        return g.create_condition(default_value=default_value)
    except CUDAError:
        pytest.skip("Conditional nodes not supported (requires CC >= 9.0)")


class StreamWrapper:
    """
    A wrapper around Stream for testing IsStreamType conversions.
    """

    def __init__(self, stream):
        self._stream = stream

    def __cuda_stream__(self):
        return self._stream.__cuda_stream__()
