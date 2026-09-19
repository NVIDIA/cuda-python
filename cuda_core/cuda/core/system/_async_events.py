# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Cancellable asyncio waiting for NVML event sets.

The native event wait blocks a thread until an event arrives or its timeout
expires, and it cannot be interrupted from another thread.  Awaiting it with
``asyncio.to_thread`` therefore couples task cancellation to the native call: a
cancelled task keeps its executor worker and the event-set handle borrowed
until that call returns, which for an infinite wait is never.

This module keeps the native wait bounded instead.  An awaiter issues the
native wait in slices of at most :data:`_SLICE_MS` against a monotonic deadline,
so that

* a cancelled task is drained within one slice,
* the caller's timeout budget still spans arbitrarily many slices,
* at most one native wait is in flight per event set, and
* an event consumed by a slice whose awaiter went away is handed to the next
  consumer instead of being dropped.

The slices run on a small set of threads that take requests off a queue: paying
a thread pool per request would dominate the cost of a short wait, and these
workers are reused across waits.  The state machine and the dispatcher are pure
Python so that they can be exercised without NVML; the
:mod:`cuda.core.system` extension types only supply the native call and convert
the result into their public types.
"""

from __future__ import annotations

import asyncio
import math
import queue
import threading
import time

from cuda.bindings import nvml

__all__ = ["EventSetWaiting"]


# Longest native wait slice: bounds both the cancellation latency and how often
# an idle wait wakes up.
_SLICE_MS = 100

# Upper bound on the threads that may be blocked in a native wait at once.
_MAX_WORKERS = 8

# A slice that timed out.  Slices are an implementation detail, so the timeout
# of one slice is a value rather than an exception: only the caller's own
# deadline raises, once, with the project's NVML timeout type.
_TIMED_OUT = object()

# How long an idle worker waits for the next slice before retiring.  Workers
# stop by themselves, so nothing has to wake them when the interpreter exits.
_IDLE_POLL_S = 0.05

# How long the drain's wait for a borrowed slice runs before it is renewed, and
# how often it looks at the outcome.
_DRAIN_WAIT_MS = 1000
_DRAIN_POLL_S = 0.0005


def _settle(future, payload, error):
    """Hand a finished slice to the event loop; runs on the loop thread."""
    if future.done():
        return
    if error is None:
        future.set_result(payload)
    else:
        future.set_exception(error)


def _timeout_exception():
    """The project's NVML timeout, raised for the caller's own deadline."""
    return nvml.TimeoutError(nvml.Return.ERROR_TIMEOUT)


class _Outcome:
    """How the borrowed slice ended.

    The fast path only needs the future it awaits; this slot exists for the
    drain, which has to read the slice's result after the task cancelled the
    future that carried it.  One slot per event set is enough: the lease allows
    a single borrowed slice at a time.
    """

    __slots__ = ("error", "finished", "payload")

    def __init__(self) -> None:
        self.error = None
        self.payload = None
        self.finished = False

    def reset(self) -> None:
        self.error = None
        self.payload = None
        self.finished = False

    def wait(self, timeout_ms: int) -> bool:
        """Block a worker thread until the slice ends; only draining uses this."""
        deadline = time.monotonic() + timeout_ms / 1000
        while not self.finished:
            if time.monotonic() >= deadline:
                return False
            time.sleep(_DRAIN_POLL_S)
        return True


class _Dispatcher:
    """Runs blocking native waits on a bounded, lazily grown set of threads.

    A worker takes one slice off a queue, calls the native wait, and hands the
    outcome back to the event loop.  Workers are non-daemon and retire once they
    have been idle for a moment, so an in-flight slice always completes before
    the event set it borrowed can be released, and a process that stops waiting
    still exits promptly.
    """

    def __init__(self, max_workers: int = _MAX_WORKERS) -> None:
        self._queue = queue.SimpleQueue()
        self._max_workers = max_workers
        self._threads = []
        self._pending = 0
        self._lock = threading.Lock()

    def _run(self) -> None:
        while True:
            try:
                request = self._queue.get(timeout=_IDLE_POLL_S)
            except queue.Empty:
                with self._lock:
                    if not self._queue.qsize():
                        self._threads.remove(threading.current_thread())
                        return
                continue
            loop, future, outcome, native_wait, slice_ms = request
            error = None
            try:
                payload = native_wait(slice_ms)
            except nvml.TimeoutError:
                payload = _TIMED_OUT
            except BaseException as exc:  # forwarded to the awaiter, never swallowed
                payload, error = None, exc
            finally:
                request = native_wait = None  # do not pin the event set any longer
            outcome.error = error
            outcome.payload = payload
            outcome.finished = True
            with self._lock:
                self._pending -= 1
            loop.call_soon_threadsafe(_settle, future, payload, error)

    def submit(self, loop, outcome, native_wait, slice_ms):
        """Queue one slice and return the future it will be delivered through."""
        future = loop.create_future()
        with self._lock:
            self._threads = [thread for thread in self._threads if thread.is_alive()]
            self._pending += 1
            self._queue.put((loop, future, outcome, native_wait, slice_ms))
            # One worker per in-flight slice, up to the bound: a queued slice
            # must not wait behind a long one belonging to another event set.
            while len(self._threads) < self._max_workers and self._pending > len(self._threads):
                thread = threading.Thread(target=self._run, name=f"cuda-core-nvml-{len(self._threads)}")
                thread.start()
                self._threads.append(thread)
        return future


_dispatcher = None


def _workers() -> _Dispatcher:
    """The private dispatcher, created on first use."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = _Dispatcher()
    return _dispatcher


async def _drain(outcome, loop):
    """Wait out the borrowed slice; return its event, or ``None`` if it timed out.

    The future that carried the slice was cancelled with the task, so this waits
    on the slice itself - on a worker thread, because the event loop is not the
    place to sit out a native call.  Letting the drain finish before the
    cancellation returns is what makes releasing the event set safe; a repeat
    cancel only means starting another wait for the same slice.
    """
    while not outcome.finished:
        waiting = _workers().submit(loop, _Outcome(), outcome.wait, _DRAIN_WAIT_MS)
        try:
            await asyncio.shield(waiting)
        except asyncio.CancelledError:
            continue
    if outcome.error is not None:
        raise outcome.error
    return None if outcome.payload is _TIMED_OUT else outcome.payload


class EventSetWaiting:
    """Single-consumer lease and pending-event hand-off for one event set.

    NVML hands an event to whichever thread waits on the event set, so only one
    native wait may be in flight.  The lease enforces that, and the single
    pending slot keeps the event that was consumed by a waiter that went away
    (cancelled or past its deadline) for the next consumer.
    """

    __slots__ = ("_busy", "_outcome", "_pending")

    def __init__(self) -> None:
        self._busy = False
        self._outcome = _Outcome()
        self._pending = None

    @property
    def is_waiting(self) -> bool:
        """Whether a wait currently borrows the event set."""
        return self._busy

    def park(self, payload) -> None:
        """Keep an already-consumed event for the next consumer."""
        self._pending = payload

    def take_pending(self):
        """Return the parked event and clear the slot."""
        pending, self._pending = self._pending, None
        return pending

    def _acquire(self) -> None:
        if self._busy:
            raise RuntimeError("an event wait is already in flight for this event set")
        self._busy = True

    def _claim(self, convert):
        pending = self.take_pending()
        if pending is None:
            return None
        return pending if convert is None else convert(pending)

    def wait(self, native_wait, timeout_ms: int, convert=None):
        """Blocking wait; the lease is held for the whole native call."""
        self._acquire()
        try:
            pending = self._claim(convert)
            return native_wait(timeout_ms) if pending is None else pending
        finally:
            self._busy = False

    async def wait_async(self, native_wait, timeout_ms: int, convert=None):
        """Await one event, draining the native wait before cancellation returns.

        ``native_wait`` takes the slice length in milliseconds and is called
        from a worker thread.  It must be a bound method of the object that owns
        the native event set, so that awaiting this coroutine keeps that handle
        alive for as long as a slice can still borrow it.
        """
        if timeout_ms < 0:
            raise ValueError(f"timeout_ms must be >= 0, got {timeout_ms}")
        self._acquire()
        try:
            pending = self._claim(convert)
            if pending is not None:
                return pending
            loop = asyncio.get_running_loop()
            deadline = None if timeout_ms == 0 else time.monotonic() + timeout_ms / 1000
            while True:
                if deadline is None:
                    slice_ms = _SLICE_MS
                else:
                    remaining_ms = math.ceil((deadline - time.monotonic()) * 1000)
                    slice_ms = min(_SLICE_MS, remaining_ms) if remaining_ms > 0 else 1
                started = time.monotonic()
                self._outcome.reset()
                future = _workers().submit(loop, self._outcome, native_wait, slice_ms)
                try:
                    payload = await future
                    if payload is not _TIMED_OUT:
                        return payload
                except asyncio.CancelledError:
                    result = await _drain(self._outcome, loop)
                    if result is not None:
                        self.park(result)
                    raise
                if deadline is not None and time.monotonic() >= deadline:
                    raise _timeout_exception()
                # A native wait that gives up well before its slice would
                # otherwise turn this into a poll loop; ordinary timer jitter is
                # not worth another trip through the event loop.
                consumed = time.monotonic() - started
                if consumed * 2 < slice_ms / 1000:
                    await asyncio.sleep(slice_ms / 1000 - consumed)
        finally:
            self._busy = False
