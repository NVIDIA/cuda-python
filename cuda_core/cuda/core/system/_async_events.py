# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any

from cuda.bindings import nvml

__all__ = ["EventSetWaiting"]


# Longest native wait slice: bounds both the cancellation latency and how often
# an idle wait wakes up.
_SLICE_MS = 100

# The same bound in seconds, for the slice arithmetic.
_SLICE_S = _SLICE_MS / 1000

# Upper bound on the threads that may be blocked in a native wait at once.
_MAX_WORKERS = 8

# A slice that timed out.  Slices are an implementation detail, so the timeout
# of one slice is a value rather than an exception: only the caller's own
# deadline raises, once, with the project's NVML timeout type.
_TIMED_OUT = object()

# How long an idle worker waits for the next slice before retiring.  Workers
# stop by themselves, so nothing has to wake them when the interpreter exits.
_IDLE_POLL_S = 0.02

# Contended async waits must yield while acquiring the event set's thread lock.
_LEASE_POLL_S = 0.001


def _settle(future, payload, error):
    """Hand a finished slice to the event loop; runs on the loop thread."""
    if future.done():
        return
    if error is None:
        future.set_result(payload)
    else:
        future.set_exception(error)


def _notify(loop, future, payload, error):
    """Notify a live loop; a closed loop leaves the result in the outcome."""
    try:
        loop.call_soon_threadsafe(_settle, future, payload, error)
    except RuntimeError:
        if not loop.is_closed():
            raise


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

    __slots__ = ("_abandoned", "_lock", "drain_future", "error", "finished", "payload")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._abandoned = None
        self.drain_future = None
        self.error = None
        self.payload = None
        self.finished = False

    def reset(self) -> None:
        self._abandoned = None
        self.drain_future = None
        self.error = None
        self.payload = None
        self.finished = False

    def finish(self, payload, error):
        """Publish the slice result; a drain that is already waiting is returned."""
        with self._lock:
            self.payload = payload
            self.error = error
            self.finished = True
            waiting, self.drain_future = self.drain_future, None
            abandoned, self._abandoned = self._abandoned, None
        if abandoned is not None:
            abandoned(payload, error)
        return waiting

    def abandon(self, release):
        """Release a closed coroutine's lease only after its native wait ends."""
        with self._lock:
            if not self.finished:
                self._abandoned = release
                return
            payload, error = self.payload, self.error
        release(payload, error)

    def claim(self, loop):
        """Future that completes with this slice, or ``None`` if it already has."""
        with self._lock:
            if self.finished:
                return None
            if self.drain_future is None:
                self.drain_future = loop.create_future()
            return self.drain_future


class _Dispatcher:
    """Runs blocking native waits on a bounded, lazily grown set of threads.

    A worker takes one slice off a queue, calls the native wait, and hands the
    outcome back to the event loop.  Workers are non-daemon and retire once they
    have been idle for a moment, so an in-flight slice always completes before
    the event set it borrowed can be released, and a process that stops waiting
    still exits promptly.
    """

    def __init__(self, max_workers: int = _MAX_WORKERS) -> None:
        self._queue: queue.SimpleQueue[tuple[Any, ...]] = queue.SimpleQueue()
        self._max_workers = max_workers
        self._threads: list[threading.Thread] = []
        self._pending = 0
        self._lock = threading.Lock()

    def _run(self) -> None:
        request: tuple[Any, ...] | None
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
            waiting = outcome.finish(payload, error)
            with self._lock:
                self._pending -= 1
            _notify(loop, future, payload, error)
            if waiting is not None:
                # Draining only needs a completion notification. The outcome
                # retains errors so they cannot replace task cancellation.
                _notify(loop, waiting, None, None)

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
_dispatcher_lock = threading.Lock()


def _workers() -> _Dispatcher:
    """The private dispatcher, created on first use."""
    global _dispatcher
    with _dispatcher_lock:
        if _dispatcher is None:
            _dispatcher = _Dispatcher()
        return _dispatcher


async def _drain(outcome, loop):
    """Wait out the borrowed slice and return its payload and error.

    The future that carried the slice was cancelled together with the task, so
    this waits on the slice itself: the worker that owns it completes the
    waiter this registers.  Letting the drain finish before the cancellation
    returns is what makes releasing the event set safe, and it costs only the
    rest of the slice - never another worker.  A repeat cancel just re-shields
    the same waiter.
    """
    while True:
        waiting = outcome.claim(loop)
        if waiting is None:
            break
        try:
            await asyncio.shield(waiting)
        except asyncio.CancelledError:
            continue
        break
    return outcome.payload, outcome.error


class EventSetWaiting:
    """Serialized waits and pending-event hand-off for one event set.

    NVML hands an event to whichever thread waits on the event set, so only one
    native wait may be in flight.  A thread lock enforces that, and the single
    pending slot keeps the event that was consumed by a waiter that went away
    (cancelled or past its deadline) for the next consumer.
    """

    __slots__ = ("_lock", "_outcome", "_pending", "_pending_error")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._outcome = _Outcome()
        self._pending = None
        self._pending_error = None

    @property
    def is_waiting(self) -> bool:
        """Whether a wait currently borrows the event set."""
        return self._lock.locked()

    def park(self, payload) -> None:
        """Keep an already-consumed event for the next consumer."""
        self._pending = payload

    def take_pending(self):
        """Return the parked event and clear the slot."""
        pending, self._pending = self._pending, None
        return pending

    def _claim(self, convert):
        error, self._pending_error = self._pending_error, None
        if error is not None:
            raise error
        pending = self.take_pending()
        if pending is None:
            return None, False
        return self._convert(pending, convert), True

    def _convert(self, payload, convert):
        if convert is None:
            return payload
        try:
            return convert(payload)
        except BaseException:
            # Roll back consumption if conversion failed, even for a fresh
            # native result. The next consumer can retry without losing it.
            self.park(payload)
            raise

    def _handoff(self, payload, error):
        if error is not None:
            self._pending_error = error
        elif payload is not _TIMED_OUT:
            self.park(payload)

    def _release_abandoned(self, payload, error):
        self._handoff(payload, error)
        self._lock.release()

    def wait(self, native_wait, timeout_ms: int, convert=None):
        """Blocking wait; the lease is held for the whole native call."""
        with self._lock:
            pending, claimed = self._claim(convert)
            return pending if claimed else self._convert(native_wait(timeout_ms), convert)

    async def wait_async(self, native_wait, timeout_ms: int, convert=None):
        """Await one event, draining the native wait before cancellation returns.

        ``native_wait`` takes the slice length in milliseconds and is called
        from a worker thread.  It must be a bound method of the object that owns
        the native event set, so that awaiting this coroutine keeps that handle
        alive for as long as a slice can still borrow it.  ``convert`` turns a
        consumed payload - fresh or parked - into the public result type.
        """
        if timeout_ms < 0:
            raise ValueError(f"timeout_ms must be >= 0, got {timeout_ms}")
        loop = asyncio.get_running_loop()
        deadline = None if timeout_ms == 0 else time.monotonic() + timeout_ms / 1000
        while not self._lock.acquire(blocking=False):
            delay = _LEASE_POLL_S
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _timeout_exception()
                delay = min(delay, remaining)
            await asyncio.sleep(delay)
        in_flight = False
        try:
            if deadline is not None and time.monotonic() >= deadline:
                raise _timeout_exception()
            pending, claimed = self._claim(convert)
            if claimed:
                return pending
            outcome = self._outcome
            while True:
                if deadline is None:
                    slice_ms = _SLICE_MS
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise _timeout_exception()
                    slice_ms = _SLICE_MS if remaining >= _SLICE_S else math.ceil(remaining * 1000)
                    if slice_ms < 1:
                        slice_ms = 1
                started = time.monotonic()
                outcome.reset()
                future = _workers().submit(loop, outcome, native_wait, slice_ms)
                in_flight = True
                try:
                    payload = await future
                    if payload is not _TIMED_OUT:
                        return self._convert(payload, convert)
                except asyncio.CancelledError:
                    payload, error = await _drain(outcome, loop)
                    self._handoff(payload, error)
                    raise
                finally:
                    in_flight = not outcome.finished
                if deadline is not None and time.monotonic() >= deadline:
                    raise _timeout_exception()
                # A native wait that gives up well before its slice would
                # otherwise turn this into a poll loop; ordinary timer jitter is
                # not worth another trip through the event loop.
                consumed = time.monotonic() - started
                if consumed * 2 < slice_ms / 1000:
                    await asyncio.sleep(slice_ms / 1000 - consumed)
        finally:
            if in_flight:
                # Closing a coroutine cannot await the drain. Its worker keeps
                # the lock until the native call has returned and parked its
                # outcome, including when the owning event loop is closed.
                outcome.abandon(self._release_abandoned)
            else:
                self._lock.release()
