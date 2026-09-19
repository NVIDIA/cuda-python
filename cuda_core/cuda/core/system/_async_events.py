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

The state machine is pure Python so that it can be exercised without NVML; the
:mod:`cuda.core.system` extension types only supply the native call and convert
the result into their public types.
"""

from __future__ import annotations

import asyncio
import math
import time
from concurrent.futures import ThreadPoolExecutor

from cuda.bindings import nvml

__all__ = ["EventSetWaiting"]


# Longest native wait slice: bounds both the cancellation latency and how often
# an idle wait wakes up.
_SLICE_MS = 100

# Upper bound on the threads that may be blocked in a native wait at once.  The
# pool is private so that waiting never occupies the default executor of the
# application.
_MAX_WORKERS = 8

_executor = None


def _workers() -> ThreadPoolExecutor:
    """The private pool that holds threads blocked in a native wait."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="cuda-core-nvml")
    return _executor


def _slice_result(native):
    """Result of a finished slice, or ``None`` when it timed out or never ran."""
    if native.cancelled():
        return None
    error = native.exception()
    if error is None:
        return native.result()
    if isinstance(error, nvml.TimeoutError):
        return None
    raise error


async def _drain(native, loop):
    """Wait out an in-flight slice; return its event, or ``None`` if it timed out.

    ``native`` is the ``concurrent.futures.Future`` of the submitted slice, not
    the asyncio wrapper: cancelling the awaiting task cancels that wrapper, so
    the drain waits on the work item itself.  One wrapper is shielded
    repeatedly, which also keeps a repeat cancel from abandoning the borrowed
    event set.
    """
    wrapper = asyncio.wrap_future(native, loop=loop)
    while not native.done():
        try:
            await asyncio.shield(wrapper)
        except asyncio.CancelledError:
            continue
        except nvml.TimeoutError:
            return None
    return _slice_result(native)


class EventSetWaiting:
    """Single-consumer lease and pending-event hand-off for one event set.

    NVML hands an event to whichever thread waits on the event set, so only one
    native wait may be in flight.  The lease enforces that, and the single
    pending slot keeps the event that was consumed by a waiter that went away
    (cancelled or past its deadline) for the next consumer.
    """

    __slots__ = ("_busy", "_pending")

    def __init__(self) -> None:
        self._busy = False
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
                native = _workers().submit(native_wait, slice_ms)
                try:
                    return await asyncio.wrap_future(native, loop=loop)
                except nvml.TimeoutError:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise
                    # A native wait may return early (e.g. on an interrupt);
                    # wait out the rest of the slice so that does not become a
                    # poll loop.
                    shortfall = slice_ms / 1000 - (time.monotonic() - started)
                    if shortfall > 0:
                        await asyncio.sleep(shortfall)
                except asyncio.CancelledError:
                    result = await _drain(native, loop)
                    if result is not None:
                        self.park(result)
                    raise
        finally:
            self._busy = False
