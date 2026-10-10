# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for bounded asyncio waiting on NVML event sets."""

import asyncio
import gc
import subprocess
import sys
import threading
import time
import weakref

import pytest
from cuda_python_test_helpers.arch_check import skip_if_nvml_unsupported

from cuda.bindings import nvml
from cuda.core import system
from cuda.core.system import _async_events
from cuda.core.system._async_events import _MAX_WORKERS, _SLICE_MS, EventSetWaiting, _Dispatcher
from cuda.core.system._system_events import SystemEvents
from cuda.core.system.typing import EventType, SystemEventType

pytestmark = [
    skip_if_nvml_unsupported,
    pytest.mark.thread_unsafe(reason="patches the shared dispatcher and NVML entry points"),
]

TIMEOUT = nvml.TimeoutError(nvml.Return.ERROR_TIMEOUT)
EVENT = object()
BATCH = [f"event-{index}" for index in range(5)]


class FakeWait:
    """Deterministic stand-in for one slice of the native event-set wait.

    The real call blocks the calling thread for the requested timeout, so the
    fake does too unless it is told to deliver a result or to fail.
    """

    def __init__(self, deliver_at=None, error=None, hold=False, latency=None):
        self.calls = []
        self.in_flight = 0
        self.peak_in_flight = 0
        self.deliver_at = deliver_at
        self.error = error
        self.latency = latency
        self.release = threading.Event()
        if not hold:
            self.release.set()

    def __call__(self, timeout_ms):
        call = len(self.calls)
        self.calls.append(timeout_ms)
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            assert self.release.wait(10), "the fake native wait was never released"
            if self.error is not None:
                raise self.error
            if self.deliver_at is not None and call >= self.deliver_at:
                if self.latency is not None:
                    time.sleep(self.latency)
                return EVENT
            time.sleep(timeout_ms / 1000 if self.latency is None else self.latency)
            raise TIMEOUT
        finally:
            self.in_flight -= 1


class Owner:
    """Python stand-in for an extension type owning a native event set."""

    def __init__(self, fake):
        self.fake = fake

    def native_wait(self, timeout_ms):
        return self.fake(timeout_ms)


async def spin_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition was never met"
        await asyncio.sleep(0.001)


def test_timeout_budget_is_spent_in_bounded_slices():
    state = EventSetWaiting()
    fake = FakeWait()
    started = time.monotonic()
    with pytest.raises(system.TimeoutError):
        asyncio.run(state.wait_async(fake, 250))
    assert len(fake.calls) == 3, fake.calls
    assert all(1 <= call <= _SLICE_MS for call in fake.calls)
    assert sum(fake.calls) >= 200
    assert time.monotonic() - started >= 0.25
    assert state.is_waiting is False


def test_early_native_timeout_does_not_spin():
    """A native wait returning early still costs a full slice and full budget."""
    state = EventSetWaiting()
    fake = FakeWait(latency=0.0)
    started = time.monotonic()
    with pytest.raises(system.TimeoutError):
        asyncio.run(state.wait_async(fake, 200))
    assert len(fake.calls) == 2
    assert time.monotonic() - started >= 0.2


def test_event_is_returned_without_consuming_the_budget():
    state = EventSetWaiting()
    fake = FakeWait(deliver_at=0)
    assert asyncio.run(state.wait_async(fake, 5000)) is EVENT
    assert fake.calls == [_SLICE_MS]


def test_deadline_slice_still_delivers_a_late_event():
    """A result from the slice that expires the budget is delivered, not dropped."""
    state = EventSetWaiting()
    fake = FakeWait(deliver_at=0, latency=0.05)
    assert asyncio.run(state.wait_async(fake, 20)) is EVENT


def test_heartbeat_keeps_running_while_waiting():
    state = EventSetWaiting()
    fake = FakeWait()
    beats = []

    async def heartbeat():
        while True:
            beats.append(time.monotonic())
            await asyncio.sleep(0.005)

    async def main():
        beat = asyncio.create_task(heartbeat())
        try:
            with pytest.raises(system.TimeoutError):
                await state.wait_async(fake, 300)
        finally:
            beat.cancel()

    asyncio.run(main())
    assert len(beats) > 20


def test_cancel_before_any_native_wait():
    state = EventSetWaiting()
    fake = FakeWait(hold=True)

    async def main():
        task = asyncio.create_task(state.wait_async(fake, 0))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert fake.calls == []
    assert state.is_waiting is False


def test_cancel_drains_the_in_flight_slice():
    state = EventSetWaiting()
    fake = FakeWait(hold=True)
    in_flight_at_cancel = None

    async def main():
        nonlocal in_flight_at_cancel
        task = asyncio.create_task(state.wait_async(fake, 0))
        await spin_until(lambda: len(fake.calls) == 1)
        task.cancel()
        await asyncio.sleep(0.05)
        fake.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        in_flight_at_cancel = fake.in_flight

    asyncio.run(main())
    assert in_flight_at_cancel == 0, "cancellation returned while the native wait was still borrowed"
    assert state.is_waiting is False


def test_repeated_cancel_still_propagates_cancellation():
    state = EventSetWaiting()
    fake = FakeWait(hold=True)

    async def main():
        task = asyncio.create_task(state.wait_async(fake, 0))
        await spin_until(lambda: len(fake.calls) == 1)
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.01)
        fake.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert fake.in_flight == 0
    assert state.is_waiting is False


def test_event_consumed_by_a_cancelled_wait_is_handed_off():
    state = EventSetWaiting()
    fake = FakeWait(deliver_at=0, hold=True)

    async def main():
        task = asyncio.create_task(state.wait_async(fake, 0))
        await spin_until(lambda: len(fake.calls) == 1)
        task.cancel()
        fake.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert len(fake.calls) == 1
    idle = FakeWait(deliver_at=0)
    assert state.wait(idle, 0) is EVENT
    assert idle.calls == [], "the parked event was re-fetched from the driver"


@pytest.mark.parametrize("second_async", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_second_waiter_is_serialized_without_blocking_the_loop(second_async):
    state = EventSetWaiting()
    fake = FakeWait(hold=True, deliver_at=0)

    async def main():
        task = asyncio.create_task(state.wait_async(fake, 0))
        await spin_until(lambda: len(fake.calls) == 1)
        waiting = asyncio.create_task(
            state.wait_async(fake, 0) if second_async else asyncio.to_thread(state.wait, fake, 0)
        )
        try:
            await asyncio.sleep(0.02)
            assert not waiting.done()
            assert fake.peak_in_flight == 1
            fake.release.set()
            assert await task is EVENT
            assert await waiting is EVENT
        finally:
            fake.release.set()
            await asyncio.gather(task, waiting, return_exceptions=True)

    asyncio.run(main())
    assert fake.peak_in_flight == 1


def test_independent_event_sets_wait_concurrently():
    first, second = EventSetWaiting(), EventSetWaiting()
    left, right = FakeWait(hold=True), FakeWait(hold=True)

    async def main():
        tasks = [
            asyncio.create_task(first.wait_async(left, 0)),
            asyncio.create_task(second.wait_async(right, 0)),
        ]
        await spin_until(lambda: len(left.calls) == 1 and len(right.calls) == 1)
        left.release.set()
        right.release.set()
        await asyncio.gather(*tasks)

    left.deliver_at = 0
    right.deliver_at = 0
    asyncio.run(main())
    assert left.peak_in_flight == 1 and right.peak_in_flight == 1


def test_native_failure_is_not_reported_as_a_timeout():
    state = EventSetWaiting()
    lost = nvml.GpuIsLostError(nvml.Return.ERROR_GPU_IS_LOST)
    with pytest.raises(nvml.GpuIsLostError) as excinfo:
        asyncio.run(state.wait_async(FakeWait(error=lost), 100))
    assert excinfo.value is lost


def test_negative_timeout_is_rejected():
    state = EventSetWaiting()
    fake = FakeWait()
    with pytest.raises(ValueError, match="timeout_ms"):
        asyncio.run(state.wait_async(fake, -1))
    assert fake.calls == []


def test_parked_batch_is_delivered_without_loss():
    state = EventSetWaiting()
    state.park((BATCH, 0))
    native = []

    def convert(payload, size):
        batch, index = payload
        taken = batch[index : index + size]
        if index + len(taken) < len(batch):
            state.park((batch, index + len(taken)))
        return taken

    assert state.wait(None, 0, lambda payload: convert(payload, 2)) == BATCH[:2]
    assert state.wait(None, 0, lambda payload: convert(payload, 1)) == BATCH[2:3]
    assert state.wait(None, 0, lambda payload: convert(payload, 5)) == BATCH[3:]
    assert state.wait(native.append, 0) is None
    assert native == [0], "the drained batch should leave nothing parked"


def test_in_flight_wait_keeps_the_owner_alive():
    state = EventSetWaiting()
    fake = FakeWait(deliver_at=0, hold=True)
    owner = Owner(fake)
    reference = weakref.ref(owner)

    async def main(wait_for_event):
        task = asyncio.create_task(state.wait_async(wait_for_event, 0))
        await spin_until(lambda: len(fake.calls) == 1)
        fake.release.set()
        return await task

    # the bound method the coroutine holds is what keeps the owner alive
    assert asyncio.run(main(owner.native_wait)) is EVENT
    del owner
    gc.collect()
    assert reference() is None


def test_threads_are_reused_across_waits():
    state = EventSetWaiting()
    fake = FakeWait(deliver_at=0)
    seen = []

    async def main():
        for index in range(50):
            assert await state.wait_async(fake, 100) is EVENT
            if index in (5, 45):
                seen.append(sum(1 for thread in threading.enumerate() if thread.name.startswith("cuda-core-nvml")))

    asyncio.run(main())
    assert seen[0] == seen[1], f"workers grew while waiting: {seen}"
    assert state.is_waiting is False


def test_import_does_not_start_threads():
    code = "import cuda.core.system, threading; print(threading.active_count())"
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "1"


@pytest.mark.skipif(not system.CUDA_BINDINGS_NVML_IS_COMPATIBLE, reason="requires NVML-capable cuda-bindings")
def test_system_batch_remainder_is_preserved():
    try:
        events = system.register_events([SystemEventType.UNBIND])
    except system.UnknownError:
        pytest.skip("system events may only be registered once per process")
    batch = nvml.SystemEventData_v1(3)
    batch.event_type = [
        nvml.SystemEventType.GPU_DRIVER_BIND,
        nvml.SystemEventType.GPU_DRIVER_UNBIND,
        nvml.SystemEventType.GPU_DRIVER_BIND,
    ]
    batch.gpu_id = [0x100, 0x200, 0x300]

    head = events._take_batch((batch, 0), 2)
    assert len(head) == 2
    assert head[0].gpu_id == 0x100 and head[1].gpu_id == 0x200

    pending = events._waiting.take_pending()
    assert pending == (batch, 2)
    with pytest.raises(ValueError, match="buffer_size"):
        events._take_batch(pending, 0)

    tail = events._take_batch(pending, 2)
    assert len(tail) == 1 and tail[0].gpu_id == 0x300
    assert events._waiting.take_pending() is None


def quiet_event_set(index=0):
    """An event set that only fires on a GPU fault, so a wait always times out."""
    return system.Device(index=index).register_events([EventType.XID_CRITICAL_ERROR])


@pytest.mark.skipif(not system.CUDA_BINDINGS_NVML_IS_COMPATIBLE, reason="requires NVML-capable cuda-bindings")
def test_native_wait_async_honors_the_timeout_budget():
    events = quiet_event_set()
    started = time.monotonic()
    with pytest.raises(system.TimeoutError):
        asyncio.run(events.wait_async(timeout_ms=300))
    elapsed = time.monotonic() - started
    # Timings may be potentially flaky on loaded runners.
    # Remove if we see flaky tests in CI.
    assert 0.3 <= elapsed < 1.0, elapsed
    with pytest.raises(system.TimeoutError):
        events.wait(timeout_ms=10)  # the lease was released


@pytest.mark.skipif(not system.CUDA_BINDINGS_NVML_IS_COMPATIBLE, reason="requires NVML-capable cuda-bindings")
def test_native_cancel_drains_a_real_wait():
    events = quiet_event_set()
    elapsed = None

    async def main():
        nonlocal elapsed
        task = asyncio.create_task(events.wait_async())
        await asyncio.sleep(0.05)
        started = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        elapsed = time.monotonic() - started

    asyncio.run(main())
    # Timings may be potentially flaky on loaded runners.
    # Remove if we see flaky tests in CI.
    assert elapsed < 0.5, f"cancel returned only after {elapsed:.3f}s"
    with pytest.raises(system.TimeoutError):
        events.wait(timeout_ms=10)


@pytest.mark.skipif(not system.CUDA_BINDINGS_NVML_IS_COMPATIBLE, reason="requires NVML-capable cuda-bindings")
def test_native_two_event_sets_are_independent():
    if system.get_num_devices() < 2:
        pytest.skip("requires two devices")
    left, right = quiet_event_set(0), quiet_event_set(1)

    async def main():
        tasks = [asyncio.create_task(left.wait_async()), asyncio.create_task(right.wait_async())]
        await asyncio.sleep(0.05)
        assert left._waiting.is_waiting and right._waiting.is_waiting
        for task in tasks:
            task.cancel()
        for task in tasks:
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(main())
    assert not left._waiting.is_waiting and not right._waiting.is_waiting
    with pytest.raises(system.TimeoutError):
        left.wait(timeout_ms=10)
    with pytest.raises(system.TimeoutError):
        right.wait(timeout_ms=10)


@pytest.mark.skipif(not system.CUDA_BINDINGS_NVML_IS_COMPATIBLE, reason="requires NVML-capable cuda-bindings")
@pytest.mark.agent_authored(model="gpt-6")
def test_native_sync_wait_is_serialized_with_an_async_wait():
    events = quiet_event_set()

    async def main():
        task = asyncio.create_task(events.wait_async())
        await asyncio.sleep(0.05)
        waiting = asyncio.create_task(asyncio.to_thread(events.wait, timeout_ms=10))
        try:
            await asyncio.sleep(0.02)
            assert not waiting.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(system.TimeoutError):
                await waiting
        finally:
            task.cancel()
            await asyncio.gather(task, waiting, return_exceptions=True)

    asyncio.run(main())


# ============================================================================
# Resource accounting, scheduling and cross-loop reuse
# ============================================================================


def test_registration_failure_frees_the_event_set_once(monkeypatch):
    """a failure after the native set exists must not leak or double free."""
    gc.collect()  # drain sets owned by earlier tests before the fake allocator
    created, freed = [], []

    def create():
        created.append(0x5150)
        return 0x5150

    def register(*args):
        raise nvml.InvalidArgumentError(nvml.Return.ERROR_INVALID_ARGUMENT)

    monkeypatch.setattr(nvml, "event_set_create", create)
    monkeypatch.setattr(nvml, "event_set_free", freed.append)
    monkeypatch.setattr(nvml, "device_register_events", register)

    with pytest.raises(nvml.InvalidArgumentError):
        system.Device(index=0).register_events([EventType.CLOCK])
    gc.collect()
    assert created == [0x5150]
    assert freed.count(0x5150) == 1, f"our event set was freed {freed.count(0x5150)} times"


def test_saturated_dispatcher_serialises_slices(monkeypatch):
    """one worker serves one slice at a time, and queueing is inside the budget."""
    dispatcher = _Dispatcher(max_workers=1)
    monkeypatch.setattr(_async_events, "_dispatcher", dispatcher)

    holding = FakeWait(hold=True, deliver_at=0)
    queued = FakeWait(deliver_at=0)
    first, second = EventSetWaiting(), EventSetWaiting()

    async def main():
        held = asyncio.create_task(first.wait_async(holding, 0))
        await spin_until(lambda: len(holding.calls) == 1)
        waiting = asyncio.create_task(second.wait_async(queued, 400))
        await asyncio.sleep(0.05)
        assert queued.calls == [], "a queued slice must not reach the driver out of turn"
        holding.release.set()
        await held
        return await waiting

    started = time.monotonic()
    assert asyncio.run(main()) is EVENT
    elapsed_ms = (time.monotonic() - started) * 1000
    assert holding.peak_in_flight == 1, "the bound must hold"
    assert queued.peak_in_flight == 1
    assert elapsed_ms < 400 + 4 * _SLICE_MS, f"queueing must stay inside the budget ({elapsed_ms:.0f} ms)"


@pytest.mark.agent_authored(model="gpt-6")
def test_event_set_is_reusable_across_event_loops():
    """serial reuse across loops is fine; concurrent loops are serialized."""
    state = EventSetWaiting()
    fake = FakeWait(deliver_at=0)
    assert asyncio.run(state.wait_async(fake, 100)) is EVENT
    assert asyncio.run(state.wait_async(fake, 100)) is EVENT, "a fresh loop must not see a stale future"
    assert state.is_waiting is False

    holding = FakeWait(hold=True, deliver_at=0)
    outcome = {}

    def other_loop():
        async def run():
            outcome["result"] = await state.wait_async(holding, 0)

        asyncio.run(run())

    thread = threading.Thread(target=other_loop, name="other-loop")
    thread.start()
    try:
        while not holding.calls:
            time.sleep(0.001)
        with pytest.raises(system.TimeoutError):
            asyncio.run(state.wait_async(fake, 10))
    finally:
        holding.release.set()
        thread.join()
    assert outcome["result"] is EVENT
    assert asyncio.run(state.wait_async(fake, 100)) is EVENT


def test_parameter_bounds_match_the_signatures():
    """the type and range errors come from the annotated signatures."""
    events = system.Device(index=0).register_events([EventType.CLOCK])
    with pytest.raises(TypeError):
        events.wait(timeout_ms="soon")
    with pytest.raises(TypeError):
        asyncio.run(events.wait_async(timeout_ms="soon"))  # async def converts at await time
    with pytest.raises(ValueError, match="timeout_ms"):
        asyncio.run(events.wait_async(timeout_ms=-1))
    with pytest.raises(OverflowError):
        events.wait(timeout_ms=-1)


def test_dispatcher_bound_is_finite():
    assert 0 < _MAX_WORKERS <= 64


@pytest.mark.agent_authored(model="gpt-6")
def test_sync_waits_are_serialized_across_threads():
    from concurrent.futures import ThreadPoolExecutor

    state = EventSetWaiting()
    fake = FakeWait(hold=True, deliver_at=0)
    started = threading.Barrier(3)

    def wait():
        started.wait(timeout=5)
        return state.wait(fake, 0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.submit(wait), pool.submit(wait)
        try:
            started.wait(timeout=5)
            asyncio.run(spin_until(lambda: bool(fake.calls)))
            time.sleep(0.02)
            assert not first.done() and not second.done()
            assert fake.peak_in_flight == 1
        finally:
            fake.release.set()
        assert first.result(timeout=5) is EVENT
        assert second.result(timeout=5) is EVENT
    assert fake.peak_in_flight == 1


@pytest.mark.agent_authored(model="gpt-6")
def test_cancelling_a_queued_wait_does_not_borrow_the_event_set():
    state = EventSetWaiting()
    holding = FakeWait(hold=True, deliver_at=0)
    queued = FakeWait(deliver_at=0)

    async def main():
        first = asyncio.create_task(state.wait_async(holding, 0))
        await spin_until(lambda: bool(holding.calls))
        second = asyncio.create_task(state.wait_async(queued, 0))
        try:
            await asyncio.sleep(0.02)
            second.cancel()
            with pytest.raises(asyncio.CancelledError):
                await second
            assert queued.calls == []
            assert state.is_waiting
        finally:
            holding.release.set()
            await asyncio.gather(first, second, return_exceptions=True)

    asyncio.run(main())
    assert state.wait(queued, 0) is EVENT


@pytest.mark.parametrize("cancellation", ["cancel", "timeout", "task_group"])
@pytest.mark.agent_authored(model="gpt-6")
def test_cancelled_native_failure_is_preserved_for_the_next_wait(cancellation):
    if cancellation != "cancel" and sys.version_info < (3, 11):
        pytest.skip("asyncio.timeout and TaskGroup require Python 3.11")
    state = EventSetWaiting()
    lost = nvml.GpuIsLostError(nvml.Return.ERROR_GPU_IS_LOST)
    fake = FakeWait(error=lost, hold=True)

    async def main():
        if cancellation == "timeout":
            asyncio.get_running_loop().call_later(0.03, fake.release.set)
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.01):
                    await state.wait_async(fake, 0)
        elif cancellation == "task_group":
            async with asyncio.TaskGroup() as group:
                task = group.create_task(state.wait_async(fake, 0))
                await spin_until(lambda: bool(fake.calls))
                task.cancel()
                fake.release.set()
            assert task.cancelled()
        else:
            task = asyncio.create_task(state.wait_async(fake, 0))
            await spin_until(lambda: bool(fake.calls))
            task.cancel()
            await asyncio.sleep(0.01)
            task.cancel()
            fake.release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

    try:
        asyncio.run(main())
    finally:
        fake.release.set()
    assert fake.in_flight == 0 and not state.is_waiting
    idle = FakeWait(deliver_at=0)
    with pytest.raises(nvml.GpuIsLostError) as excinfo:
        state.wait(idle, 0)
    assert excinfo.value is lost
    assert idle.calls == []
    assert state.wait(idle, 0) is EVENT


@pytest.mark.parametrize("parked", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_conversion_failure_preserves_the_consumed_event(parked, asynchronous):
    state = EventSetWaiting()
    fake = FakeWait(deliver_at=0)
    if parked:
        state.park(EVENT)

    def convert(payload):
        raise ValueError("conversion failed")

    with pytest.raises(ValueError, match="conversion failed"):
        if asynchronous:
            asyncio.run(state.wait_async(fake, 100, convert))
        else:
            state.wait(fake, 100, convert)
    idle = FakeWait(deliver_at=0)
    assert state.wait(idle, 0) is EVENT
    assert idle.calls == []


@pytest.fixture
def fake_system_events(monkeypatch):
    # Zero denotes no native allocation, so teardown never frees a fake handle.
    monkeypatch.setattr(nvml, "system_event_set_create", lambda: 0)
    monkeypatch.setattr(nvml, "system_register_events", lambda *_args: None)
    return system.register_events([SystemEventType.UNBIND])


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_cancelled_system_batch_is_delivered_in_buffer_sized_pieces(
    fake_system_events, monkeypatch, count, asynchronous
):
    events = fake_system_events
    batch = nvml.SystemEventData_v1(count)
    batch.event_type = [nvml.SystemEventType.GPU_DRIVER_UNBIND] * count
    batch.gpu_id = list(range(0x100, 0x100 + count))
    started, release = threading.Event(), threading.Event()

    def native_wait(*args):
        started.set()
        assert release.wait(5)
        return batch

    monkeypatch.setattr(nvml, "system_event_set_wait", native_wait)

    async def cancel():
        task = asyncio.create_task(events.wait_async(buffer_size=count))
        try:
            await spin_until(started.is_set)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(cancel())

    def no_native_wait(*args):
        pytest.fail("a parked batch must not be fetched from NVML again")

    monkeypatch.setattr(nvml, "system_event_set_wait", no_native_wait)
    received = []
    for _ in range(count):
        result = asyncio.run(events.wait_async(buffer_size=1)) if asynchronous else events.wait(buffer_size=1)
        assert isinstance(result, SystemEvents)
        assert len(result) == 1
        received.append(result[0].gpu_id)
    assert received == list(range(0x100, 0x100 + count))
    assert events._waiting.take_pending() is None


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_invalid_buffer_size_does_not_consume_a_parked_system_batch(fake_system_events, asynchronous):
    events = fake_system_events
    batch = nvml.SystemEventData_v1(1)
    batch.event_type = [nvml.SystemEventType.GPU_DRIVER_UNBIND]
    batch.gpu_id = [0x100]
    events._waiting.park(batch)
    with pytest.raises(ValueError, match="buffer_size"):
        if asynchronous:
            asyncio.run(events.wait_async(buffer_size=0))
        else:
            events.wait(buffer_size=0)
    result = events.wait(buffer_size=1)
    assert isinstance(result, SystemEvents)
    assert len(result) == 1 and result[0].gpu_id == 0x100


@pytest.mark.agent_authored(model="gpt-6")
def test_fresh_sync_system_batch_uses_the_public_result_type(fake_system_events, monkeypatch):
    batch = nvml.SystemEventData_v1(1)
    batch.event_type = [nvml.SystemEventType.GPU_DRIVER_UNBIND]
    batch.gpu_id = [0x100]
    monkeypatch.setattr(nvml, "system_event_set_wait", lambda *_args: batch)
    result = fake_system_events.wait(timeout_ms=10)
    assert isinstance(result, SystemEvents)
    assert len(result) == 1 and result[0].gpu_id == 0x100


@pytest.mark.agent_authored(model="gpt-6")
def test_closed_loop_keeps_the_lease_until_the_native_slice_finishes(monkeypatch):
    failures = []
    monkeypatch.setattr(threading, "excepthook", failures.append)
    state = EventSetWaiting()
    fake = FakeWait(hold=True, deliver_at=0)
    loop = asyncio.new_event_loop()
    coroutine = state.wait_async(fake, 0)

    async def start():
        coroutine.send(None)
        await spin_until(lambda: bool(fake.calls))

    try:
        loop.run_until_complete(start())
        loop.close()
        coroutine.close()
        assert state.is_waiting
    finally:
        coroutine.close()
        loop.close()
        fake.release.set()
        asyncio.run(spin_until(lambda: not state.is_waiting))
    assert failures == []
    idle = FakeWait(deliver_at=0)
    assert state.wait(idle, 0) is EVENT
    assert idle.calls == []
    assert asyncio.run(state.wait_async(idle, 100)) is EVENT
