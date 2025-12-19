# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math

import cuda.core
import pytest
from cuda.core import (
    Device,
    Event,
    EventOptions,
)
from helpers.latch import LatchKernel
from helpers.nanosleep_kernel import NanosleepKernel


def test_event_init_disabled():
    with pytest.raises(RuntimeError, match=r"^Event objects cannot be instantiated directly\."):
        cuda.core._event.Event()  # Ensure back door is locked.


def test_timing_success(init_cuda):
    options = EventOptions(enable_timing=True)
    device = Device()
    stream = device.create_stream()

    # Create a nanosleep kernel that sleeps for 20 ms to ensure a measurable delay.
    # This guarantees elapsed_time_ms > 10 without depending on OS/driver timing characteristics.
    nanosleep = NanosleepKernel(device, sleep_duration_ms=20)

    e1 = stream.record(options=options)
    nanosleep.launch(stream)  # Insert a guaranteed delay
    e2 = stream.record(options=options)
    e2.sync()
    elapsed_time_ms = e2 - e1
    assert isinstance(elapsed_time_ms, float)
    # Sanity check: cuEventElapsedTime should always return a finite float for two completed
    # events. This guards against unexpected driver/HW anomalies (e.g. NaN or inf) or general
    # undefined behavior, without asserting anything about the magnitude of the measured time.
    assert math.isfinite(elapsed_time_ms)
    # With the nanosleep kernel between events, the kernel sleeps for 20 ms using clock64(),
    # so elapsed_time_ms should definitely be larger than 10 ms. This provides a large safety
    # margin above the ~0.5 microsecond resolution of cudaEventElapsedTime(), which should
    # make this test deterministic and non-flaky.
    assert elapsed_time_ms > 10


def test_is_sync_busy_waited(init_cuda):
    options = EventOptions(enable_timing=False, busy_waited_sync=True)
    stream = Device().create_stream()
    event = stream.record(options=options)
    assert event.is_sync_busy_waited is True

    options = EventOptions(enable_timing=False)
    stream = Device().create_stream()
    event = stream.record(options=options)
    assert event.is_sync_busy_waited is False


def test_sync(init_cuda):
    options = EventOptions(enable_timing=False)
    stream = Device().create_stream()
    event = stream.record(options=options)
    event.sync()
    assert event.is_done is True


def test_is_done(init_cuda):
    options = EventOptions(enable_timing=False)
    stream = Device().create_stream()
    event = stream.record(options=options)
    # Without a sync, the captured work might not have yet completed
    # Therefore this check should never raise an exception
    assert event.is_done in (True, False)


def test_error_timing_disabled():
    device = Device()
    device.set_current()
    enabled = EventOptions(enable_timing=True)
    disabled = EventOptions(enable_timing=False)
    stream = device.create_stream()

    event1 = stream.record(options=enabled)
    event2 = stream.record(options=disabled)
    assert not event1.is_timing_disabled
    assert event2.is_timing_disabled
    stream.sync()
    with pytest.raises(RuntimeError, match="^Both Events must be created with timing enabled"):
        event2 - event1

    event1 = stream.record(options=disabled)
    event2 = stream.record(options=disabled)
    stream.sync()
    with pytest.raises(RuntimeError, match="^Both Events must be created with timing enabled"):
        event2 - event1


def test_error_timing_recorded():
    device = Device()
    device.set_current()
    enabled = EventOptions(enable_timing=True)
    stream = device.create_stream()

    event1 = stream.record(options=enabled)
    event2 = device.create_event(options=enabled)
    event3 = device.create_event(options=enabled)

    stream.sync()
    with pytest.raises(RuntimeError, match="^Both Events must be recorded"):
        event2 - event1
    with pytest.raises(RuntimeError, match="^Both Events must be recorded"):
        event1 - event2
    with pytest.raises(RuntimeError, match="^Both Events must be recorded"):
        event3 - event2


def test_error_timing_incomplete():
    device = Device()
    device.set_current()
    latch = LatchKernel(device)
    enabled = EventOptions(enable_timing=True)
    stream = device.create_stream()

    event1 = stream.record(options=enabled)
    latch.launch(stream)
    event3 = stream.record(options=enabled)

    # event3 will never complete because the latch has not been released
    with pytest.raises(RuntimeError, match="^One or both events have not completed."):
        event3 - event1

    latch.release()
    event3.sync()
    event3 - event1  # this should work


def test_event_device(init_cuda):
    device = Device()
    event = device.create_event(options=EventOptions())
    assert event.device is device


def test_event_context(init_cuda):
    event = Device().create_event(options=EventOptions())
    context = event.context
    assert context is not None


def test_event_subclassing():
    class MyEvent(Event):
        pass

    dev = Device()
    dev.set_current()
    event = MyEvent._init(dev.device_id, dev.context)
    assert isinstance(event, MyEvent)


# ============================================================================
# Event Equality Tests
# ============================================================================


def test_event_equality_reflexive(init_cuda):
    """Event should equal itself (reflexive property)."""
    device = Device()
    stream = device.create_stream()
    event = stream.record()

    assert event == event, "Event should equal itself"


def test_event_inequality_different_events(init_cuda):
    """Different events should not be equal."""
    device = Device()
    stream = device.create_stream()

    e1 = stream.record()
    e2 = stream.record()

    assert e1 != e2, "Different events should not be equal"


def test_event_type_safety(init_cuda):
    """Comparing Event with wrong type should return False."""
    device = Device()
    stream = device.create_stream()
    event = stream.record()

    assert (event == "not an event") is False
    assert (event == 123) is False
    assert (event is None) is False


# ============================================================================
# Event Hash Tests
# ============================================================================


def test_event_hash_consistency(init_cuda):
    """Hash of same Event object should be consistent."""
    device = Device()
    stream = device.create_stream()
    event = stream.record()

    hash1 = hash(event)
    hash2 = hash(event)
    assert hash1 == hash2, "Hash should be consistent for same object"


def test_event_hash_equality(init_cuda):
    """Events with same underlying handle should hash equal."""
    device = Device()
    stream = device.create_stream()

    # Create events
    e1 = stream.record()
    e2 = stream.record()

    # Different events should have different hashes
    assert e1 != e2
    assert hash(e1) != hash(e2)

    # Same event should equal itself
    assert e1 == e1
    assert hash(e1) == hash(e1)


def test_event_dict_key(init_cuda):
    """Events should be usable as dictionary keys."""
    device = Device()
    stream = device.create_stream()

    e1 = stream.record()
    e2 = stream.record()

    # Use events as keys
    event_cache = {e1: "timing1", e2: "timing2"}

    assert len(event_cache) == 2
    assert event_cache[e1] == "timing1"
    assert event_cache[e2] == "timing2"


def test_event_set_membership(init_cuda):
    """Events should work correctly in sets."""
    device = Device()
    stream = device.create_stream()

    e1 = stream.record()
    e2 = stream.record()

    event_set = {e1, e2}
    assert len(event_set) == 2

    # Same event should not add duplicate
    event_set.add(e1)
    assert len(event_set) == 2
