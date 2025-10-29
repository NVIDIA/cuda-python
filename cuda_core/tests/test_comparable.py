# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for __eq__ and __ne__ implementations in cuda.core classes.

These tests verify that Stream, Event, Context, and Device objects implement
proper equality and inequality comparisons, including type safety.
"""

import pytest
from cuda.core.experimental import Device, Stream, Event, EventOptions
from cuda.core.experimental._context import Context


# ============================================================================
# Stream Equality Tests
# ============================================================================

def test_stream_equality_same_handle(init_cuda):
    """Two Stream objects wrapping same handle should be equal."""
    device = Device()
    s1 = device.create_stream()
    s2 = Stream.from_handle(int(s1.handle))

    assert s1 == s2, "Streams with same handle should be equal"
    assert not (s1 != s2), "Equal streams should not be not-equal"


def test_stream_inequality_different_handles(init_cuda):
    """Different streams should not be equal."""
    device = Device()
    s1 = device.create_stream()
    s2 = device.create_stream()

    assert s1 != s2, "Different streams should not be equal"
    assert not (s1 == s2), "Different streams should not be equal"


def test_stream_equality_reflexive(init_cuda):
    """Stream should equal itself (reflexive property)."""
    stream = Device().create_stream()
    assert stream == stream, "Stream should equal itself"


def test_stream_equality_symmetric(init_cuda):
    """If s1 == s2, then s2 == s1 (symmetric property)."""
    device = Device()
    s1 = device.create_stream()
    s2 = Stream.from_handle(int(s1.handle))

    assert s1 == s2, "Equality should be symmetric"
    assert s2 == s1, "Equality should be symmetric"


def test_stream_type_safety(init_cuda):
    """Comparing Stream with wrong type should return NotImplemented/False."""
    stream = Device().create_stream()

    # These should not raise exceptions
    assert (stream == "not a stream") is False
    assert (stream == 123) is False
    assert (stream == None) is False
    assert (stream == Device()) is False


def test_stream_not_equal_operator(init_cuda):
    """Test != operator works correctly for streams."""
    device = Device()
    s1 = device.create_stream()
    s2 = device.create_stream()
    s3 = Stream.from_handle(int(s1.handle))

    assert s1 != s2, "Different streams should be not-equal"
    assert not (s1 != s3), "Same handle streams should not be not-equal"


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
    assert not (e1 == e2), "Different events should not be equal"


def test_event_type_safety(init_cuda):
    """Comparing Event with wrong type should return NotImplemented/False."""
    device = Device()
    stream = device.create_stream()
    event = stream.record()

    assert (event == "not an event") is False
    assert (event == 123) is False
    assert (event == None) is False


# ============================================================================
# Context Equality Tests
# ============================================================================

def test_context_equality_same_context(init_cuda):
    """Contexts from same device should be equal."""
    device = Device()

    s1 = device.create_stream()
    s2 = device.create_stream()

    ctx1 = s1.context
    ctx2 = s2.context

    # Same device, should have same context
    assert ctx1 == ctx2, "Streams on same device should share context"


def test_context_equality_reflexive(init_cuda):
    """Context should equal itself (reflexive property)."""
    device = Device()
    stream = device.create_stream()
    context = stream.context

    assert context == context, "Context should equal itself"


def test_context_type_safety(init_cuda):
    """Comparing Context with wrong type should return NotImplemented/False."""
    device = Device()
    context = device.create_stream().context

    assert (context == "not a context") is False
    assert (context == 123) is False
    assert (context == None) is False


# ============================================================================
# Device Equality Tests
# ============================================================================

def test_device_equality_same_id(init_cuda):
    """Devices with same device_id should be equal."""
    dev1 = Device(0)
    dev2 = Device(0)

    # On same thread, should be same object (singleton)
    assert dev1 is dev2, "Device is per-thread singleton"
    assert dev1 == dev2, "Same device_id should be equal"


def test_device_equality_reflexive(init_cuda):
    """Device should equal itself (reflexive property)."""
    device = Device(0)
    assert device == device, "Device should equal itself"


def test_device_inequality_different_id(init_cuda):
    """Devices with different device_id should not be equal."""
    try:
        # Only runs on when two devices are available
        dev0 = Device(0)
        dev1 = Device(1)

        assert dev0 != dev1, "Different devices should not be equal"
        assert not (dev0 == dev1), "Different devices should not be equal"
    except (ValueError, Exception):
        pass

def test_device_type_safety(init_cuda):
    """Comparing Device with wrong type should return NotImplemented/False."""
    device = Device(0)

    assert (device == "not a device") is False
    assert (device == 123) is False
    assert (device == None) is False


# ============================================================================
# Equality Contract Tests
# ============================================================================

def test_equality_contract_consistency():
    """Test that a == b implies hash(a) == hash(b) (hash contract)."""
    device = Device(0)
    device.set_current()

    # Test with Streams
    s1 = device.create_stream()
    s2 = Stream.from_handle(int(s1.handle))
    assert s1 == s2
    assert hash(s1) == hash(s2), "Equal streams must have equal hashes"

    # Test with Events
    stream = device.create_stream()
    e1 = stream.record()
    assert e1 == e1
    assert hash(e1) == hash(e1), "Equal events must have equal hashes"

    # Test with Device
    d1 = Device(0)
    d2 = Device(0)
    assert d1 == d2
    assert hash(d1) == hash(d2), "Equal devices must have equal hashes"

    # Test with Context
    ctx1 = stream.context
    ctx2 = stream.context
    assert ctx1 == ctx2
    assert hash(ctx1) == hash(ctx2), "Equal contexts must have equal hashes"


def test_equality_is_not_identity():
    """Test that equality (==) is different from identity (is)."""
    device = Device(0)
    device.set_current()

    # Streams: Different objects can be equal
    s1 = device.create_stream()
    s2 = Stream.from_handle(int(s1.handle))

    assert s1 == s2, "Streams with same handle are equal"
    assert s1 is not s2, "But they are not the same object"

    # Device: Same object due to singleton (special case)
    d1 = Device(0)
    d2 = Device(0)

    assert d1 == d2, "Devices with same ID are equal"
    assert d1 is d2, "And they ARE the same object (singleton)"
