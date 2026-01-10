# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for __eq__ and __ne__ implementations in cuda.core classes.

These tests verify multi-type equality behavior and subclassing equality behavior
across Device, Stream, Event, and Context objects.
"""

from cuda.core import Device, Stream
from cuda.core._stream import StreamOptions

# ============================================================================
# Equality Contract Tests
# ============================================================================


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


# ============================================================================
# Subclassing Equality Tests
# ============================================================================


def test_device_subclass_equality(init_cuda):
    """Test Device subclass equality behavior.

    Device uses a singleton pattern where Device(0) always returns the same
    cached instance. This means subclassing Device doesn't create new instances;
    MyDevice(0) returns the original Device(0) instance from the cache.
    """

    class MyDevice(Device):
        pass

    device = Device(0)
    device.set_current()
    my_device = MyDevice(0)

    # Due to singleton pattern, both return the exact same instance
    assert device is my_device, "Device singleton returns same instance for same device_id"
    assert type(device) is Device, "Singleton returns original Device type, not subclass"
    assert type(my_device) is Device, "Even MyDevice(0) returns Device instance due to singleton"

    # Since they're the same object, they're equal
    assert device == my_device


def test_stream_subclass_equality(init_cuda):
    """Test Stream subclass equality behavior.

    Stream uses isinstance() for equality checking, which means a Stream instance
    and a MyStream subclass instance wrapping the same handle will compare equal.
    """

    class MyStream(Stream):
        pass

    device = Device(0)
    device.set_current()

    # Create base Stream instance
    stream = Stream._init(options=StreamOptions(), device_id=device.device_id)

    # Create another Stream wrapping same handle
    stream2 = Stream.from_handle(int(stream.handle))
    assert stream == stream2, "Streams wrapping same handle are equal"

    # Create subclass instance with different handle
    my_stream = MyStream._init(options=StreamOptions(), device_id=device.device_id)

    # Different handles -> not equal
    assert stream != my_stream, "Streams with different handles are not equal"
    assert stream.handle != my_stream.handle

    # sanity check: base and subclass compare equal (and hash equal)
    stream_from_handle = MyStream.from_handle(int(my_stream.handle))
    assert my_stream == stream_from_handle, "MyStream and Stream wrapping same handle compare equal"
    assert hash(my_stream) == hash(stream_from_handle)


def test_event_subclass_equality(init_cuda):
    """Test Event subclass equality behavior.

    Event uses isinstance() for equality checking, similar to Stream.
    """
    device = Device(0)
    device.set_current()

    # Create events using public API
    event1 = device.create_event()
    event2 = device.create_event()
    event3 = device.create_event()

    # Different events should not be equal (different handles)
    assert event1 != event2, "Different Event instances are not equal"
    assert event2 != event3, "Different Event instances are not equal"


def test_context_equality(init_cuda):
    """Test Context equality behavior."""
    device = Device(0)
    device.set_current()

    # Get context from different sources
    stream1 = device.create_stream()
    stream2 = device.create_stream()
    context1 = stream1.context
    context2 = stream2.context
    device_context = device.context

    # Same device, same primary context, should be equal
    assert context1 == context2, "Contexts from same device are equal"
    assert context1 == device_context, "Stream context equals device context"


def test_subclass_type_safety(init_cuda):
    """Test that equality checks with incompatible types return False or NotImplemented."""
    device = Device(0)
    device.set_current()

    stream = device.create_stream()
    event = stream.record()
    context = stream.context

    # None of these should be equal to each other
    assert device != stream
    assert device != event
    assert device != context
    assert stream != event
    assert stream != context
    assert event != context

    # None should be equal to arbitrary types
    assert device != "device"
    assert stream != 123
    assert event != []
    assert context != {"key": "value"}
