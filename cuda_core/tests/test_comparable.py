# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for __eq__ and __ne__ implementations in cuda.core classes.

These tests verify multi-type equality behavior and subclassing equality behavior
across Device, Stream, Event, and Context objects.
"""

from cuda.core.experimental import Device, Stream
from cuda.core.experimental._context import Context
from cuda.core.experimental._event import Event, EventOptions
from cuda.core.experimental._stream import StreamOptions

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
    assert not (device != my_device)


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
    assert not (stream != stream2)

    # Create subclass instance with different handle
    my_stream = MyStream._init(options=StreamOptions(), device_id=device.device_id)

    # Different handles -> not equal
    assert stream != my_stream, "Streams with different handles are not equal"
    assert not (stream == my_stream)
    assert stream.handle != my_stream.handle

    # sanity check: base and subclass compare equal (and hash equal)
    stream_from_handle = MyStream.from_handle(int(my_stream.handle))
    assert my_stream == stream_from_handle, "MyStream and Stream wrapping same handle compare equal"
    assert not (my_stream != stream_from_handle)
    assert hash(my_stream) == hash(stream_from_handle)


def test_event_subclass_equality(init_cuda):
    """Test Event subclass equality behavior.

    Event uses isinstance() for equality checking, similar to Stream.
    """

    class MyEvent(Event):
        pass

    device = Device(0)
    device.set_current()

    # Create two different events
    event = Event._init(device.device_id, device.context, options=EventOptions())
    my_event = MyEvent._init(device.device_id, device.context, options=EventOptions())

    # Different events should not be equal (different handles)
    assert event != my_event, "Different Event instances are not equal"
    assert not (event == my_event)

    # Same subclass type with different handles
    my_event2 = MyEvent._init(device.device_id, device.context, options=EventOptions())
    assert my_event != my_event2, "Different MyEvent instances are not equal"


def test_context_subclass_equality(init_cuda):
    """Test Context subclass equality behavior."""

    class MyContext(Context):
        pass

    device = Device(0)
    device.set_current()
    stream = device.create_stream()
    context = stream.context

    # MyContext._from_ctx() returns a Context instance, not MyContext
    my_context = MyContext._from_ctx(context._handle, device.device_id)
    assert type(my_context) is Context, "_from_ctx returns Context, not subclass"
    assert type(my_context) is not MyContext

    # Since both are Context instances with same handle, they're equal
    assert context == my_context, "Context instances with same handle are equal"
    assert not (context != my_context)

    # Create another context from different stream
    stream2 = device.create_stream()
    context2 = stream2.context

    # Same device, same primary context, should be equal
    assert context == context2, "Contexts from same device are equal"


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
