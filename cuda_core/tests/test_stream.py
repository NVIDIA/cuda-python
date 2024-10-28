# Copyright 2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

from cuda.core.experimental._stream import Stream, StreamOptions, LEGACY_DEFAULT_STREAM, PER_THREAD_DEFAULT_STREAM, default_stream
from cuda.core.experimental._event import Event, EventOptions
from cuda.core.experimental._device import Device
import pytest

def test_stream_init():
    with pytest.raises(NotImplementedError):
        Stream()

def test_stream_init_with_options():
    stream = Stream._init(options=StreamOptions(nonblocking=True, priority=0))
    assert stream.is_nonblocking is True
    assert stream.priority == 0

def test_stream_handle():
    stream = Stream._init(options=StreamOptions())
    assert isinstance(stream.handle, int)

def test_stream_is_nonblocking():
    stream = Stream._init(options=StreamOptions(nonblocking=True))
    assert stream.is_nonblocking is True

def test_stream_priority():
    stream = Stream._init(options=StreamOptions(priority=0))
    assert stream.priority == 0
    stream = Stream._init(options=StreamOptions(priority=-1))
    assert stream.priority == -1
    with pytest.raises(ValueError):
        stream = Stream._init(options=StreamOptions(priority=1))

def test_stream_sync():
    stream = Stream._init(options=StreamOptions())
    stream.sync()  # Should not raise any exceptions

def test_stream_record():
    stream = Stream._init(options=StreamOptions())
    event = stream.record()
    assert isinstance(event, Event)

def test_stream_record_invalid_event():
    stream = Stream._init(options=StreamOptions())
    with pytest.raises(TypeError):
        stream.record(event="invalid_event")

def test_stream_wait_event():
    stream = Stream._init(options=StreamOptions())
    event = Event._init()
    stream.record(event)
    stream.wait(event)  # Should not raise any exceptions

def test_stream_wait_invalid_event():
    stream = Stream._init(options=StreamOptions())
    with pytest.raises(ValueError):
        stream.wait(event_or_stream="invalid_event")

def test_stream_device():
    stream = Stream._init(options=StreamOptions())
    device = stream.device
    assert isinstance(device, Device)

def test_stream_context():
    stream = Stream._init(options=StreamOptions())
    context = stream.context
    assert context is not None

def test_stream_from_handle():
    stream = Stream.from_handle(0)
    assert isinstance(stream, Stream)

def test_legacy_default_stream():
    assert isinstance(LEGACY_DEFAULT_STREAM, Stream)

def test_per_thread_default_stream():
    assert isinstance(PER_THREAD_DEFAULT_STREAM, Stream)

def test_default_stream():
    stream = default_stream()
    assert isinstance(stream, Stream)
