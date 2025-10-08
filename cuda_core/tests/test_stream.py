# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from cuda.core.experimental import Device, Stream, StreamOptions
from cuda.core.experimental._event import Event
from cuda.core.experimental._stream import LEGACY_DEFAULT_STREAM, PER_THREAD_DEFAULT_STREAM
from cuda.core.experimental._utils.cuda_utils import driver


def test_stream_init_disabled():
    with pytest.raises(RuntimeError, match=r"^Stream objects cannot be instantiated directly\."):
        Stream()  # Reject at front door.


def test_stream_init_with_options(init_cuda):
    stream = Device().create_stream(options=StreamOptions(nonblocking=True, priority=0))
    assert stream.is_nonblocking is True
    assert stream.priority == 0


def test_stream_handle(init_cuda):
    stream = Device().create_stream(options=StreamOptions())
    assert isinstance(stream.handle, driver.CUstream)


def test_stream_is_nonblocking(init_cuda):
    stream = Device().create_stream(options=StreamOptions(nonblocking=True))
    assert stream.is_nonblocking is True


def test_stream_priority(init_cuda):
    stream = Device().create_stream(options=StreamOptions(priority=0))
    assert stream.priority == 0
    stream = Device().create_stream(options=StreamOptions(priority=-1))
    assert stream.priority == -1
    with pytest.raises(ValueError):
        stream = Device().create_stream(options=StreamOptions(priority=1))


def test_stream_sync(init_cuda):
    stream = Device().create_stream(options=StreamOptions())
    stream.sync()  # Should not raise any exceptions


def test_stream_record(init_cuda):
    stream = Device().create_stream(options=StreamOptions())
    event = stream.record()
    assert isinstance(event, Event)


def test_stream_record_invalid_event(init_cuda):
    stream = Device().create_stream(options=StreamOptions())
    with pytest.raises(TypeError):
        stream.record(event="invalid_event")


def test_stream_wait_event(init_cuda):
    s1 = Device().create_stream()
    s2 = Device().create_stream()
    e1 = s1.record()
    s2.wait(e1)  # Should not raise any exceptions
    s2.sync()


def test_stream_wait_invalid_event(init_cuda):
    stream = Device().create_stream(options=StreamOptions())
    with pytest.raises(ValueError):
        stream.wait(event_or_stream="invalid_event")


def test_stream_device(init_cuda):
    stream = Device().create_stream(options=StreamOptions())
    device = stream.device
    assert isinstance(device, Device)


def test_stream_context(init_cuda):
    stream = Device().create_stream(options=StreamOptions())
    context = stream.context
    assert context is not None
    assert context._handle is not None


def test_stream_from_foreign_stream(init_cuda):
    device = Device()
    other_stream = device.create_stream(options=StreamOptions())
    stream = device.create_stream(obj=other_stream)
    # convert to int to work around NVIDIA/cuda-python#465
    assert int(other_stream.handle) == int(stream.handle)
    device = stream.device
    assert isinstance(device, Device)
    context = stream.context
    assert context is not None


def test_stream_from_handle():
    stream = Stream.from_handle(0)
    assert isinstance(stream, Stream)


def test_legacy_default_stream():
    assert isinstance(LEGACY_DEFAULT_STREAM, Stream)


def test_per_thread_default_stream():
    assert isinstance(PER_THREAD_DEFAULT_STREAM, Stream)


def test_stream_subclassing(init_cuda):
    class MyStream(Stream):
        pass

    dev = Device()
    dev.set_current()
    stream = MyStream._init(options=StreamOptions(), device_id=dev.device_id)
    assert isinstance(stream, MyStream)


def test_stream_legacy_default_subclassing():
    class MyStream(Stream):
        pass

    stream = MyStream._legacy_default()
    assert isinstance(stream, MyStream)


def test_stream_per_thread_default_subclassing():
    class MyStream(Stream):
        pass

    stream = MyStream._per_thread_default()
    assert isinstance(stream, MyStream)
