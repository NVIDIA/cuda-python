# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
import sys
import threading
import time
import weakref
from contextlib import closing

import pytest
from helpers.misc import StreamWrapper

from cuda.core import Device, LegacyPinnedMemoryResource, host_launch

N = 16


def _as_bytes(buffer):
    return bytes((ctypes.c_uint8 * N).from_address(int(buffer.handle)))


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_host_launch_orders_between_prior_and_later_work(init_cuda):
    dev = Device(0)
    dev.set_current()
    seen = []
    with (
        closing(dev.create_stream()) as stream,
        dev.allocate(N, stream=stream) as device_buffer,
        LegacyPinnedMemoryResource().allocate(N) as host_buffer,
    ):
        device_buffer.fill(0x11, stream=stream)
        device_buffer.copy_to(host_buffer, stream=stream)
        result = host_launch(stream, lambda: seen.append(_as_bytes(host_buffer)))
        device_buffer.fill(0x22, stream=stream)
        device_buffer.copy_to(host_buffer, stream=stream)
        stream.sync()

        assert result is None
        assert seen == [b"\x11" * N]
        assert _as_bytes(host_buffer) == b"\x22" * N


@pytest.mark.thread_unsafe(reason="the release is a main-thread pending call")
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_host_launch_releases_callable_after_it_ran(init_cuda):
    dev = Device(0)
    dev.set_current()
    ran = []

    class Callback:
        def __call__(self):
            ran.append(True)

    with closing(dev.create_stream()) as stream:
        callback = Callback()
        ref = weakref.ref(callback)
        host_launch(stream, callback)
        del callback
        stream.sync()
        assert ran == [True]

        deadline = time.monotonic() + 5
        while ref() is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ref() is None, "host_launch did not release the callable after it ran"


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_host_launch_ctypes_bytes_user_data_is_copied(init_cuda):
    """The bytes are copied at launch; a later change to the caller's buffer is not seen."""
    dev = Device(0)
    dev.set_current()
    seen = []
    gate = threading.Event()

    @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    def callback(data):
        seen.append(ctypes.string_at(data, N))

    with closing(dev.create_stream()) as stream:
        data = bytearray(range(N))
        try:
            # Hold the stream on the driver thread (Event.wait releases the GIL) while
            # the caller overwrites its buffer; the copy made at launch is unaffected.
            host_launch(stream, gate.wait)
            host_launch(stream, callback, user_data=data)
            data[:] = b"\xff" * N
        finally:
            gate.set()
        stream.sync()
        assert seen == [bytes(range(N))]


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_host_launch_ctypes_int_user_data_passes_through(init_cuda):
    dev = Device(0)
    dev.set_current()
    seen = []
    buf = ctypes.create_string_buffer(b"\x5a")
    pointer = ctypes.addressof(buf)

    @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    def callback(data):
        seen.append((data, ctypes.c_uint8.from_address(data).value))

    with closing(dev.create_stream()) as stream:
        host_launch(stream, callback, user_data=pointer)
        stream.sync()
        assert seen == [(pointer, 0x5A)]


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_host_launch_rejects_user_data_for_python_callable(init_cuda):
    dev = Device(0)
    dev.set_current()
    with closing(dev.create_stream()) as stream, pytest.raises(ValueError):
        host_launch(stream, lambda: None, user_data=b"data")


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_host_launch_rejects_non_callable(init_cuda):
    dev = Device(0)
    dev.set_current()
    with closing(dev.create_stream()) as stream, pytest.raises(TypeError):
        host_launch(stream, 42)


@pytest.mark.parametrize("prototype", [ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p), ctypes.CFUNCTYPE(None)])
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_host_launch_rejects_wrong_ctypes_prototype(init_cuda, prototype):
    dev = Device(0)
    dev.set_current()
    callback = prototype(lambda *_args: 0)
    with closing(dev.create_stream()) as stream, pytest.raises(TypeError):
        host_launch(stream, callback)


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_host_launch_forwards_graph_builder(init_cuda):
    dev = Device(0)
    dev.set_current()
    ran = []
    with closing(dev.create_stream()) as stream:
        gb = stream.create_graph_builder().begin_building()
        host_launch(gb, lambda: ran.append(True))
        gb.end_building()
        with closing(gb.complete()) as graph:
            graph.upload(stream)
            graph.launch(stream)
            graph.launch(stream)
            stream.sync()
            assert ran == [True, True]


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_host_launch_rejects_capturing_stream(init_cuda):
    dev = Device(0)
    dev.set_current()
    ran = []
    with closing(dev.create_stream()) as stream:
        gb = stream.create_graph_builder().begin_building()
        try:
            with pytest.raises(RuntimeError, match="capturing"):
                host_launch(gb.stream, lambda: ran.append(True))
        finally:
            gb.end_building()
        graph = gb.complete()
        graph.close()
        assert ran == []


@pytest.mark.thread_unsafe(reason="replaces the process-global sys.unraisablehook")
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_host_launch_python_exception_is_reported_as_unraisable(init_cuda, monkeypatch):
    dev = Device(0)
    dev.set_current()
    records = []
    monkeypatch.setattr(sys, "unraisablehook", records.append)
    ran = []

    def throws():
        raise RuntimeError("callback raised")

    with closing(dev.create_stream()) as stream:
        host_launch(stream, throws)
        stream.sync()
        host_launch(stream, lambda: ran.append(True))
        stream.sync()
        assert ran == [True]
        assert [record.exc_type for record in records] == [RuntimeError]
        assert "callback raised" in str(records[0].exc_value)


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_host_launch_two_streams_keep_their_own_data(init_cuda):
    dev = Device(0)
    dev.set_current()
    seen1, seen2 = [], []
    with (
        closing(dev.create_stream()) as stream1,
        closing(dev.create_stream()) as stream2,
        dev.allocate(N, stream=stream1) as device_buffer1,
        dev.allocate(N, stream=stream2) as device_buffer2,
        LegacyPinnedMemoryResource().allocate(N) as host_buffer1,
        LegacyPinnedMemoryResource().allocate(N) as host_buffer2,
    ):
        device_buffer1.fill(0x33, stream=stream1)
        device_buffer1.copy_to(host_buffer1, stream=stream1)
        host_launch(stream1, lambda: seen1.append(_as_bytes(host_buffer1)))
        device_buffer2.fill(0x44, stream=stream2)
        device_buffer2.copy_to(host_buffer2, stream=stream2)
        host_launch(stream2, lambda: seen2.append(_as_bytes(host_buffer2)))
        stream1.sync()
        stream2.sync()
        assert seen1 == [b"\x33" * N]
        assert seen2 == [b"\x44" * N]


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_host_launch_rejects_none_stream(init_cuda):
    with pytest.raises(TypeError):
        host_launch(None, lambda: None)


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_host_launch_accepts_stream_protocol_with_deprecation(init_cuda):
    dev = Device(0)
    dev.set_current()
    ran = []
    with closing(dev.create_stream()) as stream:
        with pytest.warns(DeprecationWarning, match="stream protocol is deprecated"):
            host_launch(StreamWrapper(stream), lambda: ran.append(True))
        stream.sync()
        assert ran == [True]
