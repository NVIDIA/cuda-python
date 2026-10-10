# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the host-callback helper, from the #311 regression survey.

These add no API. They pin down behaviour of the existing helper that the suite
did not cover:

* how a Python exception raised inside a callback is reported (the trampoline is
  ``noexcept with gil``, so nothing may cross the C ABI),
* that a caller-owned raw ``user_data`` pointer reaches a ctypes callback
  unchanged,
* that two devices running their own graphs keep callback parameters and results
  separate.

The ordinary-stream callback entry point does not exist yet, so the stream side
of the #311 matrix stays blocked on that dependency and is not asserted here.
"""

import ctypes

import pytest

from cuda.core import Device, LegacyPinnedMemoryResource
from cuda.core.graph import GraphDefinition

N = 16


def _as_bytes(buffer):
    return bytes((ctypes.c_uint8 * N).from_address(int(buffer.handle)))


@pytest.mark.thread_unsafe(reason="replaces the process-global sys.unraisablehook")
def test_python_callback_exception_is_reported_as_unraisable(init_cuda):
    """A raising callback is reported, not propagated, and the graph still finishes."""
    import sys

    records = []
    previous_hook = sys.unraisablehook
    sys.unraisablehook = records.append
    ran = []
    dev = Device(0)
    dev.set_current()
    stream = dev.create_stream()
    device_buffer = dev.allocate(N, stream=stream)
    host_buffer = LegacyPinnedMemoryResource().allocate(N)

    def throws():
        ran.append("callback")
        raise RuntimeError("callback raised")

    try:
        graph_def = GraphDefinition()
        graph_def.memset(device_buffer, 0x33, N).callback(throws).memcpy(host_buffer, device_buffer, N)
        graph = graph_def.instantiate()
        graph.upload(stream)
        graph.launch(stream)
        stream.sync()

        later = []
        after_def = GraphDefinition()
        after_def.callback(lambda: later.append("later"))
        after = after_def.instantiate()
        after.upload(stream)
        after.launch(stream)
        stream.sync()
    finally:
        sys.unraisablehook = previous_hook

    assert ran == ["callback"]
    assert [record.exc_type for record in records] == [RuntimeError]
    assert "callback raised" in str(records[0].exc_value)
    # The node after the raising callback still ran: the launch and the sync
    # returned normally instead of turning the exception into a CUDA error.
    assert _as_bytes(host_buffer)[:4] == b"\x33" * 4
    assert later == ["later"]


def test_ctypes_callback_receives_caller_owned_user_data_pointer(init_cuda):
    """``user_data`` given as an int is the caller's pointer, passed through as-is."""
    dev = Device(0)
    dev.set_current()
    stream = dev.create_stream()
    payload = LegacyPinnedMemoryResource().allocate(4)
    bytes_view = (ctypes.c_uint8 * 4).from_address(int(payload.handle))
    bytes_view[:] = [0x5A, 0x00, 0x00, 0x00]
    raw_pointer = int(payload.handle)

    callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    seen = []

    @callback_type
    def read_byte(data):
        seen.append((data, ctypes.cast(data, ctypes.POINTER(ctypes.c_uint8))[0]))

    graph_def = GraphDefinition()
    graph_def.callback(read_byte, user_data=raw_pointer)
    graph = graph_def.instantiate()
    graph.upload(stream)
    graph.launch(stream)
    stream.sync()

    assert seen == [(raw_pointer, 0x5A)]
    # The caller keeps owning the storage; the graph copied nothing.
    bytes_view[:] = [0x00, 0x00, 0x00, 0x00]


def test_two_devices_keep_callback_parameters_separate(device_x2):
    """Two devices, two graphs each: no callback observes another one's value."""
    per_stream = 2
    slot_count = len(device_x2) * per_stream
    # One pinned allocation, cut into one distinct slot per (device, stream), so
    # the callbacks cannot alias each other's storage; the owner lives here.
    host_slots = LegacyPinnedMemoryResource().allocate(N * slot_count)
    host_base = int(host_slots.handle)
    runs = []
    streams = []
    device_buffers = []
    expected = []
    for device_index, dev in enumerate(device_x2):
        dev.set_current()
        for stream_index in range(per_stream):
            value = 0xA1 + device_index * 0x10 + stream_index
            slot = device_index * per_stream + stream_index
            expected.append(((device_index, stream_index), value))
            device_stream = dev.create_stream()
            device_buffer = dev.allocate(N, stream=device_stream)
            host_pointer = host_base + slot * N

            def snapshot(label=(device_index, stream_index), host_pointer=host_pointer):
                runs.append((label, ctypes.c_uint8.from_address(host_pointer).value))

            graph_def = GraphDefinition()
            graph_def.memset(device_buffer, value, N).memcpy(
                host_pointer, device_buffer, N, dst_owner=host_slots
            ).callback(snapshot)
            graph = graph_def.instantiate()
            graph.upload(device_stream)
            graph.launch(device_stream)
            streams.append(device_stream)
            device_buffers.append(device_buffer)

    for device_stream in streams:
        device_stream.sync()

    assert sorted(runs) == sorted(expected)
    assert len(runs) == slot_count
