# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp

import pytest
from cuda.core import Device, EventOptions
from helpers.buffers import compare_equal_buffers, make_scratch_buffer
from helpers.latch import LatchKernel
from helpers.logging import TimestampedLogger

ENABLE_LOGGING = False  # Set True for test debugging and development
CHILD_TIMEOUT_SEC = 30
NBYTES = 64


class TestEventIpc:
    """Check the basic usage of IPC-enabled events with a latch kernel."""

    @pytest.mark.flaky(reruns=2)
    def test_main(self, ipc_device, ipc_memory_resource):
        log = TimestampedLogger(prefix="parent: ", enabled=ENABLE_LOGGING)
        device = ipc_device
        mr = ipc_memory_resource
        stream1 = device.create_stream()
        # TODO: We pick a timeout here to ensure forward progress (it needs to be
        # less than CHILD_TIMEOUT_SEC) when a pinned memory resource is in use,
        # in which case the call to buffer.copy_from(...) below is a synchronous
        # operation that blocks the host. But calling the latch kernel here does not
        # make any sense. We should refactor this test.
        latch = LatchKernel(device, timeout_sec=5)

        # Start the child process.
        q_out, q_in = [mp.Queue() for _ in range(2)]
        process = mp.Process(target=self.child_main, args=(log, q_out, q_in))
        process.start()

        # Prepare scratch buffers.
        target = make_scratch_buffer(device, 0, NBYTES)
        ones = make_scratch_buffer(device, 1, NBYTES)
        twos = make_scratch_buffer(device, 2, NBYTES)

        # Allocate the buffer and send it to the child.
        buffer = mr.allocate(NBYTES, stream=stream1)
        log("sending buffer")
        q_out.put(buffer)

        # Stream 1:
        log("enqueuing latch kernel on stream1")
        latch.launch(stream1)
        log("enqueuing copy on stream1")
        buffer.copy_from(ones, stream=stream1)

        ipc_event_options = EventOptions(ipc_enabled=True)
        e = stream1.record(options=ipc_event_options)
        log(f"recorded event ({hex(e.handle)})")
        q_out.put(e)
        log("sent event")

        # Wait on the child.
        log("waiting for child")
        none = q_in.get(timeout=CHILD_TIMEOUT_SEC)
        assert none is None

        log("releasing stream1")
        latch.release()
        process.join(timeout=CHILD_TIMEOUT_SEC)
        assert process.exitcode == 0
        log("done")

        # Finish up.
        target.copy_from(buffer, stream=stream1)
        stream1.sync()
        assert compare_equal_buffers(target, twos)

    def child_main(self, log, q_in, q_out):
        log.prefix = " child: "
        log("ready")
        device = Device()
        device.set_current()
        stream2 = device.create_stream()
        twos = make_scratch_buffer(device, 2, NBYTES)
        buffer = q_in.get(timeout=CHILD_TIMEOUT_SEC)
        log("got buffer")
        e = q_in.get(timeout=CHILD_TIMEOUT_SEC)
        log(f"got event ({hex(e.handle)})")
        stream2.wait(e)
        log("enqueuing copy on stream2")
        buffer.copy_from(twos, stream=stream2)
        log("signaling parent")
        q_out.put(None)
        log("waiting")
        stream2.sync()
        log("done")


@pytest.mark.flaky(reruns=2)
def test_event_is_monadic(ipc_device):
    """Check that IPC-enabled events are always bound and cannot be reset."""
    device = ipc_device
    with pytest.raises(TypeError, match=r"^IPC-enabled events must be bound; use Stream.record for creation\.$"):
        device.create_event({"ipc_enabled": True})

    stream = device.create_stream()
    e = stream.record(options={"ipc_enabled": True})
    with pytest.raises(
        TypeError,
        match=r"^IPC-enabled events should not be re-recorded, instead create a new event by supplying options\.$",
    ):
        stream.record(e)


@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize(
    "options", [{"ipc_enabled": True, "enable_timing": True}, EventOptions(ipc_enabled=True, enable_timing=True)]
)
def test_event_timing_disabled(ipc_device, options):
    """Check that IPC-enabled events cannot be created with timing enabled."""
    device = ipc_device
    stream = device.create_stream()
    with pytest.raises(TypeError, match=r"^IPC-enabled events cannot use timing\.$"):
        stream.record(options=options)


class TestIpcEventProperties:
    """
    Check that event properties are properly set after transfer to a child
    process.
    """

    @pytest.mark.flaky(reruns=2)
    @pytest.mark.parametrize("busy_waited_sync", [True, False])
    @pytest.mark.parametrize("use_options_cls", [True, False])
    @pytest.mark.parametrize("use_option_kw", [True, False])
    def test_main(self, ipc_device, busy_waited_sync, use_options_cls, use_option_kw):
        device = ipc_device
        stream = device.create_stream()

        # Start the child process.
        q_out, q_in = [mp.Queue() for _ in range(2)]
        process = mp.Process(target=self.child_main, args=(q_out, q_in))
        process.start()

        # Create an event and send it.
        options = (
            EventOptions(ipc_enabled=True, busy_waited_sync=busy_waited_sync)
            if use_options_cls
            else {"ipc_enabled": True, "busy_waited_sync": busy_waited_sync}
        )
        e = stream.record(options=options) if use_option_kw else stream.record(None, options)
        q_out.put(e)

        # Check its properties.
        props = q_in.get(timeout=CHILD_TIMEOUT_SEC)
        assert props[0] == e.get_ipc_descriptor()
        assert props[1] == e.is_ipc_enabled
        assert props[2] == e.is_timing_disabled
        assert props[3] == e.is_sync_busy_waited
        assert props[4] is None
        assert props[5] is None

        process.join(timeout=CHILD_TIMEOUT_SEC)
        assert process.exitcode == 0

    def child_main(self, q_in, q_out):
        device = Device()
        device.set_current()

        # Get the event.
        e = q_in.get(timeout=CHILD_TIMEOUT_SEC)

        # Send its properties.
        props = (
            e.get_ipc_descriptor(),
            e.is_ipc_enabled,
            e.is_timing_disabled,
            e.is_sync_busy_waited,
            e.device,
            e.context,
        )
        q_out.put(props)
