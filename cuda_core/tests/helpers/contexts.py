# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

from cuda.core._utils.cuda_utils import driver, handle_return

__all__ = [
    "assert_device_operations_use_bound_context",
    "current_context_handle",
    "no_current_context",
    "use_context",
]


def current_context_handle():
    """Return the current CUDA context handle, or zero if none is current."""
    return int(handle_return(driver.cuCtxGetCurrent()))


def _assert_event_record_rejected_from_ambient_context(event):
    """Assert that recording ``event`` into a stream from the ambient context fails.

    An event and the stream it records must belong to the same context;
    otherwise cuEventRecord fails with CUDA_ERROR_INVALID_HANDLE. cuStreamCreate
    creates the probe stream in whatever context is currently ambient, so this
    is a live check that ``event`` was not actually recorded from there.
    """
    err, ambient_stream = driver.cuStreamCreate(0)
    handle_return(err)
    try:
        (record_status,) = driver.cuEventRecord(event.handle, ambient_stream)
        assert record_status == driver.CUresult.CUDA_ERROR_INVALID_HANDLE, (
            "Recording an event into a stream from a different context should fail with "
            f"CUDA_ERROR_INVALID_HANDLE, got {record_status!r}"
        )
    finally:
        handle_return(driver.cuStreamDestroy(ambient_stream))


def assert_device_operations_use_bound_context(device):
    """Check that Device operations use its bound context and preserve the ambient context."""
    bound_context = device.context
    ambient_context_handle = current_context_handle()
    assert int(bound_context.handle) != ambient_context_handle, (
        "Precondition failed: the device's bound context must not be the current (ambient) context."
    )
    stream = event = builder = None

    try:
        stream = device.create_stream()
        assert stream.context == bound_context
        assert current_context_handle() == ambient_context_handle
        # Live check: query the driver directly, rather than comparing cached
        # metadata, to confirm the stream was actually created in the bound
        # context rather than whatever was ambient.
        driver_stream_ctx = handle_return(driver.cuStreamGetCtx(stream.handle))
        assert int(driver_stream_ctx) == int(bound_context.handle), (
            "cuStreamGetCtx reports a context other than the one the stream was created in."
        )

        event = device.create_event()
        assert event.context == bound_context
        assert current_context_handle() == ambient_context_handle
        # Only exercised when the ambient context belongs to a different
        # physical device: cuEventRecord's cross-context rejection is
        # guaranteed distinct there. Two contexts on the *same* device (e.g.
        # a green context vs. the primary context) may resolve to the same
        # underlying device context for this check, so skip rather than
        # assert unverified driver behavior.
        if ambient_context_handle and handle_return(driver.cuCtxGetDevice()) != device.device_id:
            _assert_event_record_rejected_from_ambient_context(event)

        builder = device.create_graph_builder()
        assert builder.stream.context == bound_context
        assert current_context_handle() == ambient_context_handle

        device.sync()
        assert current_context_handle() == ambient_context_handle

        builder.close()
        builder = None
        assert current_context_handle() == ambient_context_handle

        event.close()
        event = None
        assert current_context_handle() == ambient_context_handle

        stream.close()
        stream = None
        assert current_context_handle() == ambient_context_handle
    finally:
        if builder is not None:
            builder.close()
        if event is not None:
            event.close()
        if stream is not None:
            stream.close()


@contextmanager
def no_current_context():
    """Temporarily remove the calling thread's sole current CUDA context."""
    if current_context_handle() == 0:
        raise RuntimeError("no_current_context requires a current CUDA context")

    previous = handle_return(driver.cuCtxPopCurrent())
    try:
        if current_context_handle() != 0:
            raise RuntimeError("no_current_context requires exactly one stacked CUDA context")
        yield
    finally:
        handle_return(driver.cuCtxPushCurrent(previous))


@contextmanager
def use_context(device, context):
    """Temporarily make a context current and restore the previous context."""
    if current_context_handle() == 0:
        raise RuntimeError("use_context requires a current CUDA context to restore")

    previous = device.set_current(context)
    if previous is None:
        raise RuntimeError("Device.set_current() did not return the previous CUDA context")
    try:
        yield
    finally:
        device.set_current(previous)
