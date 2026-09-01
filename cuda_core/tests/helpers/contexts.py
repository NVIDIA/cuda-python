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


def assert_device_operations_use_bound_context(device):
    """Check that Device operations use its bound context and preserve the ambient context."""
    bound_context = device.context
    ambient_context_handle = current_context_handle()
    stream = event = builder = None

    try:
        stream = device.create_stream()
        assert stream.context == bound_context
        assert current_context_handle() == ambient_context_handle

        event = device.create_event()
        assert event.context == bound_context
        assert current_context_handle() == ambient_context_handle

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
