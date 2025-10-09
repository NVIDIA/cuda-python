# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time

import cuda.core.experimental
import helpers
import numpy as np
import pytest
from cuda.core.experimental import (
    Device,
    Event,
    EventOptions,
    LaunchConfig,
    LegacyPinnedMemoryResource,
    Program,
    ProgramOptions,
    launch,
)

from conftest import skipif_need_cuda_headers
from cuda_python_test_helpers import IS_WSL


def test_event_init_disabled():
    with pytest.raises(RuntimeError, match=r"^Event objects cannot be instantiated directly\."):
        cuda.core.experimental._event.Event()  # Ensure back door is locked.


def test_timing_success(init_cuda):
    options = EventOptions(enable_timing=True)
    stream = Device().create_stream()
    delay_seconds = 0.5
    e1 = stream.record(options=options)
    time.sleep(delay_seconds)
    e2 = stream.record(options=options)
    e2.sync()
    elapsed_time_ms = e2 - e1
    assert isinstance(elapsed_time_ms, float)
    # Using a generous tolerance, to avoid flaky tests:
    # We only want to exercise the __sub__ method, this test is not meant
    # to stress-test the CUDA driver or time.sleep().
    delay_ms = delay_seconds * 1000
    if os.name == "nt" or IS_WSL:  # noqa: SIM108
        # For Python <=3.10, the Windows timer resolution is typically limited to 15.6 ms by default.
        generous_tolerance = 100
    else:
        # Most modern Linux kernels have a default timer resolution of 1 ms.
        generous_tolerance = 20
    assert delay_ms - generous_tolerance <= elapsed_time_ms < delay_ms + generous_tolerance


def test_is_sync_busy_waited(init_cuda):
    options = EventOptions(enable_timing=False, busy_waited_sync=True)
    stream = Device().create_stream()
    event = stream.record(options=options)
    assert event.is_sync_busy_waited is True

    options = EventOptions(enable_timing=False)
    stream = Device().create_stream()
    event = stream.record(options=options)
    assert event.is_sync_busy_waited is False


def test_sync(init_cuda):
    options = EventOptions(enable_timing=False)
    stream = Device().create_stream()
    event = stream.record(options=options)
    event.sync()
    assert event.is_done is True


def test_is_done(init_cuda):
    options = EventOptions(enable_timing=False)
    stream = Device().create_stream()
    event = stream.record(options=options)
    # Without a sync, the captured work might not have yet completed
    # Therefore this check should never raise an exception
    assert event.is_done in (True, False)


def test_error_timing_disabled():
    device = Device()
    device.set_current()
    enabled = EventOptions(enable_timing=True)
    disabled = EventOptions(enable_timing=False)
    stream = device.create_stream()

    event1 = stream.record(options=enabled)
    event2 = stream.record(options=disabled)
    assert not event1.is_timing_disabled
    assert event2.is_timing_disabled
    stream.sync()
    with pytest.raises(RuntimeError, match="^Both Events must be created with timing enabled"):
        event2 - event1

    event1 = stream.record(options=disabled)
    event2 = stream.record(options=disabled)
    stream.sync()
    with pytest.raises(RuntimeError, match="^Both Events must be created with timing enabled"):
        event2 - event1


def test_error_timing_recorded():
    device = Device()
    device.set_current()
    enabled = EventOptions(enable_timing=True)
    stream = device.create_stream()

    event1 = stream.record(options=enabled)
    event2 = device.create_event(options=enabled)
    event3 = device.create_event(options=enabled)

    stream.sync()
    with pytest.raises(RuntimeError, match="^Both Events must be recorded"):
        event2 - event1
    with pytest.raises(RuntimeError, match="^Both Events must be recorded"):
        event1 - event2
    with pytest.raises(RuntimeError, match="^Both Events must be recorded"):
        event3 - event2


@skipif_need_cuda_headers  # libcu++
@pytest.mark.skipif(tuple(int(i) for i in np.__version__.split(".")[:2]) < (2, 1), reason="need numpy 2.1.0+")
def test_error_timing_incomplete():
    device = Device()
    device.set_current()

    # This kernel is designed to busy loop until a signal is received
    code = """
#include <cuda/atomic>

extern "C"
__global__ void wait(int* val) {
    cuda::atomic_ref<int, cuda::thread_scope_system> signal{*val};
    while (true) {
        if (signal.load(cuda::memory_order_relaxed)) {
            break;
        }
    }
}
"""

    arch = "".join(f"{i}" for i in device.compute_capability)
    program_options = ProgramOptions(
        std="c++17",
        arch=f"sm_{arch}",
        include_path=helpers.CCCL_INCLUDE_PATHS,
    )
    prog = Program(code, code_type="c++", options=program_options)
    mod = prog.compile(target_type="cubin")
    ker = mod.get_kernel("wait")

    mr = LegacyPinnedMemoryResource()
    b = mr.allocate(4)
    arr = np.from_dlpack(b).view(np.int32)
    arr[0] = 0

    config = LaunchConfig(grid=1, block=1)
    ker_args = (arr.ctypes.data,)

    enabled = EventOptions(enable_timing=True)
    stream = device.create_stream()

    event1 = stream.record(options=enabled)
    launch(stream, config, ker, *ker_args)
    event3 = stream.record(options=enabled)

    # event3 will never complete because the stream is waiting on wait() to complete
    with pytest.raises(RuntimeError, match="^One or both events have not completed."):
        event3 - event1

    arr[0] = 1
    event3.sync()
    event3 - event1  # this should work
    b.close()


def test_event_device(init_cuda):
    device = Device()
    event = device.create_event(options=EventOptions())
    assert event.device is device


def test_event_context(init_cuda):
    event = Device().create_event(options=EventOptions())
    context = event.context
    assert context is not None


def test_event_subclassing():
    class MyEvent(Event):
        pass

    dev = Device()
    dev.set_current()
    event = MyEvent._init(dev.device_id, dev.context)
    assert isinstance(event, MyEvent)
