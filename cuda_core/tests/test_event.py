# Copyright 2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

import os
import time

import pytest

import cuda.core.experimental
from cuda.core.experimental import Device, EventOptions


def test_event_init_disabled():
    with pytest.raises(RuntimeError, match=r"^Event objects cannot be instantiated directly\."):
        cuda.core.experimental._event.Event()  # Ensure back door is locked.


@pytest.mark.parametrize("enable_timing", [True, False, None])
def test_timing(init_cuda, enable_timing):
    options = EventOptions(enable_timing=enable_timing)
    stream = Device().create_stream()
    delay_seconds = 0.5
    e1 = stream.record(options=options)
    time.sleep(delay_seconds)
    e2 = stream.record(options=options)
    e2.sync()
    for e in (e1, e2):
        assert e.is_timing_disabled == (True if enable_timing is None else not enable_timing)
    if enable_timing:
        elapsed_time_ms = e2 - e1
        assert isinstance(elapsed_time_ms, float)
        # Using a generous tolerance, to avoid flaky tests:
        # We only want to exercise the __sub__ method, this test is not meant
        # to stress-test the CUDA driver or time.sleep().
        delay_ms = delay_seconds * 1000
        if os.name == "nt":  # noqa: SIM108
            # For Python <=3.10, the Windows timer resolution is typically limited to 15.6 ms by default.
            generous_tolerance = 100
        else:
            # Most modern Linux kernels have a default timer resolution of 1 ms.
            generous_tolerance = 20
        assert delay_ms - generous_tolerance <= elapsed_time_ms < delay_ms + generous_tolerance
    else:
        with pytest.raises(RuntimeError) as e:
            elapsed_time_ms = e2 - e1
            msg = str(e)
            assert "disabled by default" in msg
            assert "CUDA_ERROR_INVALID_HANDLE" in msg


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
