# Copyright 2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

import time

import pytest

from cuda.core.experimental import Device, EventOptions
from cuda.core.experimental._utils import CUDAError


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
        assert delay_seconds * 1000 <= elapsed_time_ms < delay_seconds * 1000 + 2  # tolerance 2 ms
    else:
        with pytest.raises(CUDAError) as e:
            elapsed_time_ms = e2 - e1
            assert "CUDA_ERROR_INVALID_HANDLE" in str(e)


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
