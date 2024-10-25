# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

from cuda import cuda
from cuda.core.experimental._event import  EventOptions, Event
from cuda.core.experimental._utils import handle_return
from cuda.core.experimental._device import Device
import pytest


def test_is_timing_disabled():
    options = EventOptions(enable_timing=False)
    event = Event._init(options)
    assert event.is_timing_disabled == True

def test_is_sync_busy_waited():
    options = EventOptions(busy_waited_sync=True)
    event = Event._init(options)
    assert event.is_sync_busy_waited == True

def test_is_ipc_supported():
    options = EventOptions(support_ipc=True)
    try:
        event = Event._init(options)
    except NotImplementedError:
        assert True
    else:
        assert False

def test_sync():
    options = EventOptions()
    event = Event._init(options)
    event.sync()
    assert event.is_done == True

def test_is_done():
    options = EventOptions()
    event = Event._init(options)
    assert event.is_done == True

def test_handle():
    options = EventOptions()
    event = Event._init(options)
    assert isinstance(event.handle, int)
