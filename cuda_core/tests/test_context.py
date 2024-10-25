# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

from cuda.core.experimental._context import Context
from cuda.core.experimental._device import Device
from cuda.core.experimental._utils import handle_return
from cuda import cuda
import pytest


def test_context_initialization():
    context = Context()
    assert context is not None

def test_context_from_ctx(reestablish_valid_context):
    device = Device()
    dev_id = 0

    # push the primary context and set it as the current context for the device
    ctx = handle_return(cuda.cuDevicePrimaryCtxRetain(dev_id))
    handle_return(cuda.cuCtxPushCurrent(ctx))
    device.set_current(Context._from_ctx(ctx, 0))

    # pop the context
    handle_return(cuda.cuCtxPopCurrent())

    # the device's context *has* been initialized, but if this is the method used to guard active context dependant calls, it should return false
    assert device._check_context_initialized() == False