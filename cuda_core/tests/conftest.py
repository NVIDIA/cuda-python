# Copyright 2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

from cuda.core.experimental._device import Device
from cuda.core.experimental import _device
from cuda import cuda
from cuda.core.experimental._utils import handle_return
import pytest

@pytest.fixture(scope="module", autouse=True)
def ensure_no_context():
    device = Device()
    device.set_current()
    with _device._tls_lock:
        if hasattr(_device._tls, 'devices'):
            del _device._tls.devices

@pytest.fixture(scope="function")
def init_cuda():
    device = Device()
    device.set_current()
    yield
    handle_return(cuda.cuCtxPopCurrent())
    with _device._tls_lock:
        del _device._tls.devices