# Copyright 2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

import glob
import os
import sys

try:
    from cuda.bindings import driver
except ImportError:
    from cuda import cuda as driver

import pytest

from cuda.core.experimental import Device, _device
from cuda.core.experimental._utils import handle_return


@pytest.fixture(scope="function")
def init_cuda():
    device = Device()
    device.set_current()
    yield
    _device_unset_current()


def _device_unset_current():
    handle_return(driver.cuCtxPopCurrent())
    with _device._tls_lock:
        del _device._tls.devices


@pytest.fixture(scope="function")
def deinit_cuda():
    yield
    _device_unset_current()


# samples relying on cffi could fail as the modules cannot be imported
sys.path.append(os.getcwd())


@pytest.fixture(scope="session", autouse=True)
def clean_up_cffi_files():
    yield
    files = glob.glob(os.path.join(os.getcwd(), "_cpu_obj*"))
    for f in files:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
