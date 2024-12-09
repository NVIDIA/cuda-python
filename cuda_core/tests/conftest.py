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
    from cuda.bindings import driver, nvrtc
except ImportError:
    from cuda import cuda as driver
    from cuda import nvrtc
import pytest

from cuda.core.experimental import Device, _device
from cuda.core.experimental._utils import handle_return


@pytest.fixture(scope="session", autouse=True)
def always_init_cuda():
    handle_return(driver.cuInit(0))


@pytest.fixture(scope="function")
def init_cuda():
    # TODO: rename this to e.g. init_context
    device = Device()
    device.set_current()
    yield
    _device_unset_current()


def _device_unset_current():
    ctx = handle_return(driver.cuCtxGetCurrent())
    if int(ctx) == 0:
        # no active context, do nothing
        return
    handle_return(driver.cuCtxPopCurrent())
    with _device._tls_lock:
        del _device._tls.devices


@pytest.fixture(scope="function")
def deinit_cuda():
    # TODO: rename this to e.g. deinit_context
    yield
    _device_unset_current()


# samples relying on cffi could fail as the modules cannot be imported
sys.path.append(os.getcwd())


@pytest.fixture(scope="session", autouse=True)
def clean_up_cffi_files():
    yield
    files = glob.glob(os.path.join(os.getcwd(), "_cpu_obj*"))
    for f in files:
        try:  # noqa: SIM105
            os.remove(f)
        except FileNotFoundError:
            pass  # noqa: SIM105


def can_load_generated_ptx():
    _, driver_ver = driver.cuDriverGetVersion()
    _, nvrtc_major, nvrtc_minor = nvrtc.nvrtcVersion()
    return not nvrtc_major * 1000 + nvrtc_minor * 10 > driver_ver
