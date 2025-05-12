# Copyright 2024 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import sys

try:
    from cuda.bindings import driver
except ImportError:
    from cuda import cuda as driver
import pytest

from cuda.core.experimental import Device, _device
from cuda.core.experimental._utils.cuda_utils import handle_return


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
    if hasattr(_device._tls, "devices"):
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


skipif_testing_with_compute_sanitizer = pytest.mark.skipif(
    os.environ.get("CUDA_PYTHON_TESTING_WITH_COMPUTE_SANITIZER", "0") == "1",
    reason="The compute-sanitizer is running, and this test causes an API error.",
)
