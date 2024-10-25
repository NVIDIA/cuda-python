# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

from cuda.core.experimental._context import Context
from cuda.core.experimental._device import Device
import pytest

@pytest.fixture(scope='module')
def init_cuda():
    Device().set_current()

def test_context_initialization():
    try:
        context = Context()
    except NotImplementedError as e:
        assert True
    else:
        assert False, "Expected NotImplementedError was not raised"