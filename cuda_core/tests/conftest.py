# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

import pytest
from cuda.core.experimental._device import Device

@pytest.fixture(scope="module")
def init_cuda():
    device = Device()
    device.set_current()
    