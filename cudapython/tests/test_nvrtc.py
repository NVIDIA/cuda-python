# Copyright 2021 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import pytest
from cudapython import nvrtc

def ASSERT_DRV(err):
    if isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError('Nvrtc Error: {}'.format(err))
    else:
        raise RuntimeError('Unknown error type: {}'.format(err))

def test_nvrtcGetSupportedArchs():
    err, supportedArchs = nvrtc.nvrtcGetSupportedArchs()
    ASSERT_DRV(err)
    assert len(supportedArchs) != 0
