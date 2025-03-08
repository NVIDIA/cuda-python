# Copyright 2021-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import pytest
from cuda import nvrtc

def ASSERT_DRV(err):
    if isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError('Nvrtc Error: {}'.format(err))
    else:
        raise RuntimeError('Unknown error type: {}'.format(err))

def nvrtcVersionLessThan(major, minor):
    err, major_version, minor_version = nvrtc.nvrtcVersion()
    ASSERT_DRV(err)
    return major_version < major or (major == major_version and minor_version < minor)

@pytest.mark.skipif(nvrtcVersionLessThan(11, 3), reason='When nvrtcGetSupportedArchs was introduced')
def test_nvrtcGetSupportedArchs():
    err, supportedArchs = nvrtc.nvrtcGetSupportedArchs()
    ASSERT_DRV(err)
    assert len(supportedArchs) != 0


@pytest.mark.skipif(nvrtcVersionLessThan(12, 1), reason="Preempt Segmentation Fault (see #499)")
def test_nvrtcGetLoweredName_failure():
    err, name = nvrtc.nvrtcGetLoweredName(None, b"I'm an elevated name!")
    assert err == nvrtc.nvrtcResult.NVRTC_ERROR_INVALID_PROGRAM
    assert name is None
    err, name = nvrtc.nvrtcGetLoweredName(0, b"I'm another elevated name!")
    assert err == nvrtc.nvrtcResult.NVRTC_ERROR_INVALID_PROGRAM
    assert name is None
