# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest
from cuda.bindings import nvrtc


def ASSERT_DRV(err):
    if isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"Nvrtc Error: {err}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


def nvrtcVersionLessThan(major, minor):
    err, major_version, minor_version = nvrtc.nvrtcVersion()
    ASSERT_DRV(err)
    return major_version < major or (major == major_version and minor_version < minor)


@pytest.mark.skipif(nvrtcVersionLessThan(11, 3), reason="When nvrtcGetSupportedArchs was introduced")
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
