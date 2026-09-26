# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


@pytest.mark.agent_authored(model="claude-sonnet-5")
@pytest.mark.skipif(nvrtcVersionLessThan(13, 3), reason="When nvrtcGetBundledHeadersInfo was introduced")
def test_nvrtcGetBundledHeadersInfo():
    info = nvrtc.nvrtcBundledHeadersInfo()
    assert isinstance(info, nvrtc.nvrtcBundledHeadersInfo)

    err, info, errorLog = nvrtc.nvrtcGetBundledHeadersInfo()
    ASSERT_DRV(err)
    assert isinstance(info, nvrtc.nvrtcBundledHeadersInfo)
    assert info.available in (0, 1)
    assert info.compressedSize >= 0
    assert info.uncompressedSize >= 0
    assert info.cudaVersionMajor >= 0
    assert info.cudaVersionMinor >= 0
    assert info.numFiles >= 0
    assert errorLog is None
