# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.bindings._v2 import nvrtc


def nvrtc_version_less_than(major, minor):
    major_version, minor_version = nvrtc.version()
    return major_version < major or (major == major_version and minor_version < minor)


@pytest.mark.skipif(nvrtc_version_less_than(11, 3), reason="When nvrtcGetSupportedArchs was introduced")
def test_get_supported_archs():
    supported_archs = nvrtc.get_supported_archs()
    assert len(supported_archs) != 0


@pytest.mark.skipif(nvrtc_version_less_than(12, 1), reason="Preempt Segmentation Fault (see #499)")
def test_get_lowered_name_failure():
    with pytest.raises(nvrtc.InvalidProgramError):
        nvrtc.get_lowered_name(0, b"I'm an elevated name!")
    with pytest.raises(nvrtc.InvalidProgramError):
        nvrtc.get_lowered_name(0, b"I'm another elevated name!")


@pytest.mark.agent_authored(model="claude-sonnet-5")
@pytest.mark.skipif(nvrtc_version_less_than(13, 3), reason="When nvrtcGetBundledHeadersInfo was introduced")
def test_get_bundled_headers_info():
    info = nvrtc.BundledHeadersInfo()
    assert isinstance(info, nvrtc.BundledHeadersInfo)

    info, error_log = nvrtc.get_bundled_headers_info()
    assert isinstance(info, nvrtc.BundledHeadersInfo)
    assert info.available in (0, 1)
    assert info.compressed_size >= 0
    assert info.uncompressed_size >= 0
    assert info.cuda_version_major >= 0
    assert info.cuda_version_minor >= 0
    assert info.num_files >= 0
    assert error_log is None
