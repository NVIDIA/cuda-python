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
