# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.bindings._example_helpers.common import nvrtc_supports_cubin


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize(
    ("nvrtc_version", "expected"),
    [
        ((11, 0), False),  # predates nvrtcGetCUBIN
        ((11, 1), True),  # the release that added it
        ((11, 8), True),
        ((12, 0), True),  # x.0 releases: minor alone would say False
        ((12, 9), True),
        ((13, 0), True),
        ((13, 4), True),
    ],
)
def test_nvrtc_supports_cubin(nvrtc_version, expected):
    assert nvrtc_supports_cubin(nvrtc_version) is expected


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize("nvrtc_version", [(12, 0), (13, 0)])
def test_minor_only_check_would_be_wrong_for_x_0_releases(nvrtc_version):
    """Pins the exact regression: `nvrtc_minor >= 1` disagrees on every x.0."""
    _, nvrtc_minor = nvrtc_version
    assert (nvrtc_minor >= 1) is False
    assert nvrtc_supports_cubin(nvrtc_version) is True
