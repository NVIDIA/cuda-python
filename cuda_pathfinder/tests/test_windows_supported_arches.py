# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from cuda.pathfinder import DynamicLibUnknownError, windows_supported_arches


@pytest.mark.parametrize(
    ("libname", "expected"),
    [
        ("cudart", ("x64", "arm64")),
        ("cudla", ("arm64",)),
        ("cufile", ()),
        ("cuda", ("x64", "arm64")),
        ("cusparseLt", ("x64", "arm64")),
    ],
)
@pytest.mark.agent_authored(model="gpt-5")
def test_windows_supported_arches(libname, expected):
    assert windows_supported_arches(libname) == expected


@pytest.mark.agent_authored(model="gpt-5")
def test_windows_supported_arches_rejects_unknown_libname():
    with pytest.raises(DynamicLibUnknownError, match=r"Unknown library name: 'not_a_real_lib'"):
        windows_supported_arches("not_a_real_lib")
