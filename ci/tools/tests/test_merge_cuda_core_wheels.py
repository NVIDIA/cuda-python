# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from ci.tools.merge_cuda_core_wheels import cuda_variant_from_wheel


@pytest.mark.agent_authored(model="gpt-5.6-sol")
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("cuda_core-1.2.3-cp312-cp312-manylinux_x86_64.cu12.whl", "cu12"),
        ("cuda_core-2.0.0-cp315-cp315-win_amd64.cu14.whl", "cu14"),
    ],
)
def test_cuda_variant_from_wheel(name, expected):
    assert cuda_variant_from_wheel(Path(name)) == expected


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_cuda_variant_from_wheel_rejects_missing_suffix():
    with pytest.raises(ValueError, match=r"does not contain a \.cuN suffix"):
        cuda_variant_from_wheel(Path("cuda_core-1.2.3-py3-none-any.whl"))
