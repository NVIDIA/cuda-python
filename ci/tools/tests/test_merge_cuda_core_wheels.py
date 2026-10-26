# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from zipfile import ZipFile

import pytest
from wheel.wheelfile import WheelFile

from ci.tools.merge_cuda_core_wheels import merge_wheels


@pytest.mark.agent_authored(model="gpt-6-astra")
def test_merged_wheel_preserves_floor_helper_and_all_abi_variants(tmp_path):
    wheels = []
    floor_helper = "SUPPORTED_BINDINGS = {12: '12.9.8', 13: '13.4.1'}\n"
    for major in (12, 13):
        path = tmp_path / f"cuda_core-1.2.3-py3-none-any.cu{major}.whl"
        with WheelFile(path, "w") as wheel:
            wheel.writestr("cuda/core/__init__.py", "from . import _bindings_floor\n")
            wheel.writestr("cuda/core/_version.py", "__version__ = '1.2.3'\n")
            wheel.writestr("cuda/core/_bindings_floor.py", floor_helper)
            wheel.writestr("cuda/core/_abi.py", f"CUDA_MAJOR = {major}\n")
            wheel.writestr(
                "cuda_core-1.2.3.dist-info/METADATA", "Metadata-Version: 2.1\nName: cuda-core\nVersion: 1.2.3\n"
            )
            wheel.writestr(
                "cuda_core-1.2.3.dist-info/WHEEL",
                "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            )
        wheels.append(path)

    merged = merge_wheels(wheels, tmp_path / "merged", show_wheel_contents=False)

    with ZipFile(merged) as wheel:
        assert wheel.read("cuda/core/_bindings_floor.py").decode() == floor_helper
        for major in (12, 13):
            assert wheel.read(f"cuda/core/cu{major}/_abi.py").decode() == f"CUDA_MAJOR = {major}\n"
        assert "cuda/core/_abi.py" not in wheel.namelist()
