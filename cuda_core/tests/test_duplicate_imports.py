# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests cuda.core.__init__.py does not import duplicate modules."""

import sys
from importlib.metadata import distribution

import pytest

from cuda import bindings

# Issue only appears when cuda/core/cu<cuda_major> directory is present in
# site-packages; skip when 'cuda.core.cu<cuda_major>' is not found
cuda_major = bindings.__version__.split(".")[0]
pip_submodule = f"cuda.core.{cuda_major}"
dist = distribution("cuda-core")

if not dist.locate_file(f"cuda/core/cu{cuda_major}"):
    pytest.skip(f"{pip_submodule} not present", allow_module_level=True)


def test_typing_module_imports():
    """
    Importing cuda.core.system should not also import cuda.core.cuXX.system
    """
    import cuda.core
    import cuda.core.system  # NOQA

    assert "cuda.core.system" in sys.modules
    assert f"cuda.core.cu{cuda_major}" not in sys.modules
