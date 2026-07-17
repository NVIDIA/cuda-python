# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests cuda.core.__init__.py does not import duplicate modules."""

import sys

from cuda import bindings

cuda_major = bindings.__version__.split(".")[0]


def test_typing_module_imports():
    """
    Importing cuda.core.system should not also import cuda.core.cuXX.system
    """

    assert "cuda.core.system" in sys.modules
    assert f"cuda.core.cu{cuda_major}.system" not in sys.modules
