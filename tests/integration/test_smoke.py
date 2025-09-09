# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib


def test_import_cuda_top_level():
    m = importlib.import_module("cuda")
    assert hasattr(m, "__file__")


def test_import_subpackages_present_if_installed():
    for name in [
        "cuda.core",
        "cuda.bindings",
        "cuda.pathfinder",
    ]:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError:
            pass
