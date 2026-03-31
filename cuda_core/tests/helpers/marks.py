# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Reusable pytest marks for cuda_core tests."""

import importlib
import types

import pytest


def requires(module, *version):
    """Skip the test if a module is missing or older than the given version.

    Usage::

        @requires(np, 2, 1)
        def test_foo(): ...


        @requires("scipy", 1, 12)
        def test_bar(): ...
    """
    if isinstance(module, str):
        name = module
        try:
            module = importlib.import_module(name)
        except ImportError:
            return pytest.mark.skip(reason=f"{name} is not installed")
    elif isinstance(module, types.ModuleType):
        name = module.__name__
    else:
        raise TypeError(f"expected module or string, got {type(module).__name__}")

    n = len(version)
    parts = module.__version__.split(".")[:n]
    installed = tuple(int(p) for p in parts)
    ver_str = ".".join(str(v) for v in version)
    return pytest.mark.skipif(installed < version, reason=f"need {name} {ver_str}+")
