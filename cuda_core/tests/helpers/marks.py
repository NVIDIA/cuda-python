# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Reusable pytest marks for cuda_core tests."""

import inspect

import pytest


def requires_module(module, *args, **kwargs):
    """Skip the test if a module is missing or older than required.

    Thin wrapper around :func:`pytest.importorskip`.  The first argument
    may be a module object or a string; all remaining positional and
    keyword arguments (``minversion``, ``reason``, ``exc_type``) are
    forwarded.

    Prefer this over ``pytest.importorskip`` when:

    - You need finer granularity than module scope or a test body; this
      mark can decorate classes, individual tests, or ``pytest.param`` entries.
    - You want to skip before fixtures run, avoiding setup costs.
    - The module is already imported and you want to pass it directly.

    Usage::

        @requires_module("numpy", "2.1")
        def test_foo(): ...


        @requires_module(np, minversion="2.1")
        def test_bar(): ...
    """
    if inspect.ismodule(module):
        module = module.__name__
    elif not isinstance(module, str):
        raise TypeError(f"expected module or string, got {type(module).__name__}")

    try:
        pytest.importorskip(module, *args, **kwargs)
    except pytest.skip.Skipped as exc:
        return pytest.mark.skipif(True, reason=str(exc))
    else:
        return pytest.mark.skipif(False, reason="")
