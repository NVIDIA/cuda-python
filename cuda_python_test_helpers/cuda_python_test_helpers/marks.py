# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable pytest marks and skip helpers for CUDA Python test suites."""

import inspect
import os

import pytest

from cuda.pathfinder import get_cuda_path_or_home


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


def _cuda_headers_available() -> bool:
    """Return True if CUDA headers are available, False if no CUDA path is set.

    Raises AssertionError if a CUDA path is set but has no include/ subdirectory.
    """
    cuda_path = get_cuda_path_or_home()
    if cuda_path is None:
        return False
    assert os.path.isdir(os.path.join(cuda_path, "include")), (
        f"CUDA path {cuda_path} does not contain an 'include' subdirectory"
    )
    return True


skipif_need_cuda_headers = pytest.mark.skipif(
    not _cuda_headers_available(),
    reason="need CUDA header",
)
