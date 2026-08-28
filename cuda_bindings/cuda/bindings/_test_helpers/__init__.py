# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated shim package.

These helpers moved to ``cuda_python_test_helpers`` in #2384 (2026-08-04).
This package is kept for one cuda-core release cycle so that already-released
cuda-core test trees (<=1.1.x), which still import from here, keep working
against newer cuda-bindings wheels. See #2725.

.. deprecated:: cuda-core-1.2
    Remove once cuda-core >= 1.2 is the oldest supported release.
"""

import importlib
import pathlib
import sys


def _ensure_cuda_python_test_helpers_importable() -> None:
    """Make ``cuda_python_test_helpers`` importable if it isn't already.

    Keep in sync with the equivalent fallback in cuda_core/tests/conftest.py
    and cuda_bindings/tests/conftest.py.

    Unlike those conftest.py copies, this module's own ``__file__`` is not a
    reliable anchor: this shim is imported from an installed cuda-bindings
    wheel (e.g. site-packages), not necessarily from a monorepo checkout. So
    in addition to walking up from ``__file__`` (dev/editable checkouts), we
    also walk up from the current working directory. This matters for the
    released-cuda-core compat job: pytest runs from a checkout of the
    released cuda-core tag (e.g. ``.../cuda-core-released/cuda_core``), which
    has its own sibling ``cuda_python_test_helpers`` one level up, separate
    from any copy checked out for cuda-bindings itself.
    """
    try:
        import cuda_python_test_helpers

        return
    except ImportError:
        pass

    # Don't call .resolve(): resolving symlinks can make a parent point
    # somewhere other than the monorepo root if a sub-directory is symlinked.
    candidates = [*pathlib.Path(__file__).parents, pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
    for candidate in candidates:
        test_helpers_root = candidate / "cuda_python_test_helpers"
        if (test_helpers_root / "cuda_python_test_helpers" / "__init__.py").is_file():
            sys.path.insert(0, str(test_helpers_root))
            importlib.invalidate_caches()
            return

    raise ModuleNotFoundError(
        "cuda_python_test_helpers is required by cuda.bindings._test_helpers but is not "
        "installed, and no source checkout was found near this file or the current "
        "working directory."
    )


_ensure_cuda_python_test_helpers_importable()
