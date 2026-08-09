# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest plugin registered via the ``pytest11`` entry point.

Automatically tags collected items with package markers and gates cython
tests on CUDA header availability.  Loaded by pytest whenever
``cuda-python-test-helpers`` is installed, and also explicitly via
``pytest_plugins`` in each subpackage conftest so the fallback sys.path
install path is covered too.
"""

import itertools

import pytest

from cuda_python_test_helpers.marks import _cuda_headers_available

_PACKAGE_MARKERS = {
    "cuda_pathfinder": "pathfinder",
    "cuda_bindings": "bindings",
    "cuda_core": "core",
}


def _segments(item) -> tuple[str, ...]:
    """Path segments for ``item``, preferring the real filesystem path.

    ``item.nodeid`` is relative to pytest's *rootdir*, and each subpackage
    ships its own pytest.ini -- so ``pushd ./cuda_core && pytest tests/``,
    which is exactly what ci/tools/run-tests does, makes rootdir ``cuda_core/``
    and every nodeid start at ``tests/``. Matching a package name against the
    nodeid therefore never fires there. ``item.path`` is absolute and does not
    move with rootdir.
    """
    path = getattr(item, "path", None)
    if path is not None:
        return tuple(path.parts)
    return tuple(item.nodeid.replace("\\", "/").split("/"))


def _followed_by(segments: tuple[str, ...], first: str, second: str) -> bool:
    """True if ``first`` appears immediately before ``second``."""
    return any(a == first and b == second for a, b in itertools.pairwise(segments))


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    have_headers = _cuda_headers_available()
    for item in items:
        segments = _segments(item)

        # Package markers by path: "<package>/tests/..."
        for package, marker in _PACKAGE_MARKERS.items():
            if _followed_by(segments, package, "tests"):
                item.add_marker(getattr(pytest.mark, marker))

        # Smoke tests: "tests/integration/..."
        if _followed_by(segments, "tests", "integration"):
            item.add_marker(pytest.mark.smoke)

        # Cython tests: any "tests/cython/..." subtree
        if _followed_by(segments, "tests", "cython"):
            item.add_marker(pytest.mark.cython)

            # Gate core cython tests on CUDA_PATH
            if "core" in item.keywords and not have_headers:
                item.add_marker(
                    pytest.mark.skip(
                        reason="Environment variable CUDA_PATH or CUDA_HOME is not set: skipping core cython tests"
                    )
                )
