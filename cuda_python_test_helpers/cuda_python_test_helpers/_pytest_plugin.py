# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest plugin registered via the ``pytest11`` entry point.

Automatically tags collected items with package markers and gates cython
tests on CUDA header availability.  Loaded by pytest whenever
``cuda-python-test-helpers`` is installed, and also explicitly via
``pytest_plugins`` in each subpackage conftest so the fallback sys.path
install path is covered too.
"""

import pytest

from cuda_python_test_helpers.marks import _cuda_headers_available


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    have_headers = _cuda_headers_available()
    for item in items:
        nodeid = item.nodeid.replace("\\", "/")

        # Package markers by path
        if nodeid.startswith("cuda_pathfinder/tests/") or "/cuda_pathfinder/tests/" in nodeid:
            item.add_marker(pytest.mark.pathfinder)
        if nodeid.startswith("cuda_bindings/tests/") or "/cuda_bindings/tests/" in nodeid:
            item.add_marker(pytest.mark.bindings)
        if nodeid.startswith("cuda_core/tests/") or "/cuda_core/tests/" in nodeid:
            item.add_marker(pytest.mark.core)

        # Smoke tests
        if nodeid.startswith("tests/integration/") or "/tests/integration/" in nodeid:
            item.add_marker(pytest.mark.smoke)

        # Cython tests (any tests/cython subtree)
        if (
            "/tests/cython/" in nodeid
            or nodeid.endswith("/tests/cython")
            or ("/cython/" in nodeid and "/tests/" in nodeid)
        ):
            item.add_marker(pytest.mark.cython)

            # Gate core cython tests on CUDA_PATH
            if "core" in item.keywords and not have_headers:
                item.add_marker(
                    pytest.mark.skip(
                        reason="Environment variable CUDA_PATH or CUDA_HOME is not set: skipping core cython tests"
                    )
                )
