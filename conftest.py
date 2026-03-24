# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os

import pytest

from cuda.pathfinder import get_cuda_path_or_home


# Please keep in sync with the copy in cuda_core/tests/conftest.py.
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
