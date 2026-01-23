# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest

# Import centralized CUDA environment variable handling
try:
    from cuda.pathfinder._utils.env_vars import get_cuda_home_or_path
except ImportError as e:
    raise ImportError(
        "Failed to import cuda.pathfinder. Please ensure cuda-pathfinder is installed: pip install cuda-pathfinder"
    ) from e


def pytest_collection_modifyitems(config, items):
    cuda_home = get_cuda_home_or_path()
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

            # Gate core cython tests on CUDA_HOME
            if "core" in item.keywords and not cuda_home:
                item.add_marker(pytest.mark.skip(reason="CUDA_HOME not set; skipping core cython tests"))
