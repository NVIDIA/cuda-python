# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import pathlib
import sys

# Keep the shared pytest plugin available in wheel-test jobs, which run from
# the monorepo checkout without installing cuda-python-test-helpers.
try:
    import cuda_python_test_helpers._pytest_plugin  # noqa: F401
except ImportError as e:
    _test_helpers_root = pathlib.Path(__file__).parents[2] / "cuda_python_test_helpers"
    if not _test_helpers_root.is_dir():
        raise RuntimeError(f"cuda-python-test-helpers not installed and not found at {_test_helpers_root}") from e
    for _key in list(sys.modules):
        if _key == "cuda_python_test_helpers" or _key.startswith("cuda_python_test_helpers."):
            del sys.modules[_key]
    sys.path.insert(0, str(_test_helpers_root))
    importlib.invalidate_caches()

pytest_plugins = ["cuda_python_test_helpers._pytest_plugin"]
