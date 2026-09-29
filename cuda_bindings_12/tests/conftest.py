# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import pathlib
import sys

# BEGIN SYNCED PYTEST PLUGIN BOOTSTRAP
# Keep every block with this marker byte-for-byte identical.
# Before editing or reviewing, find all copies; see root AGENTS.md.
# Keep the shared pytest plugin available in wheel-test jobs, which run from
# the monorepo checkout without installing cuda-python-test-helpers.
try:
    import cuda_python_test_helpers._pytest_plugin  # noqa: F401
except ImportError as e:
    # Don't call .resolve(): resolving symlinks can make parents[2] point
    # somewhere other than the monorepo root if a sub-directory is symlinked.
    _test_helpers_root = pathlib.Path(__file__).parents[2] / "cuda_python_test_helpers"
    if not _test_helpers_root.is_dir():
        raise RuntimeError(f"cuda-python-test-helpers not installed and not found at {_test_helpers_root}") from e
    for _k in list(sys.modules):
        if _k == "cuda_python_test_helpers" or _k.startswith("cuda_python_test_helpers."):
            del sys.modules[_k]
    sys.path.insert(0, str(_test_helpers_root))
    importlib.invalidate_caches()

pytest_plugins = ["cuda_python_test_helpers._pytest_plugin"]
# END SYNCED PYTEST PLUGIN BOOTSTRAP
