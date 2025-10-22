# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import sys

CUDA_PATH = os.environ.get("CUDA_PATH")
CUDA_INCLUDE_PATH = None
CCCL_INCLUDE_PATHS = None
if CUDA_PATH is not None:
    path = os.path.join(CUDA_PATH, "include")
    if os.path.isdir(path):
        CUDA_INCLUDE_PATH = path
        CCCL_INCLUDE_PATHS = (path,)
        path = os.path.join(path, "cccl")
        if os.path.isdir(path):
            CCCL_INCLUDE_PATHS = (path,) + CCCL_INCLUDE_PATHS


try:
    import cuda_python_test_helpers
except ImportError:
    # Import shared platform helpers for tests across repos
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "cuda_python_test_helpers"))
    import cuda_python_test_helpers

# If we imported the package instead of the module, get the actual module
if hasattr(cuda_python_test_helpers, '__path__'):
    # We imported the package, need to get the actual module
    import cuda_python_test_helpers.cuda_python_test_helpers as cuda_python_test_helpers


IS_WSL = cuda_python_test_helpers.IS_WSL
supports_ipc_mempool = cuda_python_test_helpers.supports_ipc_mempool


del cuda_python_test_helpers