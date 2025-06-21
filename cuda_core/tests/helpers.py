# Copyright 2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

CUDA_PATH = os.environ.get("CUDA_PATH")
CUDA_INCLUDE_PATH = None
CCCL_INCLUDE_PATHS = None
if CUDA_PATH is not None:
    path = os.path.join(CUDA_PATH, "include")
    if os.path.isdir(path):
        CUDA_INCLUDE_PATH = path
        path = os.path.join(path, "cccl")
        if os.path.isdir(path):
            CCCL_INCLUDE_PATHS = (path, CUDA_INCLUDE_PATH)
