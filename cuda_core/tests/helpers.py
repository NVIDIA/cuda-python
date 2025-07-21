# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

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
