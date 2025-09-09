# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional


def get_cuda_home_or_path() -> Optional[str]:
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home is None:
        cuda_home = os.environ.get("CUDA_PATH")
    return cuda_home
