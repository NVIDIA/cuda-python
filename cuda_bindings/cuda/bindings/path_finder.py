# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# This file is for TEMPORARY BACKWARD COMPATIBILITY only.
# cuda.bindings.pathfinder is slated to be removed after the next cuda-bindings release.

from cuda.bindings._path_finder.temporary_backward_compatibility import (
    load_nvidia_dynamic_library as _load_nvidia_dynamic_library,
)
from cuda.pathfinder import SUPPORTED_NVIDIA_LIBNAMES as _SUPPORTED_LIBNAMES

__all__ = [
    "_load_nvidia_dynamic_library",
    "_SUPPORTED_LIBNAMES",
]
