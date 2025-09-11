# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import warnings
from typing import Optional


def _paths_differ(a: str, b: str) -> bool:
    """
    Return True if paths are observably different.

    Strategy:
    1) Compare os.path.normcase(os.path.normpath(...)) for quick, robust textual equality.
       - Handles trailing slashes and case-insensitivity on Windows.
    2) If still different AND both exist, use os.path.samefile to resolve symlinks/junctions.
    3) Otherwise (nonexistent paths or samefile unavailable), treat as different.
    """
    norm_a = os.path.normcase(os.path.normpath(a))
    norm_b = os.path.normcase(os.path.normpath(b))
    if norm_a == norm_b:
        return False

    try:
        if os.path.exists(a) and os.path.exists(b):
            # samefile raises on non-existent paths; only call when both exist.
            return not os.path.samefile(a, b)
    except OSError:
        # Fall through to "different" if samefile isn't applicable/available.
        pass

    # If normalized strings differ and we couldn't prove they're the same entry, treat as different.
    return True


def get_cuda_home_or_path() -> Optional[str]:
    cuda_home = os.environ.get("CUDA_HOME")
    cuda_path = os.environ.get("CUDA_PATH")

    if cuda_home and cuda_path and _paths_differ(cuda_home, cuda_path):
        warnings.warn(
            "Both CUDA_HOME and CUDA_PATH are set but differ:\n"
            f"  CUDA_HOME={cuda_home}\n"
            f"  CUDA_PATH={cuda_path}\n"
            "Using CUDA_HOME (higher priority).",
            UserWarning,
            stacklevel=2,
        )

    if cuda_home is not None:
        return cuda_home
    return cuda_path
