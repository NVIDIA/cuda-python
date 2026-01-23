# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from cuda.pathfinder._utils.platform_aware import IS_WINDOWS


def _abs_norm(path: str | None) -> str | None:
    """Return normalized absolute path, or None if path is None."""
    if path:
        return os.path.normpath(os.path.abspath(path))
    return None


def _is_executable(filepath: str) -> bool:
    """Check if a file exists and is executable."""
    if not os.path.isfile(filepath):
        return False
    if IS_WINDOWS:
        # On Windows, any file can be executed; check extension
        return filepath.lower().endswith((".exe", ".bat", ".cmd"))
    else:
        # On Unix, check execute permission
        return os.access(filepath, os.X_OK)
