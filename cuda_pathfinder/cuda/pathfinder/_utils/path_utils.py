# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from cuda.pathfinder._utils.platform_aware import IS_WINDOWS


def _abs_norm(path: str | None) -> str | None:
    """Return normalized absolute path, or None if path is None.

    Converts relative paths to absolute and normalizes path separators
    for the current platform.

    Args:
        path: Path to normalize, or None.

    Returns:
        Normalized absolute path, or None if input is None.
    """
    if path:
        result: str = os.path.normpath(os.path.abspath(path))
        return result
    return None


def _is_executable(filepath: str) -> bool:
    """Check if a file exists and is executable.

    On Windows, checks if the file exists and has an executable extension
    (.exe, .bat, .cmd). On Unix-like systems, checks if the file exists
    and has the execute permission bit set (os.X_OK).

    Args:
        filepath: Path to the file to check.

    Returns:
        True if the file is executable, False otherwise.
    """
    if not os.path.isfile(filepath):
        return False
    if IS_WINDOWS:
        # On Windows, executables must have specific extensions (.exe, .bat, .cmd)
        return filepath.lower().endswith((".exe", ".bat", ".cmd"))
    else:
        # On Unix, check execute permission
        return os.access(filepath, os.X_OK)
