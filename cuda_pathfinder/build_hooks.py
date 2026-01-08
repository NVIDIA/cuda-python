# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Custom build hooks for cuda-pathfinder.

This module validates git tags are available before setuptools-scm runs,
ensuring proper version detection during pip install. All PEP 517 build
hooks are delegated to setuptools.build_meta.
"""

import os
import subprocess


# Validate tags before importing setuptools.build_meta (which will use setuptools-scm)
def _validate_git_tags_available():
    """Verify that git tags are available for setuptools-scm version detection."""
    # Check if git is available
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True, timeout=5)  # noqa: S607
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        raise RuntimeError(
            "Git is not available in PATH. setuptools-scm requires git to determine version from tags.\n"
            "Please ensure git is installed and available in your PATH."
        ) from None

    # Find git repository root (setuptools_scm root='..')
    repo_root = os.path.dirname(os.path.dirname(__file__))

    # Check if git describe works (this is what setuptools-scm uses)
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--long", "--match", "cuda-pathfinder-v*[0-9]*"],  # noqa: S607
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git describe failed! This means setuptools-scm will fall back to version '0.1.x'.\n"
                f"\n"
                f"Error: {result.stderr.strip()}\n"
                f"\n"
                f"This usually means:\n"
                f"  1. Git tags are not fetched (run: git fetch --tags)\n"
                f"  2. Running from wrong directory (setuptools_scm root='..')\n"
                f"  3. No matching tags found\n"
                f"\n"
                f"To fix:\n"
                f"  git fetch --tags\n"
                f"\n"
                f"To debug, run: git describe --tags --long --match 'cuda-pathfinder-v*[0-9]*'"
            ) from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "git describe command timed out. This may indicate git repository issues.\n"
            "Please check your git repository state."
        ) from None


# Validate tags before any build operations
_validate_git_tags_available()

# Import and re-export all PEP 517 hooks from setuptools.build_meta
from setuptools.build_meta import *  # noqa: F403, E402
