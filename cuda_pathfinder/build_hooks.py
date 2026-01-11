# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom build hooks for cuda-pathfinder to validate version detection.

This module wraps setuptools.build_meta to add version validation that detects
when setuptools-scm falls back to default versions (0.0.x or 0.1.dev*) due to
shallow clones or missing git tags.
"""

import subprocess
import sys
from pathlib import Path

from setuptools import build_meta as _build_meta


def _validate_version():
    """Validate that setuptools-scm did not fall back to default version.

    This checks if cuda-pathfinder version is a fallback (0.0.x or 0.1.dev*) which
    indicates setuptools-scm failed to detect version from git tags.
    """
    repo_root = Path(__file__).resolve().parent.parent
    validation_script = repo_root / "scripts" / "validate_version.py"

    if not validation_script.exists():
        # If validation script doesn't exist, skip validation (shouldn't happen)
        return

    # Run validation script
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(validation_script),
            "cuda-pathfinder",
            "cuda/pathfinder/_version.py",
            "1.3.*",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=10,
    )

    if result.returncode != 0:
        error_msg = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"Version validation failed for cuda-pathfinder:\n{error_msg}\n"
            f"This build will fail to prevent using incorrect fallback version."
        )


# Delegate all PEP 517 hooks to setuptools.build_meta, but add version validation
def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    result = _build_meta.prepare_metadata_for_build_editable(metadata_directory, config_settings)
    _validate_version()
    return result


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    result = _build_meta.prepare_metadata_for_build_wheel(metadata_directory, config_settings)
    _validate_version()
    return result


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _validate_version()
    return _build_meta.build_editable(wheel_directory, config_settings, metadata_directory)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _validate_version()
    return _build_meta.build_wheel(wheel_directory, config_settings, metadata_directory)


# Delegate other hooks unchanged
build_sdist = _build_meta.build_sdist
get_requires_for_build_editable = _build_meta.get_requires_for_build_editable
get_requires_for_build_wheel = _build_meta.get_requires_for_build_wheel
get_requires_for_build_sdist = _build_meta.get_requires_for_build_sdist
