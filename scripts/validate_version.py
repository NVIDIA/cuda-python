#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Validate that setuptools-scm did not fall back to default version.

This script checks if a package version is a fallback version (0.0.x or 0.1.dev*)
which indicates setuptools-scm failed to detect version from git tags.

Usage:
    python scripts/validate_version.py <package_name> <version_file_path> <expected_pattern>

Example:
    python scripts/validate_version.py cuda-pathfinder cuda/pathfinder/_version.py "1.3.*|12.9.*|13.*"
"""

import re
import sys
from pathlib import Path


def validate_version(package_name: str, version_file_path: str, expected_pattern: str) -> None:
    """Validate that version matches expected pattern and is not a fallback.

    Args:
        package_name: Name of the package (for error messages)
        version_file_path: Path to _version.py file (relative to repo root)
        expected_pattern: Regex pattern for expected version format (e.g., "1.3.*|12.9.*|13.*")

    Raises:
        RuntimeError: If version is a fallback or doesn't match expected pattern
    """
    version_file = Path(version_file_path)
    if not version_file.exists():
        # Version file might not exist yet if validation runs during prepare_metadata
        # In that case, skip validation silently (it will be validated later in build hooks)
        # This allows prepare_metadata to complete, and validation will happen in build_editable/build_wheel
        return

    # Read version from _version.py
    with open(version_file, encoding="utf-8") as f:
        content = f.read()

    # Extract __version__
    version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not version_match:
        raise RuntimeError(
            f"Could not find __version__ in {version_file_path}\n"
            f"This may indicate setuptools-scm failed to generate version metadata."
        )

    version = version_match.group(1)

    # Check for fallback versions
    if version.startswith("0.0") or version.startswith("0.1.dev"):
        raise RuntimeError(
            f"ERROR: {package_name} has fallback version '{version}'.\n"
            f"\n"
            f"This indicates setuptools-scm failed to detect version from git tags.\n"
            f"This usually happens when:\n"
            f"  1. Repository is a shallow clone without tags in history\n"
            f"  2. Git tags are not fetched (run: git fetch --tags)\n"
            f"  3. Running from wrong directory\n"
            f"\n"
            f"To fix:\n"
            f"  git fetch --unshallow\n"
            f"  git fetch --tags\n"
            f"\n"
            f"If you cannot fix the git setup, ensure CI performs a full clone or\n"
            f"fetches enough history to include tags."
        )

    # Check if version matches expected pattern
    if not re.match(expected_pattern.replace("*", ".*"), version):
        raise RuntimeError(
            f"ERROR: {package_name} version '{version}' does not match expected pattern '{expected_pattern}'.\n"
            f"\n"
            f"This may indicate:\n"
            f"  1. Wrong branch/commit is being built\n"
            f"  2. Git tags are incorrect\n"
            f"  3. Version detection logic is broken\n"
        )

    # Success - version is valid
    print(f"✅ {package_name} version '{version}' is valid", file=sys.stderr)  # noqa: T201


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <package_name> <version_file_path> <expected_pattern>", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    package_name = sys.argv[1]
    version_file_path = sys.argv[2]
    expected_pattern = sys.argv[3]

    try:
        validate_version(package_name, version_file_path, expected_pattern)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)  # noqa: T201
        sys.exit(1)
