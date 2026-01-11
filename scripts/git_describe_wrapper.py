#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Git describe wrapper for setuptools-scm that fails loudly if no matching tags found.

This script is used as a replacement for git_describe_command in pyproject.toml.
It provides better error messages and ensures setuptools-scm doesn't silently
fall back to 0.1.x when tags are missing.

Usage:
    python git_describe_wrapper.py <tag_pattern>

Example:
    python git_describe_wrapper.py "v*[0-9]*"
"""

import subprocess
import sys

if len(sys.argv) < 2:
    print("Usage: python git_describe_wrapper.py <tag_pattern>", file=sys.stderr)  # noqa: T201
    sys.exit(1)

tag_pattern = sys.argv[1]

# Check if git is available
try:
    subprocess.run(["git", "--version"], capture_output=True, check=True, timeout=5)  # noqa: S607
except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
    print("ERROR: Git is not available in PATH.", file=sys.stderr)  # noqa: T201
    print("setuptools-scm requires git to determine version from tags.", file=sys.stderr)  # noqa: T201
    sys.exit(1)

# Run git describe (setuptools-scm expects --dirty --tags --long)
result = subprocess.run(  # noqa: S603
    ["git", "describe", "--dirty", "--tags", "--long", "--match", tag_pattern],  # noqa: S607
    capture_output=True,
    text=True,
    timeout=5,
)

if result.returncode != 0:
    print(f"ERROR: git describe failed with pattern '{tag_pattern}'", file=sys.stderr)  # noqa: T201
    print(f"Error: {result.stderr.strip()}", file=sys.stderr)  # noqa: T201
    print("", file=sys.stderr)  # noqa: T201
    print("This means setuptools-scm will fall back to version '0.1.x'.", file=sys.stderr)  # noqa: T201
    print("", file=sys.stderr)  # noqa: T201
    print("This usually means:", file=sys.stderr)  # noqa: T201
    print("  1. Git tags are not fetched (run: git fetch --tags)", file=sys.stderr)  # noqa: T201
    print("  2. Running from wrong directory", file=sys.stderr)  # noqa: T201
    print("  3. No matching tags found", file=sys.stderr)  # noqa: T201
    print("", file=sys.stderr)  # noqa: T201
    print("To fix:", file=sys.stderr)  # noqa: T201
    print("  git fetch --tags", file=sys.stderr)  # noqa: T201
    sys.exit(1)

print(result.stdout.strip())  # noqa: T201
