#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Shim that calls scripts/git_describe_command_runner.py if we're in a git repo.

This shim checks if we're in a git repository (by looking for scripts/ directory)
and delegates to the shared git_describe_command_runner.py script.

NOTE:
- cuda_bindings/git_describe_command_shim.py
- cuda_core/git_describe_command_shim.py
- cuda_pathfinder/git_describe_command_shim.py
are EXACT COPIES, PLEASE KEEP ALL FILES IN SYNC.
"""

import subprocess
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python git_describe_command_shim.py <tag_pattern>", file=sys.stderr)  # noqa: T201
    sys.exit(1)

tag_pattern = sys.argv[1]

# Find repo root (go up from package directory)
package_dir = Path(__file__).parent
repo_root = package_dir.parent
scripts_dir = repo_root / "scripts"
git_describe_script = scripts_dir / "git_describe_command_runner.py"

# Check if we're in a git repo (scripts/ should exist)
if not scripts_dir.exists():
    print("ERROR: scripts/ directory not found.", file=sys.stderr)  # noqa: T201
    print("This indicates we're not in a git repository.", file=sys.stderr)  # noqa: T201
    print("git_describe_command_shim should not be called in this context.", file=sys.stderr)  # noqa: T201
    sys.exit(1)

# Check if the shared script exists
if not git_describe_script.exists():
    print(f"ERROR: {git_describe_script} not found.", file=sys.stderr)  # noqa: T201
    print("The git_describe_command_runner script is missing.", file=sys.stderr)  # noqa: T201
    sys.exit(1)

# Call the shared script (from repo root so it can find .git)
result = subprocess.run(  # noqa: S603
    [sys.executable, str(git_describe_script), tag_pattern],
    cwd=repo_root,
    timeout=10,
)

sys.exit(result.returncode)
