#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that git_describe_command_shim.py files are identical across packages.

This script verifies that the three shim files:
- cuda_bindings/git_describe_command_shim.py
- cuda_core/git_describe_command_shim.py
- cuda_pathfinder/git_describe_command_shim.py

are exact copies of each other.
"""

import sys
from pathlib import Path

# Find repo root (assume script is in toolshed/)
repo_root = Path(__file__).parent.parent

shim_files = [
    repo_root / "cuda_bindings" / "git_describe_command_shim.py",
    repo_root / "cuda_core" / "git_describe_command_shim.py",
    repo_root / "cuda_pathfinder" / "git_describe_command_shim.py",
]

# Check all files exist
missing_files = [f for f in shim_files if not f.exists()]
if missing_files:
    print("ERROR: Missing shim files:", file=sys.stderr)
    for f in missing_files:
        print(f"  {f}", file=sys.stderr)
    sys.exit(1)

# Read all files
file_contents = {}
for shim_file in shim_files:
    file_contents[shim_file] = shim_file.read_text()

# Compare all pairs
errors = []
for i, file1 in enumerate(shim_files):
    for file2 in shim_files[i + 1 :]:
        if file_contents[file1] != file_contents[file2]:
            errors.append((file1, file2))

if errors:
    print("ERROR: git_describe_command_shim.py files are not identical:", file=sys.stderr)
    for file1, file2 in errors:
        print(f"  {file1.relative_to(repo_root)} != {file2.relative_to(repo_root)}", file=sys.stderr)
    print("", file=sys.stderr)
    print("These files must be kept in sync. Please copy one to the others.", file=sys.stderr)
    sys.exit(1)

print("✓ All git_describe_command_shim.py files are identical")
sys.exit(0)
