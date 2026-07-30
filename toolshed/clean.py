# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remove generated build artifacts from the cuda-python worktree.

Switching a checkout between the cu12 and cu13 pixi environments leaves behind
compiled Cython extensions, generated ``.pyx``/``.cpp`` sources, and the
``cache_driver``/``cache_runtime``/``cache_nvrtc`` parser caches from the
previous CUDA major. The next build then fails with errors that point at code
which is perfectly fine. Removing the stale artifacts is the fix.

Which files are generated is taken from git rather than from a hand-maintained
glob list: anything git reports as ignored is, by definition, generated. That
also means tracked files are never touched -- including the handful of ``*.cpp``
files checked in despite the blanket ``*.cpp`` ignore rule
(``param_packer.cpp``, ``loader.cpp``, ``*_impl.cpp``).

Deletion is then restricted to paths this repository actually owns, so an
unrecognized ignored directory at the top level (a virtualenv, editor state, a
local scratch directory) is never removed just because it happens to be
ignored. See CLEAN_TOP_LEVEL_DIRS and CLEAN_TOP_LEVEL_ARTIFACTS.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

# Top-level directories owned by this repository. Ignored files underneath them
# are build artifacts and are safe to remove. Keep in sync with the top-level
# layout enforced by toolshed/check_spdx.py.
CLEAN_TOP_LEVEL_DIRS = frozenset(
    {
        ".github",
        "benchmarks",
        "ci",
        "cuda_bindings",
        "cuda_core",
        "cuda_pathfinder",
        "cuda_python",
        "cuda_python_test_helpers",
        "toolshed",
    }
)

# Artifacts that tools drop at the top level. Anything else ignored at the top
# level is left alone -- this is an allowlist, not a denylist, so unrecognized
# state is preserved by default.
CLEAN_TOP_LEVEL_ARTIFACTS = frozenset(
    {
        ".benchmarks",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "dist",
    }
)

# Directory names protected at any depth. `.pixi` holds a pixi workspace's
# solved environments, which take a long time to rebuild and are never the cause
# of a stale-artifact build failure. The rest are hand-made virtualenvs that
# contributors keep next to a sub-package; they are developer state, not output
# of a build in this repository.
PROTECTED_DIR_NAMES = frozenset({".pixi", ".venv", "venv", ".env"})


def repo_root() -> Path:
    """Return the top level of the worktree containing this script."""
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],  # noqa: S607
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        check=True,
        text=True,
    )
    return Path(out.stdout.strip())


def should_clean(rel_path: str) -> bool:
    """Return True if `rel_path` (repo-relative, posix, git-style) may be removed."""
    parts = PurePosixPath(rel_path).parts
    if not parts:
        return False
    if any(part in PROTECTED_DIR_NAMES for part in parts):
        return False
    if len(parts) == 1:
        return parts[0] in CLEAN_TOP_LEVEL_ARTIFACTS
    return parts[0] in CLEAN_TOP_LEVEL_DIRS


def ignored_paths(root: Path) -> list[str]:
    """Return the repo-relative ignored paths that are safe to remove.

    `git status --ignored --porcelain -z` reports an ignored directory as a
    single entry rather than recursing into it, so the returned list contains
    the outermost path of each artifact tree.
    """
    out = subprocess.run(
        ["git", "status", "--porcelain", "-z", "--ignored"],  # noqa: S607
        cwd=root,
        capture_output=True,
        check=True,
        text=True,
    )
    paths = []
    for entry in out.stdout.split("\0"):
        # Porcelain v1 format: two status characters, a space, then the path.
        if not entry.startswith("!! "):
            continue
        rel = entry[3:].rstrip("/")
        if should_clean(rel):
            paths.append(rel)
    return sorted(paths)


def size_of(path: Path) -> int:
    """Return the total size in bytes of `path`, recursing into directories."""
    if path.is_dir() and not path.is_symlink():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())
    return path.lstat().st_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Remove generated build artifacts from the worktree.")
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="list what would be removed without removing anything",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    paths = ignored_paths(root)
    if not paths:
        print("Nothing to clean.")
        return 0

    total = 0
    verb = "Would remove" if args.dry_run else "Removing"
    for rel in paths:
        target = root / rel
        total += size_of(target)
        print(f"{verb} {rel}")
        if args.dry_run:
            continue
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()

    action = "Would reclaim" if args.dry_run else "Reclaimed"
    print(f"\n{len(paths)} path(s). {action} {total / 1024 / 1024:.1f} MiB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
