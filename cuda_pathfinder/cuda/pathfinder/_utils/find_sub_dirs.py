# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import site
import sys
from collections.abc import Sequence
from pathlib import Path


def _is_dir(path: Path) -> bool:
    """``path.is_dir()``, but False instead of raising on an inaccessible path.

    This walks directories nobody here controls, so it has to tolerate whatever
    it runs into. Path.is_dir() only swallows the errnos in pathlib's ignore
    list, and raises for the rest (EACCES, ENAMETOOLONG); os.path.isdir, which
    this replaces, returned False for all of them.
    """
    try:
        return path.is_dir()
    except OSError:
        return False


def find_sub_dirs_no_cache(parent_dirs: Sequence[str], sub_dirs: Sequence[str]) -> list[str]:
    # Results stay str: they are consumed by _binaries, _dynamic_libs, _headers
    # and _static_libs, so the type flip belongs in its own change.
    results = []
    for base in parent_dirs:
        stack = [(Path(base), 0)]  # (current_path, index into sub_dirs)
        while stack:
            current_path, idx = stack.pop()
            if idx == len(sub_dirs):
                if _is_dir(current_path):
                    results.append(str(current_path))
                continue

            sub = sub_dirs[idx]
            if sub == "*":
                try:
                    entries = sorted(current_path.iterdir(), key=lambda entry: entry.name)
                except OSError:
                    continue
                for entry_path in entries:
                    if _is_dir(entry_path):
                        stack.append((entry_path, idx + 1))
            else:
                next_path = current_path / sub
                if _is_dir(next_path):
                    stack.append((next_path, idx + 1))
    return results


@functools.cache
def find_sub_dirs_cached(parent_dirs: Sequence[str], sub_dirs: Sequence[str]) -> list[str]:
    return find_sub_dirs_no_cache(parent_dirs, sub_dirs)


def find_sub_dirs(parent_dirs: Sequence[str], sub_dirs: Sequence[str]) -> list[str]:
    return find_sub_dirs_cached(tuple(parent_dirs), tuple(sub_dirs))


def find_sub_dirs_sys_path(sub_dirs: Sequence[str]) -> list[str]:
    return find_sub_dirs(sys.path, sub_dirs)


def find_sub_dirs_all_sitepackages(sub_dirs: Sequence[str]) -> list[str]:
    parent_dirs = list(site.getsitepackages())
    if site.ENABLE_USER_SITE:
        user_site = site.getusersitepackages()
        if user_site:
            # Determine insertion index based on whether we're in a virtual environment (fixes #1716):
            # - In venv (PEP 405): venv site-packages should come first, then user-site-packages,
            #   then system site-packages. Insert at index 1 (after venv, before system).
            # - Not in venv (PEP 370): user-site-packages should come before system site-packages.
            #   Insert at index 0.
            # Detect venv by checking if sys.prefix differs from sys.base_prefix
            insert_idx = 1 if sys.prefix != sys.base_prefix else 0
            parent_dirs.insert(insert_idx, user_site)
    return find_sub_dirs(parent_dirs, sub_dirs)
