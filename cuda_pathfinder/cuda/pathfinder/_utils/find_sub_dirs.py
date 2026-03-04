# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import site
import sys
from collections.abc import Sequence


def find_sub_dirs_no_cache(parent_dirs: Sequence[str], sub_dirs: Sequence[str]) -> list[str]:
    results = []
    for base in parent_dirs:
        stack = [(base, 0)]  # (current_path, index into sub_dirs)
        while stack:
            current_path, idx = stack.pop()
            if idx == len(sub_dirs):
                if os.path.isdir(current_path):
                    results.append(current_path)
                continue

            sub = sub_dirs[idx]
            if sub == "*":
                try:
                    entries = sorted(os.listdir(current_path))
                except OSError:
                    continue
                for entry in entries:
                    entry_path = os.path.join(current_path, entry)
                    if os.path.isdir(entry_path):
                        stack.append((entry_path, idx + 1))
            else:
                next_path = os.path.join(current_path, sub)
                if os.path.isdir(next_path):
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
