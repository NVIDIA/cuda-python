# Copyright 2024-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import functools
import os
import site
import sys


def find_sub_dirs_no_cache(parent_dirs, sub_dirs):
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
def find_sub_dirs_cached(parent_dirs, sub_dirs):
    return find_sub_dirs_no_cache(parent_dirs, sub_dirs)


def find_sub_dirs(parent_dirs, sub_dirs):
    return find_sub_dirs_cached(tuple(parent_dirs), tuple(sub_dirs))


def find_sub_dirs_sys_path(sub_dirs):
    return find_sub_dirs(sys.path, sub_dirs)


def find_sub_dirs_all_sitepackages(sub_dirs):
    return find_sub_dirs((site.getusersitepackages(),) + tuple(site.getsitepackages()), sub_dirs)
