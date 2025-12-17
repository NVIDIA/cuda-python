# Copyright 2024-2025 NVIDIA Corporation.  All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import functools
import os
import sys


@functools.cache
def _impl(sys_path, sub_dirs):
    results = []
    for base in sys_path:
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


def sys_path_find_sub_dirs(sub_dirs):
    return _impl(tuple(sys.path), tuple(sub_dirs))