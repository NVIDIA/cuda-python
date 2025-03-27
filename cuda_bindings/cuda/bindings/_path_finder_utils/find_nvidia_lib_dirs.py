# Copyright 2024-2025 NVIDIA Corporation.  All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import functools
import os
import sys


@functools.cache
def _find_nvidia_lib_dirs(sys_path):
    results = []
    for base in sys_path:
        nvidia_base = os.path.join(base, "nvidia")
        if not os.path.isdir(nvidia_base):
            continue
        try:
            subdirs = os.listdir(nvidia_base)
        except OSError:
            continue
        for sub in subdirs:
            sub_path = os.path.join(nvidia_base, sub)
            lib_path = os.path.join(sub_path, "lib")
            if os.path.isdir(lib_path):
                results.append(lib_path)
    return results


def find_nvidia_lib_dirs():
    return _find_nvidia_lib_dirs(tuple(sys.path))
