# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import glob
import os
from typing import Optional

from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import IS_WINDOWS
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages


@functools.cache
def find_nvidia_header_directory(libname: str) -> Optional[str]:
    if libname != "nvshmem":
        raise RuntimeError(f"UNKNOWN {libname=}")

    if libname == "nvshmem" and IS_WINDOWS:
        # nvshmem has no Windows support.
        return None

    # Installed from a wheel
    nvidia_sub_dirs = ("nvidia", "nvshmem", "include")
    hdr_dir: str  # help mypy
    for hdr_dir in find_sub_dirs_all_sitepackages(nvidia_sub_dirs):
        nvshmem_h_path = os.path.join(hdr_dir, "nvshmem.h")
        if os.path.isfile(nvshmem_h_path):
            return hdr_dir

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and os.path.isdir(conda_prefix):
        hdr_dir = os.path.join(conda_prefix, "include")
        nvshmem_h_path = os.path.join(hdr_dir, "nvshmem.h")
        if os.path.isfile(nvshmem_h_path):
            return hdr_dir

    for hdr_dir in sorted(glob.glob("/usr/include/nvshmem_*"), reverse=True):
        nvshmem_h_path = os.path.join(hdr_dir, "nvshmem.h")
        if os.path.isfile(nvshmem_h_path):
            return hdr_dir

    return None
