# Copyright 2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import functools
import glob
import os

from cuda.bindings._path_finder.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.bindings._path_finder.supported_libs import IS_WINDOWS


@functools.cache
def find_nvidia_header_directory(libname: str) -> str:
    assert libname == "nvshmem"
    assert not IS_WINDOWS

    # Installed from a wheel
    nvidia_sub_dirs = ("nvidia", "nvshmem", "include")
    for hdr_dir in find_sub_dirs_all_sitepackages(nvidia_sub_dirs):
        nvshmem_h_path = os.path.join(hdr_dir, "nvshmem.h")
        if os.path.isfile(nvshmem_h_path):
            return hdr_dir

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and os.path.isdir(conda_prefix):
        hdr_dir = os.path.join(conda_prefix, "include")
        if os.path.isdir(hdr_dir):
            nvshmem_h_path = os.path.join(hdr_dir, "nvshmem.h")
            if os.path.isfile(nvshmem_h_path):
                return hdr_dir

    for hdr_dir in sorted(glob.glob("/usr/include/nvshmem_*")):
        if os.path.isdir(hdr_dir):
            nvshmem_h_path = os.path.join(hdr_dir, "nvshmem.h")
            if os.path.isfile(nvshmem_h_path):
                return hdr_dir

    return None
