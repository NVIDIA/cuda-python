# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import glob
import os
from typing import Optional

from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import IS_WINDOWS
from cuda.pathfinder._headers import supported_nvidia_headers
from cuda.pathfinder._utils.env_vars import get_cuda_home_or_path
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages


def _find_nvshmem_header_directory() -> Optional[str]:
    if IS_WINDOWS:
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


def _find_based_on_ctk_layout(libname: str, h_basename: str, anchor_point: str) -> Optional[str]:
    if libname == "nvvm":
        idir = os.path.join(anchor_point, "nvvm", "include")
        h_path = os.path.join(idir, h_basename)
        if os.path.isfile(h_path):
            return idir
    else:
        idir = os.path.join(anchor_point, "include")
        if libname in supported_nvidia_headers.CCCL_LIBNAMES:
            cdir = os.path.join(idir, "cccl")
            h_path = os.path.join(cdir, h_basename)
            if os.path.isfile(h_path):
                return cdir
        h_path = os.path.join(idir, h_basename)
        if os.path.isfile(h_path):
            return idir

    return None


def _find_ctk_header_directory(libname: str) -> Optional[str]:
    h_basename = supported_nvidia_headers.SUPPORTED_HEADERS_CTK[libname]
    candidate_dirs = supported_nvidia_headers.SUPPORTED_SITE_PACKAGE_HEADER_DIRS_CTK[libname]

    # Installed from a wheel
    for cdir in candidate_dirs:
        hdr_dir: str  # help mypy
        for hdr_dir in find_sub_dirs_all_sitepackages(tuple(cdir.split("/"))):
            h_path = os.path.join(hdr_dir, h_basename)
            if os.path.isfile(h_path):
                return hdr_dir

    conda_prefix = os.getenv("CONDA_PREFIX")
    if conda_prefix:  # noqa: SIM102
        if result := _find_based_on_ctk_layout(libname, h_basename, conda_prefix):
            return result

    cuda_home = get_cuda_home_or_path()
    if cuda_home:  # noqa: SIM102
        if result := _find_based_on_ctk_layout(libname, h_basename, cuda_home):
            return result

    return None


@functools.cache
def find_nvidia_header_directory(libname: str) -> Optional[str]:
    if libname == "nvshmem":
        return _find_nvshmem_header_directory()

    if libname in supported_nvidia_headers.SUPPORTED_HEADERS_CTK:
        return _find_ctk_header_directory(libname)

    raise RuntimeError(f"UNKNOWN {libname=}")
