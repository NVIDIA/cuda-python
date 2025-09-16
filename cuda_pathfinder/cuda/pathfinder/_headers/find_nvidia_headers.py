# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import glob
import os
from typing import Optional

from cuda.pathfinder._headers import supported_nvidia_headers
from cuda.pathfinder._headers.supported_nvidia_headers import IS_WINDOWS
from cuda.pathfinder._utils.env_vars import get_cuda_home_or_path
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages


def _abs_norm(path: Optional[str]) -> Optional[str]:
    if path:
        return os.path.normpath(os.path.abspath(path))
    return None


def _joined_isfile(dirpath: str, basename: str) -> bool:
    return os.path.isfile(os.path.join(dirpath, basename))


def _find_nvshmem_header_directory() -> Optional[str]:
    if IS_WINDOWS:
        # nvshmem has no Windows support.
        return None

    # Installed from a wheel
    nvidia_sub_dirs = ("nvidia", "nvshmem", "include")
    hdr_dir: str  # help mypy
    for hdr_dir in find_sub_dirs_all_sitepackages(nvidia_sub_dirs):
        if _joined_isfile(hdr_dir, "nvshmem.h"):
            return hdr_dir

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and os.path.isdir(conda_prefix):
        hdr_dir = os.path.join(conda_prefix, "include")
        if _joined_isfile(hdr_dir, "nvshmem.h"):
            return hdr_dir

    for hdr_dir in sorted(glob.glob("/usr/include/nvshmem_*"), reverse=True):
        if _joined_isfile(hdr_dir, "nvshmem.h"):
            return hdr_dir

    return None


def _find_based_on_ctk_layout(libname: str, h_basename: str, anchor_point: str) -> Optional[str]:
    parts = [anchor_point]
    if libname == "nvvm":
        parts.append(libname)
    parts.append("include")
    idir = os.path.join(*parts)
    if libname == "cccl":
        cdir = os.path.join(idir, "cccl")  # CTK 13
        if _joined_isfile(cdir, h_basename):
            return cdir
    if _joined_isfile(idir, h_basename):
        return idir
    return None


def _find_based_on_conda_layout(libname: str, h_basename: str, conda_prefix: str) -> Optional[str]:
    if IS_WINDOWS:
        anchor_point = os.path.join(conda_prefix, "Library")
        if not os.path.isdir(anchor_point):
            return None
    else:
        targets_include_path = glob.glob(os.path.join(conda_prefix, "targets", "*", "include"))
        if not targets_include_path:
            return None
        if len(targets_include_path) != 1:
            # Conda does not support multiple architectures.
            # QUESTION(PR#956): Do we want to issue a warning?
            return None
        anchor_point = os.path.dirname(targets_include_path[0])
    return _find_based_on_ctk_layout(libname, h_basename, anchor_point)


def _find_ctk_header_directory(libname: str) -> Optional[str]:
    h_basename = supported_nvidia_headers.SUPPORTED_HEADERS_CTK[libname]
    candidate_dirs = supported_nvidia_headers.SUPPORTED_SITE_PACKAGE_HEADER_DIRS_CTK[libname]

    # Installed from a wheel
    for cdir in candidate_dirs:
        hdr_dir: str  # help mypy
        for hdr_dir in find_sub_dirs_all_sitepackages(tuple(cdir.split("/"))):
            if _joined_isfile(hdr_dir, h_basename):
                return hdr_dir

    conda_prefix = os.getenv("CONDA_PREFIX")
    if conda_prefix:  # noqa: SIM102
        if result := _find_based_on_conda_layout(libname, h_basename, conda_prefix):
            return result

    cuda_home = get_cuda_home_or_path()
    if cuda_home:  # noqa: SIM102
        if result := _find_based_on_ctk_layout(libname, h_basename, cuda_home):
            return result

    return None


@functools.cache
def find_nvidia_header_directory(libname: str) -> Optional[str]:
    """Locate the header directory for a supported NVIDIA library.

    Args:
        libname (str): The short name of the library whose headers are needed
            (e.g., ``"nvrtc"``, ``"cusolver"``, ``"nvshmem"``).

    Returns:
        str or None: Absolute path to the discovered header directory, or ``None``
        if the headers cannot be found.

    Raises:
        RuntimeError: If ``libname`` is not in the supported set.

    Search order:
        1. **NVIDIA Python wheels**

           - Scan installed distributions (``site-packages``) for header layouts
             shipped in NVIDIA wheels (e.g., ``cuda-toolkit[nvrtc]``).

        2. **Conda environments**

           - Check Conda-style installation prefixes, which use platform-specific
             include directory layouts.

        3. **CUDA Toolkit environment variables**

           - Use ``CUDA_HOME`` or ``CUDA_PATH`` (in that order).

    Notes:
        - The ``SUPPORTED_HEADERS_CTK`` dictionary maps each supported CUDA Toolkit
          (CTK) library to the name of its canonical header (e.g., ``"cublas" →
          "cublas.h"``). This is used to verify that the located directory is valid.

        - The only supported non-CTK library at present is ``nvshmem``.
    """

    if libname == "nvshmem":
        return _abs_norm(_find_nvshmem_header_directory())

    if libname in supported_nvidia_headers.SUPPORTED_HEADERS_CTK:
        return _abs_norm(_find_ctk_header_directory(libname))

    raise RuntimeError(f"UNKNOWN {libname=}")
