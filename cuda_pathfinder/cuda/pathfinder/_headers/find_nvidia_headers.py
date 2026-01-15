# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import glob
import os

from cuda.pathfinder._headers import supported_nvidia_headers
from cuda.pathfinder._utils.env_vars import get_cuda_home_or_path
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS


def _abs_norm(path: str | None) -> str | None:
    """Normalize and return absolute path, or None if path is None."""
    if path:
        return os.path.normpath(os.path.abspath(path))
    return None


def _joined_isfile(dirpath: str, basename: str) -> bool:
    return os.path.isfile(os.path.join(dirpath, basename))


def _find_under_site_packages(sub_dir: str, h_basename: str) -> str | None:
    # Installed from a wheel
    hdr_dir: str  # help mypy
    for hdr_dir in find_sub_dirs_all_sitepackages(tuple(sub_dir.split("/"))):
        if _joined_isfile(hdr_dir, h_basename):
            return hdr_dir
    return None


def _find_based_on_ctk_layout(libname: str, h_basename: str, anchor_point: str) -> str | None:
    parts = [anchor_point]
    if libname == "nvvm":
        parts.append(libname)
    parts.append("include")
    idir = os.path.join(*parts)
    if libname == "cccl":
        if IS_WINDOWS:
            cdir_ctk12 = os.path.join(idir, "targets", "x64")  # conda has this anomaly
            cdir_ctk13 = os.path.join(cdir_ctk12, "cccl")
            if _joined_isfile(cdir_ctk13, h_basename):
                return cdir_ctk13
            if _joined_isfile(cdir_ctk12, h_basename):
                return cdir_ctk12
        cdir = os.path.join(idir, "cccl")  # CTK 13
        if _joined_isfile(cdir, h_basename):
            return cdir
    if _joined_isfile(idir, h_basename):
        return idir
    return None


def _find_based_on_conda_layout(libname: str, h_basename: str, ctk_layout: bool) -> str | None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None
    if IS_WINDOWS:
        anchor_point = os.path.join(conda_prefix, "Library")
        if not os.path.isdir(anchor_point):
            return None
    else:
        if ctk_layout:
            targets_include_path = glob.glob(os.path.join(conda_prefix, "targets", "*", "include"))
            if not targets_include_path:
                return None
            if len(targets_include_path) != 1:
                # Conda does not support multiple architectures.
                # QUESTION(PR#956): Do we want to issue a warning?
                return None
            include_path = targets_include_path[0]
        else:
            include_path = os.path.join(conda_prefix, "include")
        anchor_point = os.path.dirname(include_path)
    return _find_based_on_ctk_layout(libname, h_basename, anchor_point)


def _find_ctk_header_directory(libname: str) -> str | None:
    h_basename = supported_nvidia_headers.SUPPORTED_HEADERS_CTK[libname]
    candidate_dirs = supported_nvidia_headers.SUPPORTED_SITE_PACKAGE_HEADER_DIRS_CTK[libname]

    for cdir in candidate_dirs:
        if hdr_dir := _find_under_site_packages(cdir, h_basename):
            return hdr_dir

    if hdr_dir := _find_based_on_conda_layout(libname, h_basename, True):
        return hdr_dir

    cuda_home = get_cuda_home_or_path()
    if cuda_home:  # noqa: SIM102
        if result := _find_based_on_ctk_layout(libname, h_basename, cuda_home):
            return result

    return None


@functools.cache
def find_nvidia_header_directory(libname: str) -> str | None:
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
    """

    if libname in supported_nvidia_headers.SUPPORTED_HEADERS_CTK:
        return _abs_norm(_find_ctk_header_directory(libname))

    h_basename = supported_nvidia_headers.SUPPORTED_HEADERS_NON_CTK.get(libname)
    if h_basename is None:
        raise RuntimeError(f"UNKNOWN {libname=}")

    candidate_dirs = supported_nvidia_headers.SUPPORTED_SITE_PACKAGE_HEADER_DIRS_NON_CTK.get(libname, [])
    hdr_dir: str | None  # help mypy
    for cdir in candidate_dirs:
        if hdr_dir := _find_under_site_packages(cdir, h_basename):
            return _abs_norm(hdr_dir)

    if hdr_dir := _find_based_on_conda_layout(libname, h_basename, False):
        return _abs_norm(hdr_dir)

    candidate_dirs = supported_nvidia_headers.SUPPORTED_INSTALL_DIRS_NON_CTK.get(libname, [])
    for cdir in candidate_dirs:
        for hdr_dir in sorted(glob.glob(cdir), reverse=True):
            if _joined_isfile(hdr_dir, h_basename):
                return _abs_norm(hdr_dir)

    return None
