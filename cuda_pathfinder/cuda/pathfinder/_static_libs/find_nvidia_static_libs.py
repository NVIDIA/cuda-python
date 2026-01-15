# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find CUDA static libraries and artifacts across different installation sources."""

import functools
import glob
import os
from typing import Optional

from cuda.pathfinder._static_libs.supported_nvidia_static_libs import (
    SITE_PACKAGES_STATIC_LIBDIRS,
    SUPPORTED_STATIC_LIBS,
)
from cuda.pathfinder._utils.env_vars import get_cuda_home_or_path
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS
from cuda.pathfinder._utils.search_order import SEARCH_ORDER_DESCRIPTION


def _get_lib_filename_variants(artifact_name: str) -> tuple[str, ...]:
    """Generate platform-appropriate library filename variants.

    Args:
        artifact_name: Canonical artifact name (e.g., "cudadevrt", "libdevice.10.bc").

    Returns:
        Tuple of possible filenames for the current platform.

    Examples:
        On Linux:
            "cudadevrt" -> ("libcudadevrt.a",)
            "libdevice.10.bc" -> ("libdevice.10.bc",)
        On Windows:
            "cudadevrt" -> ("cudadevrt.lib",)
            "libdevice.10.bc" -> ("libdevice.10.bc",)
    """
    # Files with extensions (e.g., .bc bitcode files) are the same on all platforms
    if "." in artifact_name:
        return (artifact_name,)

    # Platform-specific library naming conventions
    if IS_WINDOWS:
        return (f"{artifact_name}.lib",)
    else:
        return (f"lib{artifact_name}.a",)


@functools.cache
def find_nvidia_static_lib(artifact_name: str) -> Optional[str]:
    """Locate a CUDA static library or artifact file.

    This function searches for CUDA static libraries and artifacts (like
    bitcode files) across multiple installation sources.

    Args:
        artifact_name: Canonical artifact name (e.g., "libdevice.10.bc", "cudadevrt").
            Platform-specific filenames are resolved automatically:

            - "cudadevrt" -> "libcudadevrt.a" on Linux, "cudadevrt.lib" on Windows
            - "libdevice.10.bc" -> "libdevice.10.bc" (same on all platforms)

    Returns:
        Absolute path to the artifact, or ``None`` if not found.

    Raises:
        ValueError: If ``artifact_name`` is not supported.

    Search order:
        1. **NVIDIA Python wheels**

           - Scan installed distributions (``site-packages``) for libraries
             shipped in NVIDIA wheels (e.g., ``cuda-cudart``).

        2. **Conda environments**

           - Check Conda-style installation prefixes:

             - ``$CONDA_PREFIX/lib`` (Linux/Mac)
             - ``$CONDA_PREFIX/Library/lib`` (Windows)
             - ``$CONDA_PREFIX/nvvm/libdevice`` (for bitcode files)

        3. **CUDA Toolkit environment variables**

           - Use ``CUDA_HOME`` or ``CUDA_PATH`` (in that order) and look in:

             - ``lib64``, ``lib`` subdirectories
             - ``nvvm/libdevice`` (for bitcode files)
             - ``targets/*/lib64``, ``targets/*/lib`` (Linux cross-compilation)

    Examples:
        Basic usage:

        >>> from cuda.pathfinder import find_nvidia_static_lib
        >>> path = find_nvidia_static_lib("cudadevrt")
        >>> if path:
        ...     print(f"Found cudadevrt at {path}")

        Finding bitcode files:

        >>> libdevice = find_nvidia_static_lib("libdevice.10.bc")

    Note:
        Results are cached via ``functools.cache`` for performance.
        The search order is centralized and shared across all pathfinder functions.
        See :py:mod:`cuda.pathfinder._utils.search_order` for the canonical definition.
    """
    if artifact_name not in SUPPORTED_STATIC_LIBS:
        raise ValueError(f"Unknown artifact: {artifact_name!r}")

    # Get platform-appropriate filename variants
    variants = _get_lib_filename_variants(artifact_name)

    # 1. Search site-packages (NVIDIA Python wheels)
    if site_dirs := SITE_PACKAGES_STATIC_LIBDIRS.get(artifact_name):
        for rel_path in site_dirs:
            for abs_dir in find_sub_dirs_all_sitepackages(tuple(rel_path.split("/"))):
                for variant in variants:
                    path = os.path.join(abs_dir, variant)
                    if os.path.isfile(path):
                        return os.path.abspath(path)

    # 2. Search Conda environment
    if conda_prefix := os.environ.get("CONDA_PREFIX"):
        if IS_WINDOWS:
            subdirs = ("Library/lib", "Library/lib/x64", "Library/nvvm/libdevice")
        else:
            subdirs = ("lib", "nvvm/libdevice")

        for subdir in subdirs:
            for variant in variants:
                path = os.path.join(conda_prefix, subdir, variant)
                if os.path.isfile(path):
                    return os.path.abspath(path)

    # 3. Search CUDA_HOME/CUDA_PATH
    if cuda_home := get_cuda_home_or_path():
        if IS_WINDOWS:
            subdirs = ["lib/x64", "lib", "nvvm/libdevice"]
        else:
            subdirs = ["lib64", "lib", "nvvm/libdevice"]

            # On Linux, also search cross-compilation targets
            for lib_subdir in ("lib64", "lib"):
                pattern = os.path.join(cuda_home, "targets", "*", lib_subdir)
                for target_dir in sorted(glob.glob(pattern), reverse=True):
                    # Make relative to cuda_home
                    rel_path = os.path.relpath(target_dir, cuda_home)
                    subdirs.append(rel_path)

        for subdir in subdirs:
            for variant in variants:
                path = os.path.join(cuda_home, subdir, variant)
                if os.path.isfile(path):
                    return os.path.abspath(path)

    return None
