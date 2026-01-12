# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from typing import Sequence

from cuda.pathfinder._static_libs.supported_nvidia_static_libs import (
    SITE_PACKAGES_STATIC_LIBDIRS,
    SUPPORTED_STATIC_LIBS,
)
from cuda.pathfinder._utils.env_vars import get_cuda_home_or_path
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.path_utils import _abs_norm
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS
from cuda.pathfinder._utils.toolchain_tracker import (
    SearchContext,
    SearchLocation,
    ToolchainSource,
    get_default_context,
)


# Generic search locations by toolchain source (platform-specific at import time)
if IS_WINDOWS:
    CONDA_LIB_SUBDIRS = ("Library/lib", "Library/lib/x64")
    CONDA_NVVM_SUBDIRS = ("Library/nvvm/libdevice",)
    CUDA_HOME_LIB_SUBDIRS = ("lib/x64", "lib")
    CUDA_HOME_NVVM_SUBDIRS = ("nvvm/libdevice",)
else:
    CONDA_LIB_SUBDIRS = ("lib",)
    CONDA_NVVM_SUBDIRS = ("nvvm/libdevice",)
    CUDA_HOME_LIB_SUBDIRS = ("lib64", "lib")
    CUDA_HOME_NVVM_SUBDIRS = ("nvvm/libdevice",)


def _static_lib_filename_variants(artifact_name: str) -> Sequence[str]:
    """Generate platform-appropriate filename variants for an artifact.

    Args:
        artifact_name: Canonical artifact name (e.g., "cudadevrt", "libdevice.10.bc").

    Returns:
        Sequence of filenames to search for on this platform.

    Examples:
        On Windows:
            "cudadevrt" -> ("cudadevrt.lib",)
            "libdevice.10.bc" -> ("libdevice.10.bc",)
        On Linux:
            "cudadevrt" -> ("libcudadevrt.a",)
            "libdevice.10.bc" -> ("libdevice.10.bc",)
    """
    # Files that are the same on all platforms (e.g., .bc bitcode files)
    if "." in artifact_name:
        return (artifact_name,)

    # Platform-specific library naming conventions
    if IS_WINDOWS:
        return (f"{artifact_name}.lib",)
    else:
        return (f"lib{artifact_name}.a",)


def _get_cuda_home_subdirs_with_targets() -> tuple[str, ...]:
    """Get CUDA_HOME subdirectories including expanded targets/* paths.

    Returns:
        Tuple of subdirectories to search under CUDA_HOME.
    """
    import glob

    subdirs = list(CUDA_HOME_LIB_SUBDIRS + CUDA_HOME_NVVM_SUBDIRS)

    # On Linux, also search targets/*/lib64 and targets/*/lib for cross-compilation
    if not IS_WINDOWS:
        cuda_home = get_cuda_home_or_path()
        if cuda_home:
            for lib_subdir in ("lib64", "lib"):
                pattern = os.path.join(cuda_home, "targets", "*", lib_subdir)
                for target_dir in sorted(glob.glob(pattern), reverse=True):
                    # Make relative to cuda_home
                    rel_path = os.path.relpath(target_dir, cuda_home)
                    subdirs.append(rel_path)

    return tuple(subdirs)


def _create_search_locations(artifact_name: str) -> list[SearchLocation]:
    """Create generic search location configurations.

    Args:
        artifact_name: Name of the artifact to search for.

    Returns:
        List of SearchLocation objects to try.
    """
    locations = []

    # Site-packages: Create separate SearchLocation for each found directory
    relative_directories = SITE_PACKAGES_STATIC_LIBDIRS.get(artifact_name)
    if relative_directories:
        for relative_directory in relative_directories:
            for found_dir in find_sub_dirs_all_sitepackages(tuple(relative_directory.split("/"))):
                locations.append(
                    SearchLocation(
                        source=ToolchainSource.SITE_PACKAGES,
                        base_dir_func=lambda d=found_dir: d,
                        subdirs=[""],
                        filename_variants=_static_lib_filename_variants,
                    )
                )

    # Conda: Generic lib and nvvm locations
    locations.append(
        SearchLocation(
            source=ToolchainSource.CONDA,
            base_dir_func=lambda: os.environ.get("CONDA_PREFIX"),
            subdirs=CONDA_LIB_SUBDIRS + CONDA_NVVM_SUBDIRS,
            filename_variants=_static_lib_filename_variants,
        )
    )

    # CUDA_HOME: Generic lib and nvvm locations (including targets/* on Linux)
    locations.append(
        SearchLocation(
            source=ToolchainSource.CUDA_HOME,
            base_dir_func=get_cuda_home_or_path,
            subdirs=_get_cuda_home_subdirs_with_targets(),
            filename_variants=_static_lib_filename_variants,
        )
    )

    return locations


@functools.cache
def find_nvidia_static_lib(artifact_name: str, *, context: SearchContext | None = None) -> str | None:
    """Locate a CUDA static library or artifact file.

    Args:
        artifact_name: Canonical artifact name (e.g., "libdevice.10.bc", "cudadevrt").
            Platform-specific filenames are resolved automatically:
            - "cudadevrt" -> "libcudadevrt.a" on Linux, "cudadevrt.lib" on Windows
        context: Optional SearchContext for toolchain consistency tracking.
            If None, uses the default module-level context.

    Returns:
        Absolute path to the artifact, or None if not found.

    Raises:
        RuntimeError: If artifact_name is not supported.
        ToolchainMismatchError: If artifact found in different source than
            the context's preferred source.
    """
    if artifact_name not in SUPPORTED_STATIC_LIBS:
        raise RuntimeError(f"UNKNOWN {artifact_name=}")

    if context is None:
        context = get_default_context()

    locations = _create_search_locations(artifact_name)
    path = context.find(artifact_name, locations)
    return _abs_norm(path) if path else None
