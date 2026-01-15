# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for creating standardized search locations.

This module provides utilities to create SearchLocation objects for common
patterns, reducing code duplication across binary and static library finders.
"""

import functools
import glob
import os
from typing import Callable, Sequence

from cuda.pathfinder._utils.env_vars import get_cuda_home_or_path
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS
from cuda.pathfinder._utils.platform_paths import CUDA_TARGETS_LIB_SUBDIRS, PLATFORM
from cuda.pathfinder._utils.toolchain_tracker import SearchLocation, ToolchainSource
from cuda.pathfinder._utils.types import FilenameVariantFunc


@functools.cache
def _discover_cuda_home_lib_subdirs() -> tuple[str, ...]:
    """Discover available CUDA_HOME library subdirectories.
    
    On Linux, this includes cross-compilation targets (targets/*/lib64, etc.).
    On Windows, only standard lib directories are included.
    
    Returns:
        Tuple of subdirectories to search under CUDA_HOME for libraries.
        
    Note:
        This function is cached to avoid repeated glob operations.
    """
    subdirs = list(PLATFORM.cuda_home_lib_subdirs + PLATFORM.cuda_home_nvvm_subdirs)

    # On Linux, also search targets/*/lib64 and targets/*/lib for cross-compilation
    if not IS_WINDOWS:
        cuda_home = get_cuda_home_or_path()
        if cuda_home:
            for lib_subdir in CUDA_TARGETS_LIB_SUBDIRS:
                pattern = os.path.join(cuda_home, "targets", "*", lib_subdir)
                for target_dir in sorted(glob.glob(pattern), reverse=True):
                    # Make relative to cuda_home
                    rel_path = os.path.relpath(target_dir, cuda_home)
                    subdirs.append(rel_path)

    return tuple(subdirs)


def create_standard_search_locations(
    artifact_name: str,
    site_packages_dirs_map: dict[str, list[str]],
    conda_subdirs: Sequence[str],
    cuda_home_subdirs: Sequence[str] | Callable[[], Sequence[str]],
    filename_variants_func: FilenameVariantFunc,
) -> list[SearchLocation]:
    """Create standard SITE_PACKAGES/CONDA/CUDA_HOME search locations.
    
    This factory function creates a standardized list of search locations
    for CUDA artifacts, following the common pattern of checking:
    1. Site-packages (from pip/conda packages)
    2. Conda environment
    3. CUDA_HOME/CUDA_PATH
    
    Args:
        artifact_name: Name of the artifact to search for.
        site_packages_dirs_map: Mapping from artifact name to relative
            site-packages directories.
        conda_subdirs: Subdirectories to search under CONDA_PREFIX.
        cuda_home_subdirs: Either a sequence of subdirectories to search
            under CUDA_HOME, or a callable that returns such a sequence
            (for dynamic discovery).
        filename_variants_func: Function to generate platform-specific
            filename variants for the artifact.
    
    Returns:
        List of SearchLocation objects in priority order.
    """
    locations = []

    # Site-packages: Create separate SearchLocation for each found directory
    site_package_rel_paths = site_packages_dirs_map.get(artifact_name)
    if site_package_rel_paths:
        for rel_path in site_package_rel_paths:
            for absolute_dir in find_sub_dirs_all_sitepackages(tuple(rel_path.split("/"))):
                locations.append(
                    SearchLocation(
                        source=ToolchainSource.SITE_PACKAGES,
                        base_dir_func=lambda d=absolute_dir: d,
                        subdirs=("",),  # Already have full path
                        filename_variants=filename_variants_func,
                    )
                )

    # Conda: Generic locations
    locations.append(
        SearchLocation(
            source=ToolchainSource.CONDA,
            base_dir_func=lambda: os.environ.get("CONDA_PREFIX"),
            subdirs=conda_subdirs,
            filename_variants=filename_variants_func,
        )
    )

    # CUDA_HOME: Generic locations (may include dynamic discovery)
    if callable(cuda_home_subdirs):
        cuda_home_subdirs = cuda_home_subdirs()

    locations.append(
        SearchLocation(
            source=ToolchainSource.CUDA_HOME,
            base_dir_func=get_cuda_home_or_path,
            subdirs=cuda_home_subdirs,
            filename_variants=filename_variants_func,
        )
    )

    return locations


# Convenience functions for specific artifact types

def create_binary_search_locations(
    binary_name: str,
    site_packages_bindirs: dict[str, list[str]],
    filename_variants_func: FilenameVariantFunc,
) -> list[SearchLocation]:
    """Create search locations specifically for binaries.
    
    Args:
        binary_name: Name of the binary to search for.
        site_packages_bindirs: Mapping from binary name to site-packages
            binary directories.
        filename_variants_func: Function to generate binary filename variants.
    
    Returns:
        List of SearchLocation objects for binary search.
    """
    return create_standard_search_locations(
        artifact_name=binary_name,
        site_packages_dirs_map=site_packages_bindirs,
        conda_subdirs=PLATFORM.conda_bin_subdirs,
        cuda_home_subdirs=PLATFORM.cuda_home_bin_subdirs,
        filename_variants_func=filename_variants_func,
    )


def create_static_lib_search_locations(
    artifact_name: str,
    site_packages_libdirs: dict[str, list[str]],
    filename_variants_func: FilenameVariantFunc,
) -> list[SearchLocation]:
    """Create search locations specifically for static libraries.
    
    Args:
        artifact_name: Name of the static library to search for.
        site_packages_libdirs: Mapping from artifact name to site-packages
            library directories.
        filename_variants_func: Function to generate library filename variants.
    
    Returns:
        List of SearchLocation objects for static library search.
    """
    return create_standard_search_locations(
        artifact_name=artifact_name,
        site_packages_dirs_map=site_packages_libdirs,
        conda_subdirs=PLATFORM.conda_lib_subdirs + PLATFORM.conda_nvvm_subdirs,
        cuda_home_subdirs=_discover_cuda_home_lib_subdirs,
        filename_variants_func=filename_variants_func,
    )
