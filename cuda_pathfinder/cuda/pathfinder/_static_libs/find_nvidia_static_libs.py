# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import glob
import os
from typing import Optional, Sequence

from cuda.pathfinder._static_libs.artifact_search_config import ARTIFACT_CONFIGS
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


def _static_lib_filename_variants(artifact_name: str) -> Sequence[str]:
    """Get filename variants for a static library artifact.

    Args:
        artifact_name: The artifact name to get filenames for.

    Returns:
        Sequence of filenames to search for.
    """
    config = ARTIFACT_CONFIGS.get(artifact_name)
    if not config:
        return (artifact_name,)
    return config.filenames


def _get_site_packages_subdirs(artifact_name: str) -> Sequence[str]:
    """Get site-packages subdirectories for an artifact.

    Args:
        artifact_name: Name of the artifact.

    Returns:
        List of absolute paths to search in site-packages, or empty if not available.
    """
    rel_dirs = SITE_PACKAGES_STATIC_LIBDIRS.get(artifact_name)
    if not rel_dirs:
        return []

    subdirs = []
    for rel_dir in rel_dirs:
        for found_dir in find_sub_dirs_all_sitepackages(tuple(rel_dir.split("/"))):
            subdirs.append(found_dir)
    return subdirs


def _get_conda_subdirs(artifact_name: str) -> Sequence[str]:
    """Get conda subdirectories for an artifact.

    Args:
        artifact_name: Name of the artifact.

    Returns:
        List of subdirectories relative to CONDA_PREFIX.
    """
    config = ARTIFACT_CONFIGS.get(artifact_name)
    if not config:
        return []
    return config.conda_dirs


def _get_cuda_home_subdirs(artifact_name: str) -> Sequence[str]:
    """Get CUDA_HOME subdirectories for an artifact, including targets/ search if needed.

    Args:
        artifact_name: Name of the artifact.

    Returns:
        List of subdirectories to search.
    """
    config = ARTIFACT_CONFIGS.get(artifact_name)
    if not config:
        return []

    subdirs = list(config.cuda_home_dirs)

    # Add targets/* expansion for cross-compilation (Linux only)
    if config.search_targets_subdirs and not IS_WINDOWS:
        cuda_home = get_cuda_home_or_path()
        if cuda_home:
            for lib_subdir in ("lib64", "lib"):
                pattern = os.path.join(cuda_home, "targets", "*", lib_subdir)
                for target_dir in sorted(glob.glob(pattern), reverse=True):
                    # Make relative to cuda_home
                    rel_path = os.path.relpath(target_dir, cuda_home)
                    subdirs.append(rel_path)

    return subdirs


def _create_search_locations(artifact_name: str) -> list[SearchLocation]:
    """Create search location configurations for a specific artifact.

    Args:
        artifact_name: Name of the artifact to search for.

    Returns:
        List of SearchLocation objects to try.
    """
    return [
        SearchLocation(
            source=ToolchainSource.SITE_PACKAGES,
            base_dir_func=lambda: None,  # Use subdirs for full paths
            subdirs=_get_site_packages_subdirs(artifact_name),
            filename_variants=lambda _: _static_lib_filename_variants(artifact_name),
        ),
        SearchLocation(
            source=ToolchainSource.CONDA,
            base_dir_func=lambda: os.environ.get("CONDA_PREFIX"),
            subdirs=_get_conda_subdirs(artifact_name),
            filename_variants=lambda _: _static_lib_filename_variants(artifact_name),
        ),
        SearchLocation(
            source=ToolchainSource.CUDA_HOME,
            base_dir_func=get_cuda_home_or_path,
            subdirs=_get_cuda_home_subdirs(artifact_name),
            filename_variants=lambda _: _static_lib_filename_variants(artifact_name),
        ),
    ]


@functools.cache
def find_nvidia_static_lib(artifact_name: str, *, context: SearchContext | None = None) -> str | None:
    """Locate a CUDA static library or artifact file.

    Args:
        artifact_name: Name of the artifact (e.g., "libdevice.10.bc", "libcudadevrt.a").
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
