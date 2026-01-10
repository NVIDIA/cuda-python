# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from typing import Sequence

from cuda.pathfinder._binaries.supported_nvidia_binaries import SITE_PACKAGES_BINDIRS, SUPPORTED_BINARIES
from cuda.pathfinder._utils.env_vars import get_cuda_home_or_path
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.path_utils import _abs_norm
from cuda.pathfinder._utils.toolchain_tracker import (
    SearchContext,
    SearchLocation,
    ToolchainSource,
    get_default_context,
)


def _binary_filename_variants(name: str) -> Sequence[str]:
    """Generate filename variants for a binary (cross-platform).

    Args:
        name: Base binary name.

    Returns:
        Tuple of possible filenames (e.g., "nvcc", "nvcc.exe").
    """
    return (name, f"{name}.exe")


def _get_site_packages_subdirs(binary_name: str) -> Sequence[str]:
    """Get site-packages subdirectories for a binary.

    Args:
        binary_name: Name of the binary.

    Returns:
        List of subdirectories to search, or empty list if binary not in site-packages.
    """
    rel_dirs = SITE_PACKAGES_BINDIRS.get(binary_name)
    if not rel_dirs:
        return []

    # Expand site-packages paths
    subdirs = []
    for rel_dir in rel_dirs:
        for found_dir in find_sub_dirs_all_sitepackages(tuple(rel_dir.split("/"))):
            subdirs.append(found_dir)
    return subdirs


# Define search locations for binaries
def _create_search_locations(binary_name: str) -> list[SearchLocation]:
    """Create search location configurations for a specific binary.

    Args:
        binary_name: Name of the binary to search for.

    Returns:
        List of SearchLocation objects to try.
    """
    return [
        SearchLocation(
            source=ToolchainSource.SITE_PACKAGES,
            base_dir_func=lambda: None,  # Use subdirs for full paths
            subdirs=_get_site_packages_subdirs(binary_name),
            filename_variants=_binary_filename_variants,
        ),
        SearchLocation(
            source=ToolchainSource.CONDA,
            base_dir_func=lambda: os.environ.get("CONDA_PREFIX"),
            subdirs=["Library/bin", "bin"],  # Windows and Unix layouts
            filename_variants=_binary_filename_variants,
        ),
        SearchLocation(
            source=ToolchainSource.CUDA_HOME,
            base_dir_func=get_cuda_home_or_path,
            subdirs=["bin"],
            filename_variants=_binary_filename_variants,
        ),
    ]


@functools.cache
def find_nvidia_binary(binary_name: str, *, context: SearchContext | None = None) -> str | None:
    """Locate a CUDA binary executable.

    Args:
        binary_name: Name of the binary (e.g., "nvdisasm", "cuobjdump").
        context: Optional SearchContext for toolchain consistency tracking.
            If None, uses the default module-level context.

    Returns:
        Absolute path to the binary, or None if not found.

    Raises:
        RuntimeError: If binary_name is not supported.
        ToolchainMismatchError: If binary found in different source than
            the context's preferred source.
    """
    if binary_name not in SUPPORTED_BINARIES:
        raise RuntimeError(f"UNKNOWN {binary_name=}")

    if context is None:
        context = get_default_context()

    locations = _create_search_locations(binary_name)
    path = context.find(binary_name, locations)
    return _abs_norm(path) if path else None
