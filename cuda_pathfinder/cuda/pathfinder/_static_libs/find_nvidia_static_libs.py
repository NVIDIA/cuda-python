# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find CUDA static libraries and artifacts across different installation sources."""

import functools
from typing import Optional

from cuda.pathfinder._static_libs.supported_nvidia_static_libs import (
    SITE_PACKAGES_STATIC_LIBDIRS,
    SUPPORTED_STATIC_LIBS,
)
from cuda.pathfinder._utils.filename_resolver import FilenameResolver
from cuda.pathfinder._utils.path_utils import _abs_norm
from cuda.pathfinder._utils.search_factory import create_static_lib_search_locations
from cuda.pathfinder._utils.toolchain_tracker import SearchContext, get_default_context


@functools.cache
def _find_nvidia_static_lib_default(artifact_name: str) -> Optional[str]:
    """Internal cached version using default context.
    
    Args:
        artifact_name: Canonical name of the artifact to find.
        
    Returns:
        Absolute path to the artifact, or None if not found.
    """
    return _find_nvidia_static_lib_impl(artifact_name, get_default_context())


def _find_nvidia_static_lib_impl(artifact_name: str, context: SearchContext) -> Optional[str]:
    """Implementation of static library finding logic.
    
    Args:
        artifact_name: Canonical name of the artifact to find.
        context: SearchContext for toolchain consistency.
        
    Returns:
        Absolute path to the artifact, or None if not found.
    """
    if artifact_name not in SUPPORTED_STATIC_LIBS:
        raise RuntimeError(f"UNKNOWN {artifact_name=}")

    locations = create_static_lib_search_locations(
        artifact_name=artifact_name,
        site_packages_libdirs=SITE_PACKAGES_STATIC_LIBDIRS,
        filename_variants_func=FilenameResolver.for_static_lib,
    )

    path = context.find(artifact_name, locations)
    return _abs_norm(path) if path else None


def find_nvidia_static_lib(artifact_name: str, *, context: Optional[SearchContext] = None) -> Optional[str]:
    """Locate a CUDA static library or artifact file.

    This function searches for CUDA static libraries and artifacts (like
    bitcode files) across multiple installation sources, ensuring toolchain
    consistency when multiple artifacts are found.

    Args:
        artifact_name: Canonical artifact name (e.g., "libdevice.10.bc", "cudadevrt").
            Platform-specific filenames are resolved automatically:
            
            - "cudadevrt" -> "libcudadevrt.a" on Linux, "cudadevrt.lib" on Windows
            - "libdevice.10.bc" -> "libdevice.10.bc" (same on all platforms)
            
        context: Optional SearchContext for toolchain consistency tracking.
            If None, uses the default module-level context which provides
            caching and consistency across multiple calls.

    Returns:
        Absolute path to the artifact, or None if not found.

    Raises:
        RuntimeError: If ``artifact_name`` is not supported.
        ToolchainMismatchError: If artifact found in different source than
            the context's preferred source.

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
        Basic usage (uses default context with caching):

        >>> from cuda.pathfinder import find_nvidia_static_lib
        >>> path = find_nvidia_static_lib("cudadevrt")
        >>> if path:
        ...     print(f"Found cudadevrt at {path}")

        Finding bitcode files:

        >>> libdevice = find_nvidia_static_lib("libdevice.10.bc")

        Using explicit context for isolated search:

        >>> from cuda.pathfinder import SearchContext, find_nvidia_static_lib
        >>> ctx = SearchContext()
        >>> cudadevrt = find_nvidia_static_lib("cudadevrt", context=ctx)
        >>> libdevice = find_nvidia_static_lib("libdevice.10.bc", context=ctx)
        >>> # Both from same source, or ToolchainMismatchError raised

    Note:
        When using the default context (context=None), results are cached.
        When providing an explicit context, caching is bypassed to allow
        for isolated searches with different consistency requirements.
    """
    if context is None:
        # Use cached version with default context
        return _find_nvidia_static_lib_default(artifact_name)
    else:
        # Bypass cache for explicit context
        return _find_nvidia_static_lib_impl(artifact_name, context)
