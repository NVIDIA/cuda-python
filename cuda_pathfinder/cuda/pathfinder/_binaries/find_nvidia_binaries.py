# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find CUDA binary executables across different installation sources."""

import functools
from typing import Optional

from cuda.pathfinder._binaries.supported_nvidia_binaries import SITE_PACKAGES_BINDIRS, SUPPORTED_BINARIES
from cuda.pathfinder._utils.filename_resolver import FilenameResolver
from cuda.pathfinder._utils.path_utils import _abs_norm
from cuda.pathfinder._utils.search_factory import create_binary_search_locations
from cuda.pathfinder._utils.toolchain_tracker import SearchContext, get_default_context


@functools.cache
def _find_nvidia_binary_default(binary_name: str) -> Optional[str]:
    """Internal cached version using default context.
    
    Args:
        binary_name: Name of the binary to find.
        
    Returns:
        Absolute path to the binary, or None if not found.
    """
    return _find_nvidia_binary_impl(binary_name, get_default_context())


def _find_nvidia_binary_impl(binary_name: str, context: SearchContext) -> Optional[str]:
    """Implementation of binary finding logic.
    
    Args:
        binary_name: Name of the binary to find.
        context: SearchContext for toolchain consistency.
        
    Returns:
        Absolute path to the binary, or None if not found.
    """
    if binary_name not in SUPPORTED_BINARIES:
        raise RuntimeError(f"UNKNOWN {binary_name=}")

    locations = create_binary_search_locations(
        binary_name=binary_name,
        site_packages_bindirs=SITE_PACKAGES_BINDIRS,
        filename_variants_func=FilenameResolver.for_binary,
    )

    path = context.find(binary_name, locations)
    return _abs_norm(path) if path else None


def find_nvidia_binary(binary_name: str, *, context: Optional[SearchContext] = None) -> Optional[str]:
    """Locate a CUDA binary executable.

    This function searches for CUDA binaries across multiple installation
    sources, ensuring toolchain consistency when multiple artifacts are found.

    Args:
        binary_name: The name of the binary to find (e.g., ``"nvdisasm"``,
            ``"cuobjdump"``).
        context: Optional SearchContext for toolchain consistency tracking.
            If None, uses the default module-level context which provides
            caching and consistency across multiple calls.

    Returns:
        Absolute path to the discovered binary, or ``None`` if the
        binary cannot be found.

    Raises:
        RuntimeError: If ``binary_name`` is not in the supported set.
        ToolchainMismatchError: If binary found in different source than
            the context's preferred source.

    Search order:
        1. **NVIDIA Python wheels**

           - Scan installed distributions (``site-packages``) for binaries
             shipped in NVIDIA wheels (e.g., ``cuda-nvcc``).

        2. **Conda environments**

           - Check Conda-style installation prefixes (``$CONDA_PREFIX/bin`` on
             Linux/Mac or ``$CONDA_PREFIX/Library/bin`` on Windows).

        3. **CUDA Toolkit environment variables**

           - Use ``CUDA_HOME`` or ``CUDA_PATH`` (in that order) and look in the
             ``bin`` subdirectory.

    Examples:
        Basic usage (uses default context with caching):

        >>> from cuda.pathfinder import find_nvidia_binary
        >>> path = find_nvidia_binary("nvcc")
        >>> if path:
        ...     print(f"Found nvcc at {path}")

        Using explicit context for isolated search:

        >>> from cuda.pathfinder import SearchContext, find_nvidia_binary
        >>> ctx = SearchContext()
        >>> nvcc = find_nvidia_binary("nvcc", context=ctx)
        >>> nvdisasm = find_nvidia_binary("nvdisasm", context=ctx)
        >>> # Both from same source, or ToolchainMismatchError raised

    Note:
        When using the default context (context=None), results are cached.
        When providing an explicit context, caching is bypassed to allow
        for isolated searches with different consistency requirements.
    """
    if context is None:
        # Use cached version with default context
        return _find_nvidia_binary_default(binary_name)
    else:
        # Bypass cache for explicit context
        return _find_nvidia_binary_impl(binary_name, context)
