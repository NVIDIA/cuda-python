# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find CUDA binary executables across different installation sources."""

import functools
import os
from typing import Optional

from cuda.pathfinder._binaries.supported_nvidia_binaries import SITE_PACKAGES_BINDIRS, SUPPORTED_BINARIES
from cuda.pathfinder._utils.env_vars import get_cuda_home_or_path
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS


@functools.cache
def find_nvidia_binary(binary_name: str) -> Optional[str]:
    """Locate a CUDA binary executable.

    This function searches for CUDA binaries across multiple installation
    sources in priority order.

    Args:
        binary_name: The name of the binary to find (e.g., ``"nvdisasm"``,
            ``"cuobjdump"``).

    Returns:
        Absolute path to the discovered binary, or ``None`` if the
        binary cannot be found.

    Raises:
        ValueError: If ``binary_name`` is not in the supported set.

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
        Basic usage:

        >>> from cuda.pathfinder import find_nvidia_binary
        >>> path = find_nvidia_binary("nvcc")
        >>> if path:
        ...     print(f"Found nvcc at {path}")

    Note:
        Results are cached via ``functools.cache`` for performance.
    """
    if binary_name not in SUPPORTED_BINARIES:
        raise ValueError(f"Unknown binary: {binary_name!r}")

    # Filename variants (try both with and without .exe for cross-platform support)
    variants = (binary_name, f"{binary_name}.exe")

    # 1. Search site-packages (NVIDIA Python wheels)
    if site_dirs := SITE_PACKAGES_BINDIRS.get(binary_name):
        for rel_path in site_dirs:
            for abs_dir in find_sub_dirs_all_sitepackages(tuple(rel_path.split("/"))):
                for variant in variants:
                    path = os.path.join(abs_dir, variant)
                    if os.path.isfile(path):
                        return os.path.abspath(path)

    # 2. Search Conda environment
    if conda_prefix := os.environ.get("CONDA_PREFIX"):
        subdirs = ("Library/bin", "bin") if IS_WINDOWS else ("bin",)
        for subdir in subdirs:
            for variant in variants:
                path = os.path.join(conda_prefix, subdir, variant)
                if os.path.isfile(path):
                    return os.path.abspath(path)

    # 3. Search CUDA_HOME/CUDA_PATH
    if cuda_home := get_cuda_home_or_path():
        for variant in variants:
            path = os.path.join(cuda_home, "bin", variant)
            if os.path.isfile(path):
                return os.path.abspath(path)

    return None
