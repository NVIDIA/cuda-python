# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import glob
import os

from cuda.pathfinder._binaries import supported_nvidia_binaries
from cuda.pathfinder._utils.env_vars import get_cuda_home_or_path
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.path_utils import _abs_norm, _is_executable
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS


def _normalize_utility_name(utility_name: str) -> str:
    """Normalize utility name by adding .exe on Windows if needed."""
    if IS_WINDOWS and not utility_name.lower().endswith((".exe", ".bat", ".cmd")):
        return utility_name + ".exe"
    return utility_name


def _find_under_site_packages(sub_dir: str, utility_name: str) -> str | None:
    """Search for binary in site-packages subdirectories."""
    bin_path: str
    normalized_name = _normalize_utility_name(utility_name)
    for bin_dir in find_sub_dirs_all_sitepackages(tuple(sub_dir.split("/"))):
        bin_path = os.path.join(bin_dir, normalized_name)
        if _is_executable(bin_path):
            return bin_path
    return None


def _find_based_on_cuda_toolkit_layout(utility_name: str, anchor_point: str) -> str | None:
    """Search in CUDA Toolkit style bin directories."""
    normalized_name = _normalize_utility_name(utility_name)

    # Windows: try bin/x64, bin/x86_64, bin; Linux: just bin
    rel_paths = ["bin/x64", "bin/x86_64", "bin"] if IS_WINDOWS else ["bin"]

    for rel_path in rel_paths:
        for bin_dir in sorted(glob.glob(os.path.join(anchor_point, rel_path))):
            if not os.path.isdir(bin_dir):
                continue
            bin_path = os.path.join(bin_dir, normalized_name)
            if _is_executable(bin_path):
                return bin_path

    return None


def _find_based_on_conda_layout(utility_name: str) -> str | None:
    """Search in Conda environment bin directories."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None

    anchor_points = [os.path.join(conda_prefix, "Library"), conda_prefix] if IS_WINDOWS else [conda_prefix]

    for anchor_point in anchor_points:
        if not os.path.isdir(anchor_point):
            continue
        if result := _find_based_on_cuda_toolkit_layout(utility_name, anchor_point):
            return result

    return None


def _find_using_cuda_home(utility_name: str) -> str | None:
    """Search using CUDA_HOME or CUDA_PATH environment variables."""
    cuda_home = get_cuda_home_or_path()
    if cuda_home is None:
        return None
    return _find_based_on_cuda_toolkit_layout(utility_name, cuda_home)


def _find_binary_utility(utility_name: str) -> str | None:
    """Core search logic for finding a binary utility."""
    # 1. Search in site-packages (NVIDIA wheels)
    candidate_dirs = supported_nvidia_binaries.SITE_PACKAGES_BINDIRS.get(utility_name, ())
    for cdir in candidate_dirs:
        if bin_path := _find_under_site_packages(cdir, utility_name):
            return _abs_norm(bin_path)

    # 2. Search in Conda environment
    if bin_path := _find_based_on_conda_layout(utility_name):
        return _abs_norm(bin_path)

    # 3. Search in CUDA Toolkit (CUDA_HOME/CUDA_PATH)
    if bin_path := _find_using_cuda_home(utility_name):
        return _abs_norm(bin_path)

    return None


@functools.cache
def find_nvidia_binary_utility(utility_name: str) -> str | None:
    """Locate a CUDA binary utility executable.

    Args:
        utility_name (str): The name of the binary utility to find
            (e.g., ``"nvdisasm"``, ``"cuobjdump"``).

    Returns:
        str or None: Absolute path to the discovered executable, or ``None``
        if the utility cannot be found.

    Raises:
        RuntimeError: If ``utility_name`` is not in the supported set.

    Search order:
        1. **NVIDIA Python wheels**

           - Scan installed distributions (``site-packages``) for binary layouts
             shipped in NVIDIA wheels (e.g., ``cuda-nvcc``).

        2. **Conda environments**

           - Check Conda-style installation prefixes, which use platform-specific
             bin directory layouts.

        3. **CUDA Toolkit environment variables**

           - Use ``CUDA_HOME`` or ``CUDA_PATH`` (in that order).

    Example:
        >>> from cuda.pathfinder import find_nvidia_binary_utility
        >>> nvdisasm = find_nvidia_binary_utility("nvdisasm")
        >>> if nvdisasm:
        ...     print(f"Found nvdisasm at: {nvdisasm}")
    """
    if utility_name not in supported_nvidia_binaries.SUPPORTED_BINARIES:
        raise RuntimeError(f"UNKNOWN {utility_name=}")

    return _find_binary_utility(utility_name)
