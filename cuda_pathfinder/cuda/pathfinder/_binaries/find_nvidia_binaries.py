# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os

from cuda.pathfinder._binaries.supported_nvidia_binaries import SITE_PACKAGES_BINDIRS, SUPPORTED_BINARIES
from cuda.pathfinder._utils.env_vars import get_cuda_home_or_path
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.path_utils import _abs_norm
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS


def _find_binary_under_site_packages(binary_name: str) -> str | None:
    """Search for a binary in site-packages directories."""
    rel_dirs = SITE_PACKAGES_BINDIRS.get(binary_name)
    if rel_dirs is None:
        return None

    if IS_WINDOWS:
        binary_filename = f"{binary_name}.exe"
    else:
        binary_filename = binary_name

    for rel_dir in rel_dirs:
        for bin_dir in find_sub_dirs_all_sitepackages(tuple(rel_dir.split("/"))):
            binary_path = os.path.join(bin_dir, binary_filename)
            if os.path.isfile(binary_path):
                return binary_path
    return None


def _find_binary_in_conda(binary_name: str) -> str | None:
    """Search for a binary in conda prefix."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None

    if IS_WINDOWS:
        binary_filename = f"{binary_name}.exe"
        bin_dir = os.path.join(conda_prefix, "Library", "bin")
    else:
        binary_filename = binary_name
        bin_dir = os.path.join(conda_prefix, "bin")

    binary_path = os.path.join(bin_dir, binary_filename)
    if os.path.isfile(binary_path):
        return binary_path
    return None


def _find_binary_in_cuda_home(binary_name: str) -> str | None:
    """Search for a binary in CUDA_HOME or CUDA_PATH."""
    cuda_home = get_cuda_home_or_path()
    if cuda_home is None:
        return None

    if IS_WINDOWS:
        binary_filename = f"{binary_name}.exe"
    else:
        binary_filename = binary_name

    bin_dir = os.path.join(cuda_home, "bin")
    binary_path = os.path.join(bin_dir, binary_filename)
    if os.path.isfile(binary_path):
        return binary_path
    return None


@functools.cache
def find_nvidia_binary(binary_name: str) -> str | None:
    """Locate a CUDA binary executable.

    Args:
        binary_name (str): The name of the binary to find (e.g., ``"nvdisasm"``,
            ``"cuobjdump"``).

    Returns:
        str or None: Absolute path to the discovered binary, or ``None`` if the
        binary cannot be found.

    Raises:
        RuntimeError: If ``binary_name`` is not in the supported set.

    Search order:
        1. **NVIDIA Python wheels**

           - Scan installed distributions (``site-packages``) for binaries
             shipped in NVIDIA wheels (e.g., ``cuda-toolkit[nvcc]``).

        2. **Conda environments**

           - Check Conda-style installation prefixes (``$CONDA_PREFIX/bin`` on
             Linux/Mac or ``$CONDA_PREFIX/Library/bin`` on Windows).

        3. **CUDA Toolkit environment variables**

           - Use ``CUDA_HOME`` or ``CUDA_PATH`` (in that order) and look in the
             ``bin`` subdirectory.
    """
    if binary_name not in SUPPORTED_BINARIES:
        raise RuntimeError(f"UNKNOWN {binary_name=}")

    # Try site-packages first
    if binary_path := _find_binary_under_site_packages(binary_name):
        return _abs_norm(binary_path)

    # Try conda prefix
    if binary_path := _find_binary_in_conda(binary_name):
        return _abs_norm(binary_path)

    # Try CUDA_HOME/CUDA_PATH
    if binary_path := _find_binary_in_cuda_home(binary_name):
        return _abs_norm(binary_path)

    return None
