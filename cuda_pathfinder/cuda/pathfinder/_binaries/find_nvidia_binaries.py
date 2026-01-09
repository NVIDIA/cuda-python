# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from typing import Sequence

from cuda.pathfinder._binaries.supported_nvidia_binaries import SITE_PACKAGES_BINDIRS, SUPPORTED_BINARIES
from cuda.pathfinder._utils.env_vars import get_cuda_home_or_path
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.path_utils import _abs_norm


def _get_binary_filename_candidates(binary_name: str) -> tuple[str, ...]:
    """Generate possible binary filename variations.

    Returns multiple candidates to support fuzzy search across platforms.
    The filesystem will naturally filter to what exists.

    Args:
        binary_name: Base name of the binary (e.g., "nvdisasm").

    Returns:
        Tuple of possible filenames to try (exact name, with .exe extension).
    """
    # Try exact name first, then with .exe extension
    # This works across platforms - non-existent files simply won't be found
    return (binary_name, f"{binary_name}.exe")


def _find_file_in_dir(directory: str, filename_candidates: Sequence[str]) -> str | None:
    """Search for the first existing file from candidates in a directory.

    Args:
        directory: Directory to search in.
        filename_candidates: Possible filenames to try.

    Returns:
        Path to first matching file, or None if none found.
    """
    for filename in filename_candidates:
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            return file_path
    return None


def _find_binary_under_site_packages(binary_name: str) -> str | None:
    """Search for a binary in site-packages directories."""
    rel_dirs = SITE_PACKAGES_BINDIRS.get(binary_name)
    if rel_dirs is None:
        return None

    filename_candidates = _get_binary_filename_candidates(binary_name)

    for rel_dir in rel_dirs:
        for bin_dir in find_sub_dirs_all_sitepackages(tuple(rel_dir.split("/"))):
            if found := _find_file_in_dir(bin_dir, filename_candidates):
                return found
    return None


def _find_binary_in_conda(binary_name: str) -> str | None:
    """Search for a binary in conda prefix.

    Searches common conda bin directory locations across platforms.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None

    filename_candidates = _get_binary_filename_candidates(binary_name)

    # Try both Windows and Unix bin directory layouts
    # The filesystem will naturally tell us what exists
    bin_dirs = (
        os.path.join(conda_prefix, "Library", "bin"),  # Windows conda layout
        os.path.join(conda_prefix, "bin"),  # Unix conda layout
    )

    for bin_dir in bin_dirs:
        if found := _find_file_in_dir(bin_dir, filename_candidates):
            return found
    return None


def _find_binary_in_cuda_home(binary_name: str) -> str | None:
    """Search for a binary in CUDA_HOME or CUDA_PATH."""
    cuda_home = get_cuda_home_or_path()
    if cuda_home is None:
        return None

    filename_candidates = _get_binary_filename_candidates(binary_name)
    bin_dir = os.path.join(cuda_home, "bin")

    return _find_file_in_dir(bin_dir, filename_candidates)


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
