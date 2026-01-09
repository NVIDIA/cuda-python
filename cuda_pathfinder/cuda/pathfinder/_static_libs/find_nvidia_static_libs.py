# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import glob
import os

from cuda.pathfinder._static_libs.supported_nvidia_static_libs import (
    SITE_PACKAGES_STATIC_LIBDIRS,
    SUPPORTED_STATIC_LIBS,
)
from cuda.pathfinder._utils.env_vars import get_cuda_home_or_path
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.path_utils import _abs_norm
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS


def _find_artifact_under_site_packages(artifact_name: str) -> str | None:
    """Search for an artifact in site-packages directories."""
    rel_dirs = SITE_PACKAGES_STATIC_LIBDIRS.get(artifact_name)
    if rel_dirs is None:
        return None

    for rel_dir in rel_dirs:
        for lib_dir in find_sub_dirs_all_sitepackages(tuple(rel_dir.split("/"))):
            artifact_path = os.path.join(lib_dir, artifact_name)
            if os.path.isfile(artifact_path):
                return artifact_path
    return None


def _find_libdevice_in_conda() -> str | None:
    """Search for libdevice.10.bc in conda prefix."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None

    # Check multiple possible locations
    if IS_WINDOWS:
        possible_paths = [
            os.path.join(conda_prefix, "Library", "nvvm", "libdevice", "libdevice.10.bc"),
        ]
    else:
        possible_paths = [
            os.path.join(conda_prefix, "nvvm", "libdevice", "libdevice.10.bc"),
        ]

    for path in possible_paths:
        if os.path.isfile(path):
            return path
    return None


def _find_libdevice_in_cuda_home() -> str | None:
    """Search for libdevice.10.bc in CUDA_HOME or CUDA_PATH."""
    cuda_home = get_cuda_home_or_path()
    if cuda_home is None:
        return None

    libdevice_path = os.path.join(cuda_home, "nvvm", "libdevice", "libdevice.10.bc")
    if os.path.isfile(libdevice_path):
        return libdevice_path
    return None


def _find_libcudadevrt_in_conda() -> str | None:
    """Search for libcudadevrt.a in conda prefix."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None

    if IS_WINDOWS:
        # On Windows, it might be cudadevrt.lib
        possible_paths = [
            os.path.join(conda_prefix, "Library", "lib", "libcudadevrt.a"),
            os.path.join(conda_prefix, "Library", "lib", "x64", "cudadevrt.lib"),
        ]
    else:
        possible_paths = [
            os.path.join(conda_prefix, "lib", "libcudadevrt.a"),
        ]

    for path in possible_paths:
        if os.path.isfile(path):
            return path
    return None


def _find_libcudadevrt_in_cuda_home() -> str | None:
    """Search for libcudadevrt.a in CUDA_HOME or CUDA_PATH."""
    cuda_home = get_cuda_home_or_path()
    if cuda_home is None:
        return None

    if IS_WINDOWS:
        # On Windows, check for cudadevrt.lib in various locations
        possible_paths = [
            os.path.join(cuda_home, "lib", "x64", "cudadevrt.lib"),
            os.path.join(cuda_home, "lib", "cudadevrt.lib"),
        ]
    else:
        # On Linux, check lib64 and lib
        possible_paths = [
            os.path.join(cuda_home, "lib64", "libcudadevrt.a"),
            os.path.join(cuda_home, "lib", "libcudadevrt.a"),
        ]

    for path in possible_paths:
        if os.path.isfile(path):
            return path

    # Also check targets subdirectories (for cross-compilation setups)
    if not IS_WINDOWS:
        targets_pattern = os.path.join(cuda_home, "targets", "*", "lib", "libcudadevrt.a")
        for path in sorted(glob.glob(targets_pattern), reverse=True):
            if os.path.isfile(path):
                return path

    return None


@functools.cache
def find_nvidia_static_lib(artifact_name: str) -> str | None:
    """Locate a CUDA static library or artifact file.

    Args:
        artifact_name (str): The name of the artifact to find (e.g.,
            ``"libdevice.10.bc"``, ``"libcudadevrt.a"``).

    Returns:
        str or None: Absolute path to the discovered artifact, or ``None`` if the
        artifact cannot be found.

    Raises:
        RuntimeError: If ``artifact_name`` is not in the supported set.

    Search order:
        1. **NVIDIA Python wheels**

           - Scan installed distributions (``site-packages``) for artifacts
             shipped in NVIDIA wheels (e.g., ``cuda-toolkit[nvvm]``,
             ``cuda-toolkit[cudart]``).

        2. **Conda environments**

           - Check Conda-style installation prefixes for the specific artifact
             layout.

        3. **CUDA Toolkit environment variables**

           - Use ``CUDA_HOME`` or ``CUDA_PATH`` (in that order) and look in
             standard CUDA Toolkit directory layouts.
    """
    if artifact_name not in SUPPORTED_STATIC_LIBS:
        raise RuntimeError(f"UNKNOWN {artifact_name=}")

    # Try site-packages first
    if artifact_path := _find_artifact_under_site_packages(artifact_name):
        return _abs_norm(artifact_path)

    # Handle specific artifacts with custom search logic
    if artifact_name == "libdevice.10.bc":
        # Try conda prefix
        if artifact_path := _find_libdevice_in_conda():
            return _abs_norm(artifact_path)

        # Try CUDA_HOME/CUDA_PATH
        if artifact_path := _find_libdevice_in_cuda_home():
            return _abs_norm(artifact_path)

    elif artifact_name == "libcudadevrt.a":
        # Try conda prefix
        if artifact_path := _find_libcudadevrt_in_conda():
            return _abs_norm(artifact_path)

        # Try CUDA_HOME/CUDA_PATH
        if artifact_path := _find_libcudadevrt_in_cuda_home():
            return _abs_norm(artifact_path)

    return None
