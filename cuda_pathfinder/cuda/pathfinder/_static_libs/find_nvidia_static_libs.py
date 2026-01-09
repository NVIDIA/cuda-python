# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import glob
import os
from typing import Sequence

from cuda.pathfinder._static_libs.artifact_search_config import ARTIFACT_CONFIGS
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


def _search_paths(
    base_dir: str,
    subdirs: Sequence[str],
    filenames: Sequence[str],
) -> str | None:
    """Search for a file in multiple subdirectories of a base directory.

    Args:
        base_dir: The base directory to search in.
        subdirs: Subdirectories to check (relative to base_dir).
        filenames: Filenames to look for in each subdirectory.

    Returns:
        First matching file path, or None if not found.
    """
    for subdir in subdirs:
        dir_path = os.path.join(base_dir, subdir)
        for filename in filenames:
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path):
                return file_path
    return None


def _search_targets_subdirs(
    cuda_home: str,
    filenames: Sequence[str],
) -> str | None:
    """Search in targets/*/lib and targets/*/lib64 subdirectories.

    For cross-compilation setups. Returns the first match, preferring
    more recently modified targets.

    Args:
        cuda_home: The CUDA home directory.
        filenames: Filenames to search for.

    Returns:
        First matching file path, or None if not found.
    """
    for lib_subdir in ("lib64", "lib"):
        pattern = os.path.join(cuda_home, "targets", "*", lib_subdir)
        for dir_path in sorted(glob.glob(pattern), reverse=True):
            for filename in filenames:
                file_path = os.path.join(dir_path, filename)
                if os.path.isfile(file_path):
                    return file_path
    return None


def _find_artifact_in_conda(artifact_name: str) -> str | None:
    """Generic conda prefix search for any configured artifact.

    Args:
        artifact_name: The name of the artifact to find.

    Returns:
        Path to the artifact if found, None otherwise.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None

    config = ARTIFACT_CONFIGS.get(artifact_name)
    if not config:
        return None

    return _search_paths(conda_prefix, config.conda_dirs, config.filenames)


def _find_artifact_in_cuda_home(artifact_name: str) -> str | None:
    """Generic CUDA_HOME/CUDA_PATH search for any configured artifact.

    Args:
        artifact_name: The name of the artifact to find.

    Returns:
        Path to the artifact if found, None otherwise.
    """
    cuda_home = get_cuda_home_or_path()
    if cuda_home is None:
        return None

    config = ARTIFACT_CONFIGS.get(artifact_name)
    if not config:
        return None

    # Try standard directories first
    if result := _search_paths(cuda_home, config.cuda_home_dirs, config.filenames):
        return result

    # Try targets subdirectories for cross-compilation
    if config.search_targets_subdirs and not IS_WINDOWS:
        if result := _search_targets_subdirs(cuda_home, config.filenames):
            return result

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

    # Try conda prefix (generic, configuration-driven)
    if artifact_path := _find_artifact_in_conda(artifact_name):
        return _abs_norm(artifact_path)

    # Try CUDA_HOME/CUDA_PATH (generic, configuration-driven)
    if artifact_path := _find_artifact_in_cuda_home(artifact_name):
        return _abs_norm(artifact_path)

    return None
