# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Search shared by the static-library and bitcode-library finders."""

import os
from pathlib import Path

from cuda.pathfinder._utils.env_vars import get_cuda_path_or_home
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.found_via import FOUND_VIA_CONDA, FOUND_VIA_CUDA_PATH, FOUND_VIA_SITE_PACKAGES
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

from .._utils.resolve_path import resolve_path


def _resolved_if_file(directory: Path | str, filename: str) -> str | None:
    file_path = Path(directory, filename)
    if file_path.is_file():
        return str(resolve_path(file_path))
    return None


def locate_file(
    filename: str,
    *,
    site_packages_dirs: tuple[str, ...],
    conda_rel_paths: tuple[str, ...],
    ctk_rel_paths: tuple[str, ...],
) -> tuple[str, str] | None:
    """Return ``(abs_path, found_via)`` for the first place that has *filename*, else ``None``.

    Search order: NVIDIA wheels (``site-packages``), ``$CONDA_PREFIX``, then
    ``$CUDA_PATH`` / ``$CUDA_HOME``. Directories are tried first-to-last.
    """
    for rel_dir in site_packages_dirs:
        for abs_dir in find_sub_dirs_all_sitepackages(tuple(rel_dir.split("/"))):
            abs_path = _resolved_if_file(abs_dir, filename)
            if abs_path is not None:
                return abs_path, FOUND_VIA_SITE_PACKAGES

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        anchor = Path(conda_prefix, "Library") if IS_WINDOWS else Path(conda_prefix)
        for rel_path in conda_rel_paths:
            abs_path = _resolved_if_file(anchor / rel_path, filename)
            if abs_path is not None:
                return abs_path, FOUND_VIA_CONDA

    cuda_home = get_cuda_path_or_home()
    if cuda_home is not None:
        for rel_path in ctk_rel_paths:
            abs_path = _resolved_if_file(Path(cuda_home, rel_path), filename)
            if abs_path is not None:
                return abs_path, FOUND_VIA_CUDA_PATH

    return None


def not_found_message(filename: str, ctk_rel_paths: tuple[str, ...]) -> str:
    """Explain a :func:`locate_file` miss by describing the first ``CUDA_PATH`` directory it tried."""
    cuda_home = get_cuda_path_or_home()
    if cuda_home is None:
        return f'Failure finding "{filename}": CUDA_HOME/CUDA_PATH not set\n'

    directory = Path(cuda_home, ctk_rel_paths[0])
    if directory.is_dir():
        listing = [f'  listdir("{directory}"):']
        listing.extend(f"    {node}" for node in sorted(node_path.name for node_path in directory.iterdir()))
    else:
        listing = [f'  Directory does not exist: "{directory}"']
    return f'Failure finding "{filename}": No such file: {directory / filename}\n' + "\n".join(listing)
