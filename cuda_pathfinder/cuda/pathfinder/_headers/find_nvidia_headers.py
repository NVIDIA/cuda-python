# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import glob
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import (
    _resolve_system_loaded_abs_path_in_subprocess,
)
from cuda.pathfinder._dynamic_libs.search_steps import derive_ctk_root
from cuda.pathfinder._headers.header_descriptor import (
    HEADER_DESCRIPTORS,
    platform_include_subdirs,
    resolve_conda_anchor,
)
from cuda.pathfinder._utils.env_vars import get_cuda_home_or_path
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages

if TYPE_CHECKING:
    from cuda.pathfinder._headers.header_descriptor import HeaderDescriptor

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class LocatedHeaderDir:
    abs_path: str | None
    found_via: str

    def __post_init__(self) -> None:
        self.abs_path = _abs_norm(self.abs_path)


#: Type alias for a header find step callable.
HeaderFindStep = Callable[["HeaderDescriptor"], LocatedHeaderDir | None]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _abs_norm(path: str | None) -> str | None:
    if path:
        return os.path.normpath(os.path.abspath(path))
    return None


def _joined_isfile(dirpath: str, basename: str) -> bool:
    return os.path.isfile(os.path.join(dirpath, basename))


def _locate_in_anchor_layout(desc: HeaderDescriptor, anchor_point: str) -> str | None:
    """Search for a header under *anchor_point* using the descriptor's layout fields."""
    h_basename = desc.header_basename
    for rel_dir in desc.anchor_include_rel_dirs:
        idir = os.path.join(anchor_point, rel_dir)
        for subdir in platform_include_subdirs(desc):
            cdir = os.path.join(idir, subdir)
            if _joined_isfile(cdir, h_basename):
                return cdir
        if _joined_isfile(idir, h_basename):
            return idir
    return None


# ---------------------------------------------------------------------------
# Find steps
# ---------------------------------------------------------------------------


def find_in_site_packages(desc: HeaderDescriptor) -> LocatedHeaderDir | None:
    """Search pip wheel install locations."""
    for sub_dir in desc.site_packages_dirs:
        hdr_dir: str  # help mypy
        for hdr_dir in find_sub_dirs_all_sitepackages(tuple(sub_dir.split("/"))):
            if _joined_isfile(hdr_dir, desc.header_basename):
                return LocatedHeaderDir(abs_path=hdr_dir, found_via="site-packages")
    return None


def find_in_conda(desc: HeaderDescriptor) -> LocatedHeaderDir | None:
    """Search ``$CONDA_PREFIX``."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None
    anchor_point = resolve_conda_anchor(desc, conda_prefix)
    if anchor_point is None:
        return None
    found_header_path = _locate_in_anchor_layout(desc, anchor_point)
    if found_header_path:
        return LocatedHeaderDir(abs_path=found_header_path, found_via="conda")
    return None


def find_in_cuda_home(desc: HeaderDescriptor) -> LocatedHeaderDir | None:
    """Search ``$CUDA_HOME`` / ``$CUDA_PATH``."""
    cuda_home = get_cuda_home_or_path()
    if cuda_home is None:
        return None
    result = _locate_in_anchor_layout(desc, cuda_home)
    if result is not None:
        return LocatedHeaderDir(abs_path=result, found_via="CUDA_HOME")
    return None


def find_via_ctk_root_canary(desc: HeaderDescriptor) -> LocatedHeaderDir | None:
    """Try CTK header lookup via CTK-root canary probing.

    Skips immediately if the descriptor does not opt in (``use_ctk_root_canary``).
    Otherwise, system-loads ``cudart`` in a fully isolated Python subprocess, derives
    CTK root from the resolved library path, and searches the expected include
    layout under that root.
    """
    if not desc.use_ctk_root_canary:
        return None
    canary_abs_path = _resolve_system_loaded_abs_path_in_subprocess("cudart")
    if canary_abs_path is None:
        return None
    ctk_root = derive_ctk_root(canary_abs_path)
    if ctk_root is None:
        return None
    result = _locate_in_anchor_layout(desc, ctk_root)
    if result is not None:
        return LocatedHeaderDir(abs_path=result, found_via="system-ctk-root")
    return None


def find_in_system_install_dirs(desc: HeaderDescriptor) -> LocatedHeaderDir | None:
    """Search system install directories (glob patterns)."""
    for pattern in desc.system_install_dirs:
        for hdr_dir in sorted(glob.glob(pattern), reverse=True):
            if _joined_isfile(hdr_dir, desc.header_basename):
                return LocatedHeaderDir(abs_path=hdr_dir, found_via="supported_install_dir")
    return None


# ---------------------------------------------------------------------------
# Step sequence and cascade runner
# ---------------------------------------------------------------------------

#: Unified find steps — each step self-gates based on descriptor fields.
FIND_STEPS: tuple[HeaderFindStep, ...] = (
    find_in_site_packages,
    find_in_conda,
    find_in_cuda_home,
    find_via_ctk_root_canary,
    find_in_system_install_dirs,
)


def run_find_steps(desc: HeaderDescriptor, steps: tuple[HeaderFindStep, ...]) -> LocatedHeaderDir | None:
    """Run find steps in order, returning the first hit."""
    for step in steps:
        result = step(desc)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@functools.cache
def locate_nvidia_header_directory(libname: str) -> LocatedHeaderDir | None:
    """Locate the header directory for a supported NVIDIA library.

    Args:
        libname (str): The short name of the library whose headers are needed
            (e.g., ``"nvrtc"``, ``"cusolver"``, ``"nvshmem"``).

    Returns:
        LocatedHeaderDir or None: A LocatedHeaderDir object containing the absolute path
        to the discovered header directory and information about where it was found,
        or ``None`` if the headers cannot be found.

    Raises:
        RuntimeError: If ``libname`` is not in the supported set.

    Search order:
        1. **NVIDIA Python wheels** — site-packages directories from the descriptor.
        2. **Conda environments** — platform-specific conda include layouts.
        3. **CUDA Toolkit environment variables** — ``CUDA_HOME`` / ``CUDA_PATH``.
        4. **CTK root canary probe** — subprocess canary (descriptors with
           ``use_ctk_root_canary=True`` only).
        5. **System install directories** — glob patterns from the descriptor.
    """
    desc = HEADER_DESCRIPTORS.get(libname)
    if desc is None:
        raise RuntimeError(f"UNKNOWN {libname=}")
    return run_find_steps(desc, FIND_STEPS)


def find_nvidia_header_directory(libname: str) -> str | None:
    """Locate the header directory for a supported NVIDIA library.

    Args:
        libname (str): The short name of the library whose headers are needed
            (e.g., ``"nvrtc"``, ``"cusolver"``, ``"nvshmem"``).

    Returns:
        str or None: Absolute path to the discovered header directory, or ``None``
        if the headers cannot be found.

    Raises:
        RuntimeError: If ``libname`` is not in the supported set.

    Search order:
        1. **NVIDIA Python wheels** — site-packages directories from the descriptor.
        2. **Conda environments** — platform-specific conda include layouts.
        3. **CUDA Toolkit environment variables** — ``CUDA_HOME`` / ``CUDA_PATH``.
        4. **CTK root canary probe** — subprocess canary (descriptors with
           ``use_ctk_root_canary=True`` only).
        5. **System install directories** — glob patterns from the descriptor.
    """
    found = locate_nvidia_header_directory(libname)
    return found.abs_path if found else None
