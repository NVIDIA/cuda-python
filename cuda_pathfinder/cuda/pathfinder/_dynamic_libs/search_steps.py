# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composable search steps for locating NVIDIA libraries.

Each find step is a callable with signature::

    (SearchContext) -> FindResult | None

Find steps locate a library file on disk without loading it.  The
orchestrator in :mod:`load_nvidia_dynamic_lib` handles loading, the
already-loaded check, and dependency resolution.

Step sequences are defined per search strategy so that adding a new
step or strategy only requires adding a function and a tuple entry.
"""

import glob
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from cuda.pathfinder._dynamic_libs.lib_descriptor import LibDescriptor
from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import is_suppressed_dll_file
from cuda.pathfinder._utils.env_vars import get_cuda_home_or_path
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FindResult:
    """A library file located on disk (not yet loaded)."""

    abs_path: str
    found_via: str


@dataclass
class SearchContext:
    """Mutable state accumulated during the search cascade."""

    desc: LibDescriptor
    error_messages: list[str] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)

    @property
    def libname(self) -> str:
        return self.desc.name  # type: ignore[no-any-return]  # mypy can't resolve new sibling module

    @property
    def lib_searched_for(self) -> str:
        if IS_WINDOWS:
            return f"{self.libname}*.dll"
        return f"lib{self.libname}.so"

    def raise_not_found(self) -> None:
        err = ", ".join(self.error_messages)
        att = "\n".join(self.attachments)
        raise DynamicLibNotFoundError(f'Failure finding "{self.lib_searched_for}": {err}\n{att}')


#: Type alias for a find step callable.
FindStep = Callable[[SearchContext], FindResult | None]


# ---------------------------------------------------------------------------
# Shared filesystem helpers
# ---------------------------------------------------------------------------


def _no_such_file_in_sub_dirs(
    sub_dirs: Sequence[str], file_wild: str, error_messages: list[str], attachments: list[str]
) -> None:
    error_messages.append(f"No such file: {file_wild}")
    for sub_dir in find_sub_dirs_all_sitepackages(sub_dirs):
        attachments.append(f'  listdir("{sub_dir}"):')
        for node in sorted(os.listdir(sub_dir)):
            attachments.append(f"    {node}")


def _find_dll_under_dir(dirpath: str, file_wild: str) -> str | None:
    for path in sorted(glob.glob(os.path.join(dirpath, file_wild))):
        if not os.path.isfile(path):
            continue
        if not is_suppressed_dll_file(os.path.basename(path)):
            return path
    return None


def _find_so_in_rel_dirs(
    rel_dirs: tuple[str, ...],
    so_basename: str,
    error_messages: list[str],
    attachments: list[str],
) -> str | None:
    sub_dirs_searched: list[tuple[str, ...]] = []
    file_wild = so_basename + "*"
    for rel_dir in rel_dirs:
        sub_dir = tuple(rel_dir.split(os.path.sep))
        for abs_dir in find_sub_dirs_all_sitepackages(sub_dir):
            so_name = os.path.join(abs_dir, so_basename)
            if os.path.isfile(so_name):
                return so_name
            for so_name in sorted(glob.glob(os.path.join(abs_dir, file_wild))):
                if os.path.isfile(so_name):
                    return so_name
        sub_dirs_searched.append(sub_dir)
    for sub_dir in sub_dirs_searched:
        _no_such_file_in_sub_dirs(sub_dir, file_wild, error_messages, attachments)
    return None


def _find_dll_in_rel_dirs(
    rel_dirs: tuple[str, ...],
    lib_searched_for: str,
    error_messages: list[str],
    attachments: list[str],
) -> str | None:
    sub_dirs_searched: list[tuple[str, ...]] = []
    for rel_dir in rel_dirs:
        sub_dir = tuple(rel_dir.split(os.path.sep))
        for abs_dir in find_sub_dirs_all_sitepackages(sub_dir):
            dll_name = _find_dll_under_dir(abs_dir, lib_searched_for)
            if dll_name is not None:
                return dll_name
        sub_dirs_searched.append(sub_dir)
    for sub_dir in sub_dirs_searched:
        _no_such_file_in_sub_dirs(sub_dir, lib_searched_for, error_messages, attachments)
    return None


def _find_in_lib_dir_so(
    lib_dir: str, so_basename: str, error_messages: list[str], attachments: list[str]
) -> str | None:
    so_name = os.path.join(lib_dir, so_basename)
    if os.path.isfile(so_name):
        return so_name
    error_messages.append(f"No such file: {so_name}")
    attachments.append(f'  listdir("{lib_dir}"):')
    if not os.path.isdir(lib_dir):
        attachments.append("    DIRECTORY DOES NOT EXIST")
    else:
        for node in sorted(os.listdir(lib_dir)):
            attachments.append(f"    {node}")
    return None


def _find_in_lib_dir_dll(lib_dir: str, libname: str, error_messages: list[str], attachments: list[str]) -> str | None:
    file_wild = libname + "*.dll"
    dll_name = _find_dll_under_dir(lib_dir, file_wild)
    if dll_name is not None:
        return dll_name
    error_messages.append(f"No such file: {file_wild}")
    attachments.append(f'  listdir("{lib_dir}"):')
    for node in sorted(os.listdir(lib_dir)):
        attachments.append(f"    {node}")
    return None


def _find_lib_dir_using_anchor_point(libname: str, anchor_point: str, linux_lib_dir: str) -> str | None:
    if IS_WINDOWS:
        if libname == "nvvm":  # noqa: SIM108
            rel_paths = ["nvvm/bin/*", "nvvm/bin"]
        else:
            rel_paths = ["bin/x64", "bin"]
    else:
        if libname == "nvvm":  # noqa: SIM108
            rel_paths = ["nvvm/lib64"]
        else:
            rel_paths = [linux_lib_dir]

    for rel_path in rel_paths:
        for dirname in sorted(glob.glob(os.path.join(anchor_point, rel_path))):
            if os.path.isdir(dirname):
                return dirname
    return None


def _find_using_lib_dir(ctx: SearchContext, lib_dir: str | None) -> str | None:
    """Find a library file in a resolved lib directory."""
    if lib_dir is None:
        return None
    if IS_WINDOWS:
        return _find_in_lib_dir_dll(lib_dir, ctx.libname, ctx.error_messages, ctx.attachments)
    return _find_in_lib_dir_so(lib_dir, ctx.lib_searched_for, ctx.error_messages, ctx.attachments)


# ---------------------------------------------------------------------------
# Find steps
# ---------------------------------------------------------------------------


def find_in_site_packages(ctx: SearchContext) -> FindResult | None:
    """Search pip wheel install locations."""
    rel_dirs = ctx.desc.site_packages_dirs
    if not rel_dirs:
        return None
    if IS_WINDOWS:
        abs_path = _find_dll_in_rel_dirs(rel_dirs, ctx.lib_searched_for, ctx.error_messages, ctx.attachments)
    else:
        abs_path = _find_so_in_rel_dirs(rel_dirs, ctx.lib_searched_for, ctx.error_messages, ctx.attachments)
    if abs_path is not None:
        return FindResult(abs_path, "site-packages")
    return None


def find_in_conda(ctx: SearchContext) -> FindResult | None:
    """Search ``$CONDA_PREFIX``."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None
    anchor = os.path.join(conda_prefix, "Library") if IS_WINDOWS else conda_prefix
    lib_dir = _find_lib_dir_using_anchor_point(ctx.libname, anchor_point=anchor, linux_lib_dir="lib")
    abs_path = _find_using_lib_dir(ctx, lib_dir)
    if abs_path is not None:
        return FindResult(abs_path, "conda")
    return None


def find_in_cuda_home(ctx: SearchContext) -> FindResult | None:
    """Search ``$CUDA_HOME`` / ``$CUDA_PATH``."""
    cuda_home = get_cuda_home_or_path()
    if cuda_home is None:
        return None
    lib_dir = _find_lib_dir_using_anchor_point(ctx.libname, anchor_point=cuda_home, linux_lib_dir="lib64")
    abs_path = _find_using_lib_dir(ctx, lib_dir)
    if abs_path is not None:
        return FindResult(abs_path, "CUDA_HOME")
    return None


# ---------------------------------------------------------------------------
# Step sequences per strategy
# ---------------------------------------------------------------------------

#: Find steps that run before the already-loaded check and system search.
EARLY_FIND_STEPS: tuple[FindStep, ...] = (find_in_site_packages, find_in_conda)

#: Find steps that run after system search fails.
LATE_FIND_STEPS: tuple[FindStep, ...] = (find_in_cuda_home,)


# ---------------------------------------------------------------------------
# Cascade runner
# ---------------------------------------------------------------------------


def run_find_steps(ctx: SearchContext, steps: tuple[FindStep, ...]) -> FindResult | None:
    """Run find steps in order, returning the first hit."""
    for step in steps:
        result = step(ctx)
        if result is not None:
            return result
    return None
