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

This module is intentionally platform-agnostic: it does not branch on the
current operating system. Platform differences are resolved up front into the
:class:`~cuda.pathfinder._dynamic_libs.search_platform.SearchContext`.
"""

import glob
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from cuda.pathfinder._dynamic_libs.search_platform import (
    SearchContext,
    find_lib_in_dir,
    find_lib_in_site_packages,
)
from cuda.pathfinder._utils.env_vars import get_cuda_path_or_home
from cuda.pathfinder._utils.path_sort import numeric_aware_path_sort_key

from .._utils.resolve_path import resolve_path

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FindResult:
    """A library file located on disk (not yet loaded)."""

    abs_path: str
    found_via: str


#: Type alias for a find step callable.
FindStep = Callable[[SearchContext], FindResult | None]


def _iter_lib_dirs(root: str, rel_dirs: tuple[str, ...]) -> Iterator[str]:
    """Yield existing library directories under *root* in descriptor order."""
    for rel_path in rel_dirs:
        for dirname in sorted(glob.glob(os.path.join(root, rel_path))):
            if os.path.isdir(dirname):
                yield str(resolve_path(dirname))


def _find_under_root(
    ctx: SearchContext,
    root: str,
    rel_dirs: tuple[str, ...],
    found_via: str,
) -> FindResult | None:
    """Resolve *rel_dirs* under *root*, then find the requested library."""
    for lib_dir in _iter_lib_dirs(root, rel_dirs):
        abs_path = find_lib_in_dir(ctx, lib_dir)
        if abs_path is not None:
            return FindResult(abs_path, found_via)
    return None


def _find_under_anchor_root(ctx: SearchContext, root: str, found_via: str) -> FindResult | None:
    """Resolve the descriptor's general anchors under *root*."""
    return _find_under_root(ctx, root, ctx.anchor_rel_dirs, found_via)


def _derive_ctk_root_linux(resolved_lib_path: str) -> str | None:
    """Derive CTK root from Linux canary path.

    Supports:
    - ``$CTK_ROOT/lib64/libfoo.so.*``
    - ``$CTK_ROOT/lib/libfoo.so.*``
    - ``$CTK_ROOT/targets/<triple>/lib64/libfoo.so.*``
    - ``$CTK_ROOT/targets/<triple>/lib/libfoo.so.*``
    """
    lib_dir = os.path.dirname(resolved_lib_path)
    basename = os.path.basename(lib_dir)
    if basename in ("lib64", "lib"):
        parent = os.path.dirname(lib_dir)
        grandparent = os.path.dirname(parent)
        if os.path.basename(grandparent) == "targets":
            return os.path.dirname(grandparent)
        return parent
    return None


def _derive_ctk_root_windows(resolved_lib_path: str) -> str | None:
    """Derive CTK root from Windows canary path.

    Supports:
    - ``$CTK_ROOT/bin/x64/foo.dll`` (CTK 13 style)
    - ``$CTK_ROOT/bin/arm64/foo.dll`` (Windows on Arm CTK 13 style)
    - ``$CTK_ROOT/bin/foo.dll`` (CTK 12 style)
    """
    import ntpath

    lib_dir = ntpath.dirname(resolved_lib_path)
    basename = ntpath.basename(lib_dir).lower()
    if basename in ("x64", "arm64"):
        parent = ntpath.dirname(lib_dir)
        if ntpath.basename(parent).lower() == "bin":
            return ntpath.dirname(parent)
    elif basename == "bin":
        return ntpath.dirname(lib_dir)
    return None


def derive_ctk_root(resolved_lib_path: str) -> str | None:
    """Derive CTK root from a resolved canary library path."""
    ctk_root = _derive_ctk_root_linux(resolved_lib_path)
    if ctk_root is not None:
        return ctk_root
    return _derive_ctk_root_windows(resolved_lib_path)


def find_via_ctk_root(ctx: SearchContext, ctk_root: str) -> FindResult | None:
    """Find a library under a previously derived CTK root."""
    return _find_under_anchor_root(ctx, ctk_root, "system-ctk-root")


# ---------------------------------------------------------------------------
# Find steps
# ---------------------------------------------------------------------------


def find_in_site_packages(ctx: SearchContext) -> FindResult | None:
    """Search pip wheel install locations."""
    if not ctx.site_packages_rel_dirs:
        return None
    abs_path = find_lib_in_site_packages(ctx)
    if abs_path is not None:
        return FindResult(abs_path, "site-packages")
    return None


def find_in_conda(ctx: SearchContext) -> FindResult | None:
    """Search ``$CONDA_PREFIX``."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None
    anchor = ctx.conda_anchor_point(conda_prefix)
    return _find_under_anchor_root(ctx, anchor, "conda")


def find_in_install_root_env_vars(ctx: SearchContext) -> FindResult | None:
    """Search installation roots named by descriptor-specific environment variables."""
    rel_dirs = ctx.install_root_env_rel_dirs
    for env_var in ctx.install_root_env_vars:
        root = os.environ.get(env_var)
        if not root:
            continue
        result = _find_under_root(ctx, root, rel_dirs, env_var)
        if result is not None:
            return result
    return None


def find_in_cuda_path(ctx: SearchContext) -> FindResult | None:
    """Search ``$CUDA_PATH`` / ``$CUDA_HOME`` as fallback toolkit roots.

    This step runs after native system search on both platforms. On Windows,
    native ``LoadLibraryExW(..., 0)`` search can already discover CTK DLLs
    through ``PATH``, subject to the process's DLL search policy. These roots
    do not override earlier discoveries or pin the runtime toolkit.

    The returned ``found_via`` is always ``"CUDA_PATH"`` regardless of which
    environment variable actually provided the value.
    """
    cuda_home = get_cuda_path_or_home()
    if cuda_home is None:
        return None
    return _find_under_anchor_root(ctx, cuda_home, "CUDA_PATH")


def find_in_program_files_roots(ctx: SearchContext) -> FindResult | None:
    """Search descriptor-configured installation roots under Program Files."""
    for root_glob in ctx.program_files_root_globs():
        for root in sorted(glob.glob(root_glob), key=numeric_aware_path_sort_key, reverse=True):
            if not os.path.isdir(root):
                continue
            result = _find_under_anchor_root(ctx, str(resolve_path(root)), "ProgramFiles")
            if result is not None:
                return result
    return None


# ---------------------------------------------------------------------------
# Step sequences per strategy
# ---------------------------------------------------------------------------

#: Find steps that run after the already-loaded check and before system search.
EARLY_FIND_STEPS: tuple[FindStep, ...] = (find_in_site_packages, find_in_conda)

#: Find steps that run after system search fails.
LATE_FIND_STEPS: tuple[FindStep, ...] = (
    find_in_install_root_env_vars,
    find_in_cuda_path,
    find_in_program_files_roots,
)


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
