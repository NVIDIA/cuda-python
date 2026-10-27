# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS
from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import (
    _DRIVER_ONLY_LIBNAMES,
    _resolve_system_loaded_abs_path_in_subprocess,
    _try_ctk_root_canary,
    check_libname,
    raise_driver_lib_not_found,
)
from cuda.pathfinder._dynamic_libs.search_steps import (
    EARLY_FIND_STEPS,
    LATE_FIND_STEPS,
    FindResult,
    SearchContext,
    run_find_steps,
)


@dataclass(frozen=True)
class LocatedDynamicLib:
    """Information about a located dynamic library."""

    name: str
    abs_path: str
    filename: str
    found_via: str


def _find_with_system_search(libname: str) -> FindResult | None:
    # The native loader search can only be observed by loading, so it runs in
    # a fresh subprocess and the caller's process stays untouched.
    abs_path = _resolve_system_loaded_abs_path_in_subprocess(libname)
    if abs_path is None:
        return None
    return FindResult(abs_path, "system-search")


def _locate_lib(libname: str) -> FindResult:
    desc = LIB_DESCRIPTORS[libname]

    if libname in _DRIVER_ONLY_LIBNAMES:
        find = _find_with_system_search(libname)
        if find is None:
            raise_driver_lib_not_found(desc)
        return find

    ctx = SearchContext(desc)
    find = run_find_steps(ctx, EARLY_FIND_STEPS)
    if find is not None:
        return find

    find = _find_with_system_search(libname)
    if find is not None:
        return find

    find = run_find_steps(ctx, LATE_FIND_STEPS)
    if find is not None:
        return find

    if desc.ctk_root_canary_anchor_libnames:
        canary_abs_path = _try_ctk_root_canary(ctx)
        if canary_abs_path is not None:
            return FindResult(canary_abs_path, "system-ctk-root")

    ctx.raise_not_found()


def locate_nvidia_dynamic_lib(libname: str) -> LocatedDynamicLib:
    """Locate an NVIDIA dynamic library by name without loading it.

    Returns the file that :func:`load_nvidia_dynamic_lib` would load in a fresh
    process. Use it where a path is needed but the library must not be loaded
    into the current process, for example to pass a library to a build system.

    Args:
        libname (str): The short name of the library to locate (e.g.,
            ``"cudart"``, ``"nvvm"``, etc.).

    Returns:
        LocatedDynamicLib: The absolute path of the library file and how it
        was found.

    Raises:
        DynamicLibUnknownError: If ``libname`` is not a recognized library name.
        DynamicLibNotAvailableError: If ``libname`` is recognized but not
            supported on this platform.
        DynamicLibNotFoundError: If the library cannot be found.

    Search order:
        The steps of :func:`load_nvidia_dynamic_lib`, in the same order, except
        that a library already loaded in the current process is not taken into
        account:

        1. NVIDIA Python wheels (``site-packages``)
        2. Conda environment (``CONDA_PREFIX``)
        3. OS default mechanisms (``dlopen()`` / ``LoadLibraryExW()``), run in
           a fresh Python subprocess
        4. Environment variables (library-specific roots, then ``CUDA_PATH`` /
           ``CUDA_HOME``)
        5. Windows Program Files (configured libraries only)
        6. CTK root canary probe (discoverable libs only)

        Driver libraries (``"cuda"``, ``"nvml"``) use step 3 only.

    Notes:
        The result can differ from what :func:`load_nvidia_dynamic_lib` returns
        in the current process when another component has already loaded the
        library, or influences the native loader search (for example through
        its ``RPATH``).
    """
    check_libname(libname)
    find = _locate_lib(libname)
    return LocatedDynamicLib(
        name=libname,
        abs_path=find.abs_path,
        filename=os.path.basename(find.abs_path),
        found_via=find.found_via,
    )


@functools.cache
def find_nvidia_dynamic_lib(libname: str) -> str:
    """Find the absolute path of an NVIDIA dynamic library without loading it.

    See :func:`locate_nvidia_dynamic_lib` for the search order.

    Raises:
        DynamicLibUnknownError: If ``libname`` is not a recognized library name.
        DynamicLibNotAvailableError: If ``libname`` is recognized but not
            supported on this platform.
        DynamicLibNotFoundError: If the library cannot be found.
    """
    return locate_nvidia_dynamic_lib(libname).abs_path
