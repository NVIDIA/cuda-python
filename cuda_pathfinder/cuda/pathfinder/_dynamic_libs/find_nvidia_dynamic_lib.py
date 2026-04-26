# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate an NVIDIA dynamic library on disk without loading it in this process.

Resolution is delegated to ``load_nvidia_dynamic_lib`` running in a fresh
Python subprocess. The full loader runs (including ``dlopen`` /
``LoadLibraryExW``) but only inside the child, so the caller's process is left
untouched.
"""

from __future__ import annotations

import functools

from cuda.pathfinder._dynamic_libs import load_nvidia_dynamic_lib as _load_module
from cuda.pathfinder._dynamic_libs.load_dl_common import (
    DynamicLibNotAvailableError,
    DynamicLibNotFoundError,
    DynamicLibUnknownError,
)
from cuda.pathfinder._dynamic_libs.subprocess_protocol import (
    MODE_FIND,
    STATUS_OK,
    run_dynamic_lib_subprocess,
)
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

# The subprocess runs the full loader (site-packages / conda / CUDA_PATH /
# canary cascade), which can be substantially slower than a single canary
# probe. Bound it so a wedged child cannot hang the caller indefinitely.
_FIND_SUBPROCESS_TIMEOUT_SECONDS = 120.0 if IS_WINDOWS else 30.0


@functools.cache
def find_nvidia_dynamic_lib(libname: str) -> str:
    """Return the absolute path to an NVIDIA dynamic library without loading it.

    Resolution is performed by running :func:`load_nvidia_dynamic_lib` in a
    fresh Python subprocess and reporting back the resolved absolute path.
    The caller's process does **not** dlopen / LoadLibrary the library.

    Args:
        libname: Short name of the library (e.g., ``"cufile"``,
            ``"nvJitLink"``, ``"cudart"``).

    Returns:
        The absolute path the loader would have used in the caller's process.

    Raises:
        DynamicLibUnknownError: If ``libname`` is not a recognized library.
        DynamicLibNotAvailableError: If ``libname`` is recognized but not
            supported on this platform.
        DynamicLibNotFoundError: If the library cannot be located.

    Notes:
        Because resolution happens in a separate process, results may differ
        from an in-process ``load_nvidia_dynamic_lib`` if the caller's process
        has DSOs loaded with custom ``RPATH``s or has already loaded a matching
        library by some other mechanism. The intent is to report the path the
        loader would pick when not influenced by other DSOs in the caller.
    """
    # Indirect attribute access (not `from ... import`) so tests can
    # monkeypatch the source-of-truth tables in `load_nvidia_dynamic_lib`.
    if libname not in _load_module._ALL_KNOWN_LIBNAMES:
        raise DynamicLibUnknownError(
            f"Unknown library name: {libname!r}. Known names: {sorted(_load_module._ALL_KNOWN_LIBNAMES)}"
        )
    if libname not in _load_module._ALL_SUPPORTED_LIBNAMES:
        raise DynamicLibNotAvailableError(
            f"Library name {libname!r} is known but not available on {_load_module._PLATFORM_NAME}. "
            f"Supported names on {_load_module._PLATFORM_NAME}: {sorted(_load_module._ALL_SUPPORTED_LIBNAMES)}"
        )

    payload = run_dynamic_lib_subprocess(
        MODE_FIND,
        libname,
        timeout=_FIND_SUBPROCESS_TIMEOUT_SECONDS,
        error_label=f"find_nvidia_dynamic_lib subprocess for {libname!r}",
    )
    if payload.status == STATUS_OK:
        abs_path: str | None = payload.abs_path
        assert abs_path is not None
        return abs_path

    error = payload.error
    if error is not None and "message" in error:
        message = error["message"]
    else:
        message = f"find_nvidia_dynamic_lib could not locate {libname!r}"
    raise DynamicLibNotFoundError(message)
