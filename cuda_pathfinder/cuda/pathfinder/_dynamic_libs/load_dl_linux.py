# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import ctypes
import ctypes.util
import os
from typing import Optional

from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import SUPPORTED_LINUX_SONAMES

CDLL_MODE = os.RTLD_NOW | os.RTLD_GLOBAL

LIBDL_PATH = ctypes.util.find_library("dl") or "libdl.so.2"
LIBDL = ctypes.CDLL(LIBDL_PATH)
LIBDL.dladdr.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
LIBDL.dladdr.restype = ctypes.c_int


class DlInfo(ctypes.Structure):
    """Structure used by dladdr to return information about a loaded symbol."""

    _fields_ = (
        ("dli_fname", ctypes.c_char_p),  # path to .so
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    )


def abs_path_for_dynamic_library(libname: str, handle: ctypes.CDLL) -> Optional[str]:
    """Get the absolute path of a loaded dynamic library on Linux.

    Args:
        libname: The name of the library
        handle: The library handle

    Returns:
        The absolute path to the library file, or None if no expected symbol is found

    Raises:
        OSError: If dladdr fails to get information about the symbol
    """
    from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import EXPECTED_LIB_SYMBOLS

    for symbol_name in EXPECTED_LIB_SYMBOLS[libname]:
        symbol = getattr(handle, symbol_name, None)
        if symbol is not None:
            break
    else:
        return None

    addr = ctypes.cast(symbol, ctypes.c_void_p)
    info = DlInfo()
    if LIBDL.dladdr(addr, ctypes.byref(info)) == 0:
        raise OSError(f"dladdr failed for {libname=!r}")
    return info.dli_fname.decode()  # type: ignore[no-any-return]


def check_if_already_loaded_from_elsewhere(libname: str) -> Optional[LoadedDL]:
    """Check if the library is already loaded in the process.

    Args:
        libname: The name of the library to check

    Returns:
        A LoadedDL object if the library is already loaded, None otherwise

    Example:
        >>> loaded = check_if_already_loaded_from_elsewhere("cudart")
        >>> if loaded is not None:
        ...     print(f"Library already loaded from {loaded.abs_path}")
    """
    from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import SUPPORTED_LINUX_SONAMES

    for soname in SUPPORTED_LINUX_SONAMES.get(libname, ()):
        try:
            handle = ctypes.CDLL(soname, mode=os.RTLD_NOLOAD)
        except OSError:
            continue
        else:
            return LoadedDL(abs_path_for_dynamic_library(libname, handle), True, handle._handle)
    return None


def load_with_system_search(libname: str) -> Optional[LoadedDL]:
    """Try to load a library using system search paths.

    Args:
        libname: The name of the library to load

    Returns:
        A LoadedDL object if successful, None if the library cannot be loaded

    Raises:
        RuntimeError: If the library is loaded but no expected symbol is found
    """
    candidate_sonames = list(SUPPORTED_LINUX_SONAMES.get(libname, ()))
    candidate_sonames.append(f"lib{libname}.so")
    for soname in candidate_sonames:
        try:
            handle = ctypes.CDLL(soname, CDLL_MODE)
            abs_path = abs_path_for_dynamic_library(libname, handle)
            if abs_path is None:
                raise RuntimeError(f"No expected symbol for {libname=!r}")
            return LoadedDL(abs_path, False, handle._handle)
        except OSError:
            pass
    return None


def _work_around_known_bugs(libname: str, found_path: str) -> None:
    if libname == "nvrtc":
        # Work around bug/oversight in
        #   nvidia_cuda_nvrtc-13.0.48-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl
        # Issue: libnvrtc.so.13 RUNPATH is not set.
        # This workaround is highly specific
        #   - for simplicity.
        #   - to not mask bugs in future nvidia-cuda-nvrtc releases.
        #   - because a more general workaround is complicated.
        dirname, basename = os.path.split(found_path)
        if basename == "libnvrtc.so.13":
            dep_basename = "libnvrtc-builtins.so.13.0"
            dep_path = os.path.join(dirname, dep_basename)
            if os.path.isfile(dep_path):
                # In case of failure, defer to primary load, which is almost certain to fail, too.
                with contextlib.suppress(OSError):
                    ctypes.CDLL(dep_path, CDLL_MODE)


def load_with_abs_path(libname: str, found_path: str) -> LoadedDL:
    """Load a dynamic library from the given path.

    Args:
        libname: The name of the library to load
        found_path: The absolute path to the library file

    Returns:
        A LoadedDL object representing the loaded library

    Raises:
        RuntimeError: If the library cannot be loaded
    """
    _work_around_known_bugs(libname, found_path)
    try:
        handle = ctypes.CDLL(found_path, CDLL_MODE)
    except OSError as e:
        raise RuntimeError(f"Failed to dlopen {found_path}: {e}") from e
    return LoadedDL(found_path, False, handle._handle)
