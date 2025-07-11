# Copyright 2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import ctypes.util
import os
from typing import Optional

from cuda.bindings._path_finder.load_dl_common import LoadedDL

CDLL_MODE = os.RTLD_NOW | os.RTLD_GLOBAL

LIBDL_PATH = ctypes.util.find_library("dl") or "libdl.so.2"
LIBDL = ctypes.CDLL(LIBDL_PATH)
LIBDL.dladdr.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
LIBDL.dladdr.restype = ctypes.c_int


class Dl_info(ctypes.Structure):
    """Structure used by dladdr to return information about a loaded symbol."""

    _fields_ = [
        ("dli_fname", ctypes.c_char_p),  # path to .so
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


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
    from cuda.bindings._path_finder.supported_libs import EXPECTED_LIB_SYMBOLS

    for symbol_name in EXPECTED_LIB_SYMBOLS[libname]:
        symbol = getattr(handle, symbol_name, None)
        if symbol is not None:
            break
    else:
        return None

    addr = ctypes.cast(symbol, ctypes.c_void_p)
    info = Dl_info()
    if LIBDL.dladdr(addr, ctypes.byref(info)) == 0:
        raise OSError(f"dladdr failed for {libname=!r}")
    return info.dli_fname.decode()


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
    from cuda.bindings._path_finder.supported_libs import SUPPORTED_LINUX_SONAMES

    for soname in SUPPORTED_LINUX_SONAMES.get(libname, ()):
        try:
            handle = ctypes.CDLL(soname, mode=os.RTLD_NOLOAD)
        except OSError:
            continue
        else:
            return LoadedDL(handle._handle, abs_path_for_dynamic_library(libname, handle), True)
    return None


def load_with_system_search(libname: str, soname: str) -> Optional[LoadedDL]:
    """Try to load a library using system search paths.

    Args:
        libname: The name of the library to load
        soname: The soname to search for

    Returns:
        A LoadedDL object if successful, None if the library cannot be loaded

    Raises:
        RuntimeError: If the library is loaded but no expected symbol is found
    """
    try:
        handle = ctypes.CDLL(soname, CDLL_MODE)
        abs_path = abs_path_for_dynamic_library(libname, handle)
        if abs_path is None:
            raise RuntimeError(f"No expected symbol for {libname=!r}")
        return LoadedDL(handle._handle, abs_path, False)
    except OSError:
        return None


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
    try:
        handle = ctypes.CDLL(found_path, CDLL_MODE)
    except OSError as e:
        raise RuntimeError(f"Failed to dlopen {found_path}: {e}") from e
    return LoadedDL(handle._handle, found_path, False)
