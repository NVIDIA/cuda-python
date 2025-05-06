# Copyright 2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import ctypes.wintypes
from typing import Optional

import pywintypes
import win32api

from cuda.bindings._path_finder.load_dl_common import LoadedDL

# Mirrors WinBase.h (unfortunately not defined already elsewhere)
WINBASE_LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100
WINBASE_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000


def add_dll_directory(dll_abs_path: str) -> None:
    """Add a DLL directory to the search path and update PATH environment variable.

    Args:
        dll_abs_path: Absolute path to the DLL file

    Raises:
        AssertionError: If the directory containing the DLL does not exist
    """
    import os

    dirpath = os.path.dirname(dll_abs_path)
    assert os.path.isdir(dirpath), dll_abs_path
    # Add the DLL directory to the search path
    os.add_dll_directory(dirpath)
    # Update PATH as a fallback for dependent DLL resolution
    curr_path = os.environ.get("PATH")
    os.environ["PATH"] = dirpath if curr_path is None else os.pathsep.join((curr_path, dirpath))


def abs_path_for_dynamic_library(libname: str, handle: pywintypes.HANDLE) -> str:
    """Get the absolute path of a loaded dynamic library on Windows.

    Args:
        handle: The library handle

    Returns:
        The absolute path to the DLL file

    Raises:
        OSError: If GetModuleFileNameW fails
        RuntimeError: If the required path length is unreasonably long
    """
    MAX_ITERATIONS = 10  # Allows for extremely long paths (up to ~266,000 chars)
    buf_size = 260  # Start with traditional MAX_PATH

    for _ in range(MAX_ITERATIONS):
        buf = ctypes.create_unicode_buffer(buf_size)
        n_chars = ctypes.windll.kernel32.GetModuleFileNameW(ctypes.wintypes.HMODULE(handle), buf, buf_size)

        if n_chars == 0:
            raise OSError(
                f"GetModuleFileNameW failed ({libname=!r}, {buf_size=}). "
                "Long paths may require enabling the "
                "Windows 10+ long path registry setting. See: "
                "https://docs.python.org/3/using/windows.html#removing-the-max-path-limitation"
            )
        if n_chars < buf_size - 1:
            return buf.value

        buf_size *= 2  # Double the buffer size and try again

    raise RuntimeError(
        f"Failed to retrieve the full path after {MAX_ITERATIONS} attempts "
        f"(final buffer size: {buf_size} characters). "
        "This may indicate:\n"
        "  1. An extremely long path requiring Windows long path support, or\n"
        "  2. An invalid or corrupt library handle, or\n"
        "  3. An unexpected system error.\n"
        "See: https://docs.python.org/3/using/windows.html#removing-the-max-path-limitation"
    )


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
    from cuda.bindings._path_finder.supported_libs import SUPPORTED_WINDOWS_DLLS

    for dll_name in SUPPORTED_WINDOWS_DLLS.get(libname, ()):
        try:
            handle = win32api.GetModuleHandle(dll_name)
        except pywintypes.error:
            continue
        else:
            return LoadedDL(handle, abs_path_for_dynamic_library(libname, handle), True)
    return None


def load_with_system_search(libname: str, _unused: str) -> Optional[LoadedDL]:
    """Try to load a DLL using system search paths.

    Args:
        libname: The name of the library to load
        _unused: Unused parameter (kept for interface consistency)

    Returns:
        A LoadedDL object if successful, None if the library cannot be loaded
    """
    from cuda.bindings._path_finder.supported_libs import SUPPORTED_WINDOWS_DLLS

    for dll_name in SUPPORTED_WINDOWS_DLLS.get(libname, ()):
        try:
            handle = win32api.LoadLibraryEx(dll_name, 0, 0)
        except pywintypes.error:
            continue
        else:
            return LoadedDL(handle, abs_path_for_dynamic_library(libname, handle), False)

    return None


def load_with_abs_path(libname: str, found_path: str) -> LoadedDL:
    """Load a dynamic library from the given path.

    Args:
        libname: The name of the library to load
        found_path: The absolute path to the DLL file

    Returns:
        A LoadedDL object representing the loaded library

    Raises:
        RuntimeError: If the DLL cannot be loaded
    """
    from cuda.bindings._path_finder.supported_libs import LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY

    if libname in LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY:
        add_dll_directory(found_path)

    flags = WINBASE_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | WINBASE_LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
    try:
        handle = win32api.LoadLibraryEx(found_path, 0, flags)
    except pywintypes.error as e:
        raise RuntimeError(f"Failed to load DLL at {found_path}: {e}") from e
    return LoadedDL(handle, found_path, False)
