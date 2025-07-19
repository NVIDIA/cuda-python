# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import ctypes.wintypes
import os
import struct
from typing import Optional

from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL

# Mirrors WinBase.h (unfortunately not defined already elsewhere)
WINBASE_LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100
WINBASE_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000

POINTER_ADDRESS_SPACE = 2 ** (struct.calcsize("P") * 8)

# Set up kernel32 functions with proper types
kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

# GetModuleHandleW
kernel32.GetModuleHandleW.argtypes = [ctypes.wintypes.LPCWSTR]
kernel32.GetModuleHandleW.restype = ctypes.wintypes.HMODULE

# LoadLibraryExW
kernel32.LoadLibraryExW.argtypes = [
    ctypes.wintypes.LPCWSTR,  # lpLibFileName
    ctypes.wintypes.HANDLE,  # hFile (reserved, must be NULL)
    ctypes.wintypes.DWORD,  # dwFlags
]
kernel32.LoadLibraryExW.restype = ctypes.wintypes.HMODULE

# GetModuleFileNameW
kernel32.GetModuleFileNameW.argtypes = [
    ctypes.wintypes.HMODULE,  # hModule
    ctypes.wintypes.LPWSTR,  # lpFilename
    ctypes.wintypes.DWORD,  # nSize
]
kernel32.GetModuleFileNameW.restype = ctypes.wintypes.DWORD

# AddDllDirectory (Windows 7+)
kernel32.AddDllDirectory.argtypes = [ctypes.wintypes.LPCWSTR]
kernel32.AddDllDirectory.restype = ctypes.c_void_p  # DLL_DIRECTORY_COOKIE


def ctypes_handle_to_unsigned_int(handle: ctypes.wintypes.HMODULE) -> int:
    """Convert ctypes HMODULE to unsigned int."""
    handle_uint = int(handle)
    if handle_uint < 0:
        # Convert from signed to unsigned representation
        handle_uint += POINTER_ADDRESS_SPACE
    return handle_uint


def add_dll_directory(dll_abs_path: str) -> None:
    """Add a DLL directory to the search path and update PATH environment variable.

    Args:
        dll_abs_path: Absolute path to the DLL file

    Raises:
        AssertionError: If the directory containing the DLL does not exist
    """
    dirpath = os.path.dirname(dll_abs_path)
    assert os.path.isdir(dirpath), dll_abs_path

    # Add the DLL directory to the search path
    result = kernel32.AddDllDirectory(dirpath)
    if not result:
        # Fallback: just update PATH if AddDllDirectory fails
        pass

    # Update PATH as a fallback for dependent DLL resolution
    curr_path = os.environ.get("PATH")
    os.environ["PATH"] = dirpath if curr_path is None else os.pathsep.join((curr_path, dirpath))


def abs_path_for_dynamic_library(libname: str, handle: ctypes.wintypes.HMODULE) -> str:
    """Get the absolute path of a loaded dynamic library on Windows."""
    # Create buffer for the path
    buffer = ctypes.create_unicode_buffer(260)  # MAX_PATH
    length = kernel32.GetModuleFileNameW(handle, buffer, len(buffer))

    if length == 0:
        error_code = ctypes.GetLastError()  # type: ignore[attr-defined]
        raise RuntimeError(f"GetModuleFileNameW failed for {libname!r} (error code: {error_code})")

    # If buffer was too small, try with larger buffer
    if length == len(buffer):
        buffer = ctypes.create_unicode_buffer(32768)  # Extended path length
        length = kernel32.GetModuleFileNameW(handle, buffer, len(buffer))
        if length == 0:
            error_code = ctypes.GetLastError()  # type: ignore[attr-defined]
            raise RuntimeError(f"GetModuleFileNameW failed for {libname!r} (error code: {error_code})")

    return buffer.value


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
    from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import SUPPORTED_WINDOWS_DLLS

    for dll_name in SUPPORTED_WINDOWS_DLLS.get(libname, ()):
        handle = kernel32.GetModuleHandleW(dll_name)
        if handle:
            abs_path = abs_path_for_dynamic_library(libname, handle)
            return LoadedDL(abs_path, True, ctypes_handle_to_unsigned_int(handle))
    return None


def load_with_system_search(libname: str) -> Optional[LoadedDL]:
    """Try to load a DLL using system search paths.

    Args:
        libname: The name of the library to load

    Returns:
        A LoadedDL object if successful, None if the library cannot be loaded
    """
    from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import SUPPORTED_WINDOWS_DLLS

    for dll_name in SUPPORTED_WINDOWS_DLLS.get(libname, ()):
        handle = kernel32.LoadLibraryExW(dll_name, None, 0)
        if handle:
            abs_path = abs_path_for_dynamic_library(libname, handle)
            return LoadedDL(abs_path, False, ctypes_handle_to_unsigned_int(handle))

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
    from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import (
        LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY,
    )

    if libname in LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY:
        add_dll_directory(found_path)

    flags = WINBASE_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | WINBASE_LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
    handle = kernel32.LoadLibraryExW(found_path, None, flags)

    if not handle:
        error_code = ctypes.GetLastError()  # type: ignore[attr-defined]
        raise RuntimeError(f"Failed to load DLL at {found_path}: Windows error {error_code}")

    return LoadedDL(found_path, False, ctypes_handle_to_unsigned_int(handle))
