# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import ctypes.wintypes
import os
import struct
from typing import Optional

from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import (
    LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY,
    SUPPORTED_WINDOWS_DLLS,
)

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


def check_if_already_loaded_from_elsewhere(libname: str, have_abs_path: bool) -> Optional[LoadedDL]:
    for dll_name in SUPPORTED_WINDOWS_DLLS.get(libname, ()):
        handle = kernel32.GetModuleHandleW(dll_name)
        if handle:
            abs_path = abs_path_for_dynamic_library(libname, handle)
            if have_abs_path and libname in LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY:
                # This is a side-effect if the pathfinder loads the library via
                # load_with_abs_path(). To make the side-effect more deterministic,
                # activate it even if the library was already loaded from elsewhere.
                add_dll_directory(abs_path)
            return LoadedDL(abs_path, True, ctypes_handle_to_unsigned_int(handle))
    return None


def load_with_system_search(libname: str) -> Optional[LoadedDL]:
    """Try to load a DLL using system search paths.

    Args:
        libname: The name of the library to load

    Returns:
        A LoadedDL object if successful, None if the library cannot be loaded
    """
    for dll_name in SUPPORTED_WINDOWS_DLLS.get(libname, ()):
        handle = kernel32.LoadLibraryExW(dll_name, None, 0)
        if handle:
            abs_path = abs_path_for_dynamic_library(libname, handle)
            return LoadedDL(abs_path, False, ctypes_handle_to_unsigned_int(handle))

    return None


def load_with_conda_search(libname: str) -> Optional[LoadedDL]:
    """Try to load a DLL using conda search paths.

    Args:
        libname: The name of the library to load

    Returns:
        A LoadedDL object if successful, None if the library cannot be loaded
    """
    in_conda_build = False
    in_conda_env = False
    if os.getenv("CONDA_BUILD") == "1":
        in_conda_build = True
    elif os.getenv("CONDA_PREFIX"):
        in_conda_env = True
    else:
        return None

    normal_conda_lib_path = os.path.join("Library", "bin", "x64")
    if libname == "nvvm":
        normal_conda_lib_path = os.path.join("Library", "nvvm", "bin", "x64")

    for dll_name in SUPPORTED_WINDOWS_DLLS.get(libname, ()):
        if in_conda_build:
            if prefix := os.getenv("PREFIX"):
                prefix_normal_lib_path = os.path.join(prefix, normal_conda_lib_path)
                if os.path.isdir(prefix_normal_lib_path):
                    dll_name = os.path.join(prefix_normal_lib_path, dll_name)
                    handle = kernel32.LoadLibraryExW(dll_name, None, 0)
                    if handle:
                        # TODO KEITH: Do we need this abs_path_for_dynamic_library call?
                        # We're already resolving the absolute path based on the conda environment variables
                        abs_path = abs_path_for_dynamic_library(libname, handle)
                        return LoadedDL(abs_path, False, ctypes_handle_to_unsigned_int(handle))
            if build_prefix := os.getenv("BUILD_PREFIX"):
                build_prefix_normal_lib_path = os.path.join(build_prefix, normal_conda_lib_path)
                if os.path.isdir(build_prefix_normal_lib_path):
                    dll_name = os.path.join(build_prefix_normal_lib_path, dll_name)
                    handle = kernel32.LoadLibraryExW(dll_name, None, 0)
                    if handle:
                        # TODO KEITH: Do we need this abs_path_for_dynamic_library call?
                        # We're already resolving the absolute path based on the conda environment variables
                        abs_path = abs_path_for_dynamic_library(libname, handle)
                        return LoadedDL(abs_path, False, ctypes_handle_to_unsigned_int(handle))
        elif in_conda_env:
            if conda_prefix := os.getenv("CONDA_PREFIX"):
                conda_prefix_normal_lib_path = os.path.join(conda_prefix, normal_conda_lib_path)
                if os.path.isdir(conda_prefix_normal_lib_path):
                    dll_name = os.path.join(conda_prefix_normal_lib_path, dll_name)
                    handle = kernel32.LoadLibraryExW(dll_name, None, 0)
                    if handle:
                        # TODO KEITH: Do we need this abs_path_for_dynamic_library call?
                        # We're already resolving the absolute path based on the conda environment variables
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
    if libname in LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY:
        add_dll_directory(found_path)

    flags = WINBASE_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | WINBASE_LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
    handle = kernel32.LoadLibraryExW(found_path, None, flags)

    if not handle:
        error_code = ctypes.GetLastError()  # type: ignore[attr-defined]
        raise RuntimeError(f"Failed to load DLL at {found_path}: Windows error {error_code}")

    return LoadedDL(found_path, False, ctypes_handle_to_unsigned_int(handle))
