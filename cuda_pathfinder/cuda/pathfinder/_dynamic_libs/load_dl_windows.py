# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import ctypes.wintypes
import os
import struct
from collections.abc import Iterator
from typing import TYPE_CHECKING

from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL

if TYPE_CHECKING:
    from cuda.pathfinder._dynamic_libs.lib_descriptor import LibDescriptor

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


def check_if_already_loaded_from_elsewhere(desc: LibDescriptor, have_abs_path: bool) -> LoadedDL | None:
    for dll_name in desc.windows_dlls:
        handle = kernel32.GetModuleHandleW(dll_name)
        if handle:
            abs_path = abs_path_for_dynamic_library(desc.name, handle)
            if have_abs_path and desc.requires_add_dll_directory:
                # This is a side-effect if the pathfinder loads the library via
                # load_with_abs_path(). To make the side-effect more deterministic,
                # activate it even if the library was already loaded from elsewhere.
                add_dll_directory(abs_path)
            return LoadedDL(abs_path, True, ctypes_handle_to_unsigned_int(handle), "was-already-loaded-from-elsewhere")
    return None


def _iter_env_path_directories(path_value: str | None) -> Iterator[str]:
    """Yield normalized directories from PATH without consulting the current directory."""
    seen: set[str] = set()
    if not path_value:
        return

    for raw_entry in path_value.split(os.pathsep):
        entry = os.path.expandvars(raw_entry.strip().strip('"'))
        if not entry:
            continue
        if not os.path.isabs(entry):
            # Relative PATH entries would implicitly consult the current
            # directory, which we explicitly avoid for DLL lookup.
            continue
        if not os.path.isdir(entry):
            continue

        normalized_entry = os.path.normcase(os.path.normpath(entry))
        if normalized_entry in seen:
            continue
        seen.add(normalized_entry)
        yield entry


def _find_dll_on_env_path(dll_name: str) -> str | None:
    """Locate a DLL by scanning PATH entries explicitly."""
    for dirpath in _iter_env_path_directories(os.environ.get("PATH")):
        candidate = os.path.join(dirpath, dll_name)
        if os.path.isfile(candidate):
            return candidate
    return None


def _try_load_with_process_dll_search(desc: LibDescriptor, dll_name: str) -> LoadedDL | None:
    """Try the process DLL search path configured by CPython/Windows."""
    handle = kernel32.LoadLibraryExW(dll_name, None, 0)
    if not handle:
        return None

    abs_path = abs_path_for_dynamic_library(desc.name, handle)
    return LoadedDL(abs_path, False, ctypes_handle_to_unsigned_int(handle), "system-search")


def _try_load_with_env_path_fallback(desc: LibDescriptor, dll_name: str) -> LoadedDL | None:
    """Fallback for CTK-style installs exposed only via PATH."""
    found_path = _find_dll_on_env_path(dll_name)
    if found_path is None:
        return None
    return load_with_abs_path(desc, found_path, "system-search")


def load_with_system_search(desc: LibDescriptor) -> LoadedDL | None:
    """Try to load a DLL using system search paths.

    Args:
        desc: Descriptor for the library to load

    Returns:
        A LoadedDL object if successful, None if the library cannot be loaded
    """
    dll_names = tuple(reversed(desc.windows_dlls))

    # Phase 1: preserve the native process DLL search path (application dir,
    # system32, AddDllDirectory user dirs, loaded-module list).
    for dll_name in dll_names:
        loaded = _try_load_with_process_dll_search(desc, dll_name)
        if loaded is not None:
            return loaded

    if desc.packaged_with == "driver":
        return None

    # Phase 2: explicit PATH fallback for CTK-style installs only. Avoid
    # SearchPathW because its search semantics differ from LoadLibraryExW and
    # can consult the current directory.
    for dll_name in dll_names:
        loaded = _try_load_with_env_path_fallback(desc, dll_name)
        if loaded is not None:
            return loaded

    return None


def load_with_abs_path(desc: LibDescriptor, found_path: str, found_via: str | None = None) -> LoadedDL:
    """Load a dynamic library from the given path.

    Args:
        desc: Descriptor for the library to load.
        found_path: The absolute path to the DLL file.
        found_via: Label indicating how the path was discovered.

    Returns:
        A LoadedDL object representing the loaded library.

    Raises:
        RuntimeError: If the DLL cannot be loaded.
    """
    if desc.requires_add_dll_directory:
        add_dll_directory(found_path)

    flags = WINBASE_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | WINBASE_LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
    handle = kernel32.LoadLibraryExW(found_path, None, flags)

    if not handle:
        error_code = ctypes.GetLastError()  # type: ignore[attr-defined]
        raise RuntimeError(f"Failed to load DLL at {found_path}: Windows error {error_code}")

    return LoadedDL(found_path, False, ctypes_handle_to_unsigned_int(handle), found_via)
