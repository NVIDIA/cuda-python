# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import ctypes.wintypes
import os
import struct
import sys
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
psapi = ctypes.windll.psapi  # type: ignore[attr-defined]

# GetModuleHandleW
kernel32.GetModuleHandleW.argtypes = [ctypes.wintypes.LPCWSTR]
kernel32.GetModuleHandleW.restype = ctypes.wintypes.HMODULE

# GetCurrentProcess
kernel32.GetCurrentProcess.argtypes = []
kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE

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

# EnumProcessModules
psapi.EnumProcessModules.argtypes = [
    ctypes.wintypes.HANDLE,
    ctypes.POINTER(ctypes.wintypes.HMODULE),
    ctypes.wintypes.DWORD,
    ctypes.POINTER(ctypes.wintypes.DWORD),
]
psapi.EnumProcessModules.restype = ctypes.wintypes.BOOL

_CUPTI_DIAGNOSTICS_ENVVAR = "CUDA_PATHFINDER_WINDOWS_CUPTI_ALREADY_LOADED_DIAGNOSTICS"


def _cupti_diagnostics_enabled(desc_name: str) -> bool:
    raw = os.environ.get(_CUPTI_DIAGNOSTICS_ENVVAR)
    if desc_name != "cupti" or raw is None:
        return False
    return raw.strip().lower() not in ("", "0", "false", "no")


def _emit_cupti_diagnostic(message: str) -> None:
    sys.stderr.write(f"[cuda.pathfinder][cupti-diag] {message}\n")


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


def _iter_loaded_module_handles() -> Iterator[ctypes.wintypes.HMODULE]:
    process_handle = kernel32.GetCurrentProcess()
    capacity = 64
    module_size = ctypes.sizeof(ctypes.wintypes.HMODULE)
    while True:
        module_handles = (ctypes.wintypes.HMODULE * capacity)()
        needed = ctypes.wintypes.DWORD()
        ok = psapi.EnumProcessModules(
            process_handle,
            module_handles,
            ctypes.sizeof(module_handles),
            ctypes.byref(needed),
        )
        if not ok:
            error_code = ctypes.GetLastError()  # type: ignore[attr-defined]
            raise RuntimeError(f"EnumProcessModules failed (error code: {error_code})")
        count = needed.value // module_size
        if count <= capacity:
            for raw_handle in module_handles[:count]:
                if raw_handle is None:
                    continue
                yield ctypes.wintypes.HMODULE(int(raw_handle))
            return
        capacity = count


def _find_loaded_module(
    dll_names: tuple[str, ...],
    *,
    diagnostics_enabled: bool = False,
) -> tuple[ctypes.wintypes.HMODULE, str] | None:
    wanted = {dll_name.casefold() for dll_name in dll_names}
    relevant_modules: list[str] = []
    for handle in _iter_loaded_module_handles():
        abs_path = abs_path_for_dynamic_library("loaded module", handle)
        basename = os.path.basename(abs_path)
        basename_casefold = basename.casefold()
        if diagnostics_enabled and ("cupti" in basename_casefold or "nvperf" in basename_casefold):
            relevant_modules.append(f"0x{ctypes_handle_to_unsigned_int(handle):x}:{abs_path}")
        if basename_casefold in wanted:
            if diagnostics_enabled:
                _emit_cupti_diagnostic(
                    "enumerated relevant modules: " + (" | ".join(relevant_modules) if relevant_modules else "<none>")
                )
                _emit_cupti_diagnostic(
                    f"enumeration match: basename={basename!r} abs_path={abs_path!r}"
                    f" handle=0x{ctypes_handle_to_unsigned_int(handle):x}"
                )
            return handle, abs_path
    if diagnostics_enabled:
        _emit_cupti_diagnostic(
            "enumerated relevant modules: " + (" | ".join(relevant_modules) if relevant_modules else "<none>")
        )
    return None


def check_if_already_loaded_from_elsewhere(desc: LibDescriptor, have_abs_path: bool) -> LoadedDL | None:
    diagnostics_enabled = _cupti_diagnostics_enabled(desc.name)
    basename_probe_results: list[str] = []
    for dll_name in desc.windows_dlls:
        handle = kernel32.GetModuleHandleW(dll_name)
        if diagnostics_enabled:
            handle_text = "0x0" if not handle else f"0x{ctypes_handle_to_unsigned_int(handle):x}"
            basename_probe_results.append(f"{dll_name}={handle_text}")
        if handle:
            abs_path = abs_path_for_dynamic_library(desc.name, handle)
            if diagnostics_enabled:
                _emit_cupti_diagnostic("basename GetModuleHandleW results: " + ", ".join(basename_probe_results))
                _emit_cupti_diagnostic(
                    f"basename match: dll_name={dll_name!r} abs_path={abs_path!r}"
                    f" handle=0x{ctypes_handle_to_unsigned_int(handle):x}"
                )
            if have_abs_path and desc.requires_add_dll_directory:
                # This is a side-effect if the pathfinder loads the library via
                # load_with_abs_path(). To make the side-effect more deterministic,
                # activate it even if the library was already loaded from elsewhere.
                add_dll_directory(abs_path)
            return LoadedDL(abs_path, True, ctypes_handle_to_unsigned_int(handle), "was-already-loaded-from-elsewhere")
    # Observed on newer Windows CUPTI builds: GetModuleHandleW(basename)
    # can miss an already loaded DLL, so fall back to enumerating loaded modules.
    if diagnostics_enabled:
        _emit_cupti_diagnostic("basename GetModuleHandleW results: " + ", ".join(basename_probe_results))
    loaded = _find_loaded_module(desc.windows_dlls, diagnostics_enabled=diagnostics_enabled)
    if loaded is not None:
        handle, abs_path = loaded
        if have_abs_path and desc.requires_add_dll_directory:
            add_dll_directory(abs_path)
        return LoadedDL(abs_path, True, ctypes_handle_to_unsigned_int(handle), "was-already-loaded-from-elsewhere")
    return None


def load_with_system_search(desc: LibDescriptor) -> LoadedDL | None:
    """Try to load a DLL using the native Windows process DLL search path.

    This calls ``LoadLibraryExW(dll_name, NULL, 0)`` directly. Under Python
    3.8+, CPython configures the process with
    ``SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)``, so this
    search does **not** include the system ``PATH``. Directories added via
    ``AddDllDirectory()`` still participate.

    Args:
        desc: Descriptor for the library to load

    Returns:
        A LoadedDL object if successful, None if the library cannot be loaded
    """
    # Reverse tabulated names to achieve new -> old search order.
    for dll_name in reversed(desc.windows_dlls):
        handle = kernel32.LoadLibraryExW(dll_name, None, 0)
        if handle:
            abs_path = abs_path_for_dynamic_library(desc.name, handle)
            return LoadedDL(abs_path, False, ctypes_handle_to_unsigned_int(handle), "system-search")

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
