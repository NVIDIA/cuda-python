# Copyright 2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import ctypes.wintypes
import functools
from typing import Optional

import pywintypes
import win32api

from .load_dl_common import LoadedDL, add_dll_directory

# Mirrors WinBase.h (unfortunately not defined already elsewhere)
WINBASE_LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100
WINBASE_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000


def abs_path_for_dynamic_library(handle: int) -> str:
    """Get the absolute path of a loaded dynamic library on Windows.

    Args:
        handle: The library handle

    Returns:
        The absolute path to the DLL file

    Raises:
        OSError: If GetModuleFileNameW fails
    """
    buf = ctypes.create_unicode_buffer(260)
    n_chars = ctypes.windll.kernel32.GetModuleFileNameW(ctypes.wintypes.HMODULE(handle), buf, len(buf))
    if n_chars == 0:
        raise OSError("GetModuleFileNameW failed")
    return buf.value


@functools.cache
def cuDriverGetVersion() -> int:
    """Get the CUDA driver version.

    Returns:
        The CUDA driver version number

    Raises:
        AssertionError: If the driver version cannot be obtained
    """
    handle = win32api.LoadLibrary("nvcuda.dll")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    GetProcAddress = kernel32.GetProcAddress
    GetProcAddress.argtypes = [ctypes.wintypes.HMODULE, ctypes.wintypes.LPCSTR]
    GetProcAddress.restype = ctypes.c_void_p
    cuDriverGetVersion = GetProcAddress(handle, b"cuDriverGetVersion")
    assert cuDriverGetVersion

    FUNC_TYPE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_int))
    cuDriverGetVersion_fn = FUNC_TYPE(cuDriverGetVersion)
    driver_ver = ctypes.c_int()
    err = cuDriverGetVersion_fn(ctypes.byref(driver_ver))
    assert err == 0
    return driver_ver.value


def check_if_already_loaded(libname: str) -> Optional[LoadedDL]:
    """Check if the library is already loaded in the process.

    Args:
        libname: The name of the library to check

    Returns:
        A LoadedDL object if the library is already loaded, None otherwise

    Example:
        >>> loaded = check_if_already_loaded("cudart")
        >>> if loaded is not None:
        ...     print(f"Library already loaded from {loaded.abs_path}")
    """
    from .supported_libs import SUPPORTED_WINDOWS_DLLS

    for dll_name in SUPPORTED_WINDOWS_DLLS.get(libname, ()):
        try:
            handle = win32api.GetModuleHandle(dll_name)
        except pywintypes.error:
            continue
        else:
            return LoadedDL(handle, abs_path_for_dynamic_library(handle), True)
    return None


def load_with_system_search(name: str, _unused: str) -> Optional[LoadedDL]:
    """Try to load a DLL using system search paths.

    Args:
        name: The name of the library to load
        _unused: Unused parameter (kept for interface consistency)

    Returns:
        A LoadedDL object if successful, None if the library cannot be loaded
    """
    from .supported_libs import SUPPORTED_WINDOWS_DLLS

    driver_ver = cuDriverGetVersion()
    del driver_ver  # Keeping this here because it will probably be needed in the future.

    dll_names = SUPPORTED_WINDOWS_DLLS.get(name)
    if dll_names is None:
        return None

    for dll_name in dll_names:
        handle = ctypes.windll.kernel32.LoadLibraryW(ctypes.c_wchar_p(dll_name))
        if handle:
            return LoadedDL(handle, abs_path_for_dynamic_library(handle), False)

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
    from .supported_libs import LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY

    if libname in LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY:
        add_dll_directory(found_path)

    flags = WINBASE_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | WINBASE_LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
    try:
        handle = win32api.LoadLibraryEx(found_path, 0, flags)
    except pywintypes.error as e:
        raise RuntimeError(f"Failed to load DLL at {found_path}: {e}") from e
    return LoadedDL(handle, found_path, False)
