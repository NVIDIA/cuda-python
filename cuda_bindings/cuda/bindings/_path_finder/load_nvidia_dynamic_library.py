# Copyright 2025 NVIDIA Corporation.  All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import functools
import sys

if sys.platform == "win32":
    import ctypes.wintypes

    import pywintypes
    import win32api

    # Mirrors WinBase.h (unfortunately not defined already elsewhere)
    _WINBASE_LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100
    _WINBASE_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000

else:
    import ctypes
    import os

    _LINUX_CDLL_MODE = os.RTLD_NOW | os.RTLD_GLOBAL

from .find_nvidia_dynamic_library import _find_nvidia_dynamic_library
from .supported_libs import DIRECT_DEPENDENCIES, SUPPORTED_WINDOWS_DLLS


@functools.cache
def _windows_cuDriverGetVersion() -> int:
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


@functools.cache
def _windows_load_with_dll_basename(name: str) -> int:
    driver_ver = _windows_cuDriverGetVersion()
    del driver_ver  # Keeping this here because it will probably be needed in the future.

    dll_names = SUPPORTED_WINDOWS_DLLS.get(name)
    if dll_names is None:
        return None

    for dll_name in dll_names:
        try:
            return win32api.LoadLibrary(dll_name)
        except pywintypes.error:
            pass

    return None


@functools.cache
def load_nvidia_dynamic_library(libname: str) -> int:
    for dep in DIRECT_DEPENDENCIES.get(libname, ()):
        load_nvidia_dynamic_library(dep)

    found = _find_nvidia_dynamic_library(libname)
    if found.abs_path is None:
        if sys.platform == "win32":
            handle = _windows_load_with_dll_basename(libname)
            if handle:
                # Use `cdef void* ptr = <void*><intptr_t>` in cython to convert back to void*
                return handle
        else:
            try:
                handle = ctypes.CDLL(found.lib_searched_for, _LINUX_CDLL_MODE)
            except OSError:
                pass
            else:
                # Use `cdef void* ptr = <void*><uintptr_t>` in cython to convert back to void*
                return handle._handle  # C unsigned int
        found.raise_if_abs_path_is_None()

    if sys.platform == "win32":
        flags = _WINBASE_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | _WINBASE_LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
        try:
            handle = win32api.LoadLibraryEx(found.abs_path, 0, flags)
        except pywintypes.error as e:
            raise RuntimeError(f"Failed to load DLL at {found.abs_path}: {e}") from e
        # Use `cdef void* ptr = <void*><intptr_t>` in cython to convert back to void*
        return handle  # C signed int, matches win32api.GetProcAddress
    else:
        try:
            handle = ctypes.CDLL(found.abs_path, _LINUX_CDLL_MODE)
        except OSError as e:
            raise RuntimeError(f"Failed to dlopen {found.abs_path}: {e}") from e
        # Use `cdef void* ptr = <void*><uintptr_t>` in cython to convert back to void*
        return handle._handle  # C unsigned int
