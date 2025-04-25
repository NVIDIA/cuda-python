# Copyright 2025 NVIDIA Corporation.  All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import functools
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

if sys.platform == "win32":
    import ctypes.wintypes

    import pywintypes
    import win32api

    # Mirrors WinBase.h (unfortunately not defined already elsewhere)
    _WINBASE_LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100
    _WINBASE_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000

else:
    import ctypes.util

    _LINUX_CDLL_MODE = os.RTLD_NOW | os.RTLD_GLOBAL

    _LIBDL_PATH = ctypes.util.find_library("dl") or "libdl.so.2"
    _LIBDL = ctypes.CDLL(_LIBDL_PATH)
    _LIBDL.dladdr.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _LIBDL.dladdr.restype = ctypes.c_int

    class Dl_info(ctypes.Structure):
        _fields_ = [
            ("dli_fname", ctypes.c_char_p),  # path to .so
            ("dli_fbase", ctypes.c_void_p),
            ("dli_sname", ctypes.c_char_p),
            ("dli_saddr", ctypes.c_void_p),
        ]


from .find_nvidia_dynamic_library import _find_nvidia_dynamic_library
from .supported_libs import (
    DIRECT_DEPENDENCIES,
    EXPECTED_LIB_SYMBOLS,
    LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY,
    SUPPORTED_LINUX_SONAMES,
    SUPPORTED_WINDOWS_DLLS,
)


def _add_dll_directory(dll_abs_path):
    dirpath = os.path.dirname(dll_abs_path)
    assert os.path.isdir(dirpath), dll_abs_path
    # Add the DLL directory to the search path
    os.add_dll_directory(dirpath)
    # Update PATH as a fallback for dependent DLL resolution
    curr_path = os.environ.get("PATH")
    os.environ["PATH"] = dirpath if curr_path is None else os.pathsep.join((curr_path, dirpath))


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


def _abs_path_for_dynamic_library_windows(handle: int) -> str:
    buf = ctypes.create_unicode_buffer(260)
    n_chars = ctypes.windll.kernel32.GetModuleFileNameW(ctypes.wintypes.HMODULE(handle), buf, len(buf))
    if n_chars == 0:
        raise OSError("GetModuleFileNameW failed")
    return buf.value


@functools.cache
def _windows_load_with_dll_basename(name: str) -> Tuple[Optional[int], Optional[str]]:
    driver_ver = _windows_cuDriverGetVersion()
    del driver_ver  # Keeping this here because it will probably be needed in the future.

    dll_names = SUPPORTED_WINDOWS_DLLS.get(name)
    if dll_names is None:
        return None

    for dll_name in dll_names:
        handle = ctypes.windll.kernel32.LoadLibraryW(ctypes.c_wchar_p(dll_name))
        if handle:
            return handle, _abs_path_for_dynamic_library_windows(handle)

    return None, None


def _abs_path_for_dynamic_library_linux(libname: str, handle: int) -> str:
    for symbol_name in EXPECTED_LIB_SYMBOLS[libname]:
        symbol = getattr(handle, symbol_name, None)
        if symbol is not None:
            break
    else:
        return None
    addr = ctypes.cast(symbol, ctypes.c_void_p)
    info = Dl_info()
    if _LIBDL.dladdr(addr, ctypes.byref(info)) == 0:
        raise OSError(f"dladdr failed for {libname=!r}")
    return info.dli_fname.decode()


def _load_and_report_path_linux(libname: str, soname: str) -> Tuple[int, str]:
    handle = ctypes.CDLL(soname, _LINUX_CDLL_MODE)
    abs_path = _abs_path_for_dynamic_library_linux(libname, handle)
    if abs_path is None:
        raise RuntimeError(f"No expected symbol for {libname=!r}")
    return handle, abs_path


@dataclass
class LoadedDL:
    # ATTENTION: To convert `handle` back to `void*` in cython:
    #     Linux:   `cdef void* ptr = <void*><uintptr_t>`
    #     Windows: `cdef void* ptr = <void*><intptr_t>`
    handle: int
    abs_path: Optional[str]
    was_already_loaded_from_elsewhere: bool


def _load_nvidia_dynamic_library_no_cache(libname: str) -> LoadedDL:
    # Detect if the library was loaded already in some other way (i.e. not via this function).
    if sys.platform == "win32":
        for dll_name in SUPPORTED_WINDOWS_DLLS.get(libname, ()):
            try:
                handle = win32api.GetModuleHandle(dll_name)
            except pywintypes.error:
                pass
            else:
                return LoadedDL(handle, _abs_path_for_dynamic_library_windows(handle), True)
    else:
        for soname in SUPPORTED_LINUX_SONAMES.get(libname, ()):
            try:
                handle = ctypes.CDLL(soname, mode=os.RTLD_NOLOAD)
            except OSError:
                pass
            else:
                return LoadedDL(handle, _abs_path_for_dynamic_library_linux(libname, handle), True)

    for dep in DIRECT_DEPENDENCIES.get(libname, ()):
        load_nvidia_dynamic_library(dep)

    found = _find_nvidia_dynamic_library(libname)
    if found.abs_path is None:
        if sys.platform == "win32":
            handle, abs_path = _windows_load_with_dll_basename(libname)
            if handle:
                return LoadedDL(handle, abs_path, False)
        else:
            try:
                handle, abs_path = _load_and_report_path_linux(libname, found.lib_searched_for)
            except OSError:
                pass
            else:
                return LoadedDL(handle._handle, abs_path, False)
        found.raise_if_abs_path_is_None()

    if sys.platform == "win32":
        if libname in LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY:
            _add_dll_directory(found.abs_path)
        flags = _WINBASE_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | _WINBASE_LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
        try:
            handle = win32api.LoadLibraryEx(found.abs_path, 0, flags)
        except pywintypes.error as e:
            raise RuntimeError(f"Failed to load DLL at {found.abs_path}: {e}") from e
        return LoadedDL(handle, found.abs_path, False)
    else:
        try:
            handle = ctypes.CDLL(found.abs_path, _LINUX_CDLL_MODE)
        except OSError as e:
            raise RuntimeError(f"Failed to dlopen {found.abs_path}: {e}") from e
        return LoadedDL(handle._handle, found.abs_path, False)


@functools.cache
def load_nvidia_dynamic_library(libname: str) -> LoadedDL:
    return _load_nvidia_dynamic_library_no_cache(libname)
