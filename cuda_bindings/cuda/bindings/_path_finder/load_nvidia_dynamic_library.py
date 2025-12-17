import functools
import sys

if sys.platform == "win32":
    import ctypes.wintypes

    import pywintypes
    import win32api

    # Mirrors WinBase.h (unfortunately not defined already elsewhere)
    _WINBASE_LOAD_LIBRARY_SEARCH_SYSTEM32 = 0x00000800

else:
    import ctypes
    import os

    _LINUX_CDLL_MODE = os.RTLD_NOW | os.RTLD_GLOBAL

from .find_nvidia_dynamic_library import find_nvidia_dynamic_library


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

    if name == "nvJitLink":
        dll_name = "nvJitLink_120_0.dll"
    elif name == "nvrtc":
        dll_name = "nvrtc64_120_0.dll"
    elif name == "nvvm":
        dll_name = "nvvm64_40_0.dll"

    try:
        return win32api.LoadLibrary(dll_name)
    except pywintypes.error:
        pass

    return None


@functools.cache
def load_nvidia_dynamic_library(name: str) -> int:
    # First try using the platform-specific dynamic loader search mechanisms
    if sys.platform == "win32":
        handle = _windows_load_with_dll_basename(name)
        if handle:
            return handle
    else:
        dl_path = f"lib{name}.so"  # Version intentionally no specified.
        try:
            handle = ctypes.CDLL(dl_path, _LINUX_CDLL_MODE)
        except OSError:
            pass
        else:
            # Use `cdef void* ptr = <void*><uintptr_t>` in cython to convert back to void*
            return handle._handle  # C unsigned int

    dl_path = find_nvidia_dynamic_library(name)
    if sys.platform == "win32":
        try:
            handle = win32api.LoadLibrary(dl_path)
        except pywintypes.error as e:
            raise RuntimeError(f"Failed to load DLL at {dl_path}: {e}") from e
        # Use `cdef void* ptr = <void*><intptr_t>` in cython to convert back to void*
        return handle  # C signed int, matches win32api.GetProcAddress
    else:
        try:
            handle = ctypes.CDLL(dl_path, _LINUX_CDLL_MODE)
        except OSError as e:
            raise RuntimeError(f"Failed to dlopen {dl_path}: {e}") from e
        # Use `cdef void* ptr = <void*><uintptr_t>` in cython to convert back to void*
        return handle._handle  # C unsigned int