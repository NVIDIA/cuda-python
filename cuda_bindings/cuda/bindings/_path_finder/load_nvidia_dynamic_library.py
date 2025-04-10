import functools
import os
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
    print(f"\nLOOOK load_nvidia_dynamic_library({name=})", flush=True)
    # First try using the platform-specific dynamic loader search mechanisms
    if sys.platform == "win32":
        handle = _windows_load_with_dll_basename(name)
        if handle:
            print("\nLOOOK return handle", flush=True)
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
        print(f"\nLOOOK win32api.LoadLibrary({dl_path=})", flush=True)
        dirnm = os.path.dirname(dl_path)
        if os.path.isdir(dirnm):
            print(f"\nLOOOK   {dirnm=}", flush=True)
            for node in os.listdir(dirnm):
                print(f"\nLOOOK     {node=}", flush=True)
        flags = _WINBASE_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | _WINBASE_LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
        try:
            handle = win32api.LoadLibraryEx(dl_path, 0, flags)
        except pywintypes.error as e:
            raise RuntimeError(f"Failed to load DLL at {dl_path}: {e}") from e
        # Use `cdef void* ptr = <void*><intptr_t>` in cython to convert back to void*
        print("\nLOOOK return handle", flush=True)
        return handle  # C signed int, matches win32api.GetProcAddress
    else:
        try:
            handle = ctypes.CDLL(dl_path, _LINUX_CDLL_MODE)
        except OSError as e:
            raise RuntimeError(f"Failed to dlopen {dl_path}: {e}") from e
        # Use `cdef void* ptr = <void*><uintptr_t>` in cython to convert back to void*
        return handle._handle  # C unsigned int
