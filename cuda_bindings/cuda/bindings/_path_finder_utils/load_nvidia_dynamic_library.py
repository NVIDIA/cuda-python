import functools
import sys

if sys.platform == "win32":
    import pywintypes
    import win32api
else:
    import ctypes
    import os

from .find_nvidia_dynamic_library import find_nvidia_dynamic_library


@functools.cache
def load_nvidia_dynamic_library(name: str) -> int:
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
            handle = ctypes.CDLL(dl_path, mode=os.RTLD_NOW | os.RTLD_GLOBAL)
        except OSError as e:
            raise RuntimeError(f"Failed to dlopen {dl_path}: {e}") from e
        # Use `cdef void* ptr = <void*><uintptr_t>` in cython to convert back to void*
        return handle._handle  # C unsigned int
