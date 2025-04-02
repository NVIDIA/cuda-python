import ctypes
import functools
import os
import sys

from .find_nvidia_dynamic_library import find_nvidia_dynamic_library


@functools.cache
def load_nvidia_dynamic_library(name: str) -> int:
    dl_path = find_nvidia_dynamic_library(name)
    if sys.platform == "win32":
        try:
            handle = ctypes.windll.kernel32.LoadLibraryW(dl_path)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
        except Exception as e:
            raise RuntimeError(f"Failed to load DLL at {dl_path}: {e}") from e
        return handle
    else:
        try:
            handle = ctypes.CDLL(dl_path, mode=os.RTLD_NOW | os.RTLD_GLOBAL)
            return handle._handle  # Raw void* as int
        except OSError as e:
            raise RuntimeError(f"Failed to dlopen {dl_path}: {e}") from e
