import ctypes
import os

from .find_nvidia_dynamic_library import find_nvidia_dynamic_library


def load_nvidia_dynamic_library(name: str) -> int:
    path = find_nvidia_dynamic_library(name)
    try:
        handle = ctypes.CDLL(path, mode=os.RTLD_NOW | os.RTLD_GLOBAL)
        return handle._handle  # This is the actual `void*` value as an int
    except OSError as e:
        raise RuntimeError(f"Failed to dlopen {path}: {e}") from e
