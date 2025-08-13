# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import ctypes
import ctypes.util
import os
from typing import Optional

from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import SUPPORTED_LINUX_SONAMES

CDLL_MODE = os.RTLD_NOW | os.RTLD_GLOBAL

LIBDL_PATH = ctypes.util.find_library("dl") or "libdl.so.2"
LIBDL = ctypes.CDLL(LIBDL_PATH)
LIBDL.dlinfo.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
LIBDL.dlinfo.restype = ctypes.c_int


# First appeared in 2004-era glibc. Universally correct on Linux for all practical purposes.
RTLD_DI_LINKMAP = 2


class LinkMap(ctypes.Structure):
    # Minimal fields we need; layout matches glibc's struct link_map
    _fields_ = (
        ("l_addr", ctypes.c_void_p),
        ("l_name", ctypes.c_char_p),
        ("l_ld", ctypes.c_void_p),
        ("l_next", ctypes.c_void_p),
        ("l_prev", ctypes.c_void_p),
    )


def abs_path_for_dynamic_library(libname: str, handle: ctypes.CDLL) -> str:
    lm_ptr = ctypes.POINTER(LinkMap)()
    rc = LIBDL.dlinfo(ctypes.c_void_p(handle._handle), RTLD_DI_LINKMAP, ctypes.byref(lm_ptr))
    if rc == 0 and lm_ptr and lm_ptr.contents.l_name:
        path: str = lm_ptr.contents.l_name.decode()
        if path:
            return path

    raise OSError(f"abs_path_for_dynamic_library failed for {libname=!r}")


def check_if_already_loaded_from_elsewhere(libname: str) -> Optional[LoadedDL]:
    """Check if the library is already loaded in the process.

    Args:
        libname: The name of the library to check

    Returns:
        A LoadedDL object if the library is already loaded, None otherwise

    Example:
        >>> loaded = check_if_already_loaded_from_elsewhere("cudart")
        >>> if loaded is not None:
        ...     print(f"Library already loaded from {loaded.abs_path}")
    """
    from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import SUPPORTED_LINUX_SONAMES

    for soname in SUPPORTED_LINUX_SONAMES.get(libname, ()):
        try:
            handle = ctypes.CDLL(soname, mode=os.RTLD_NOLOAD)
        except OSError:
            continue
        else:
            return LoadedDL(abs_path_for_dynamic_library(libname, handle), True, handle._handle)
    return None


def load_with_system_search(libname: str) -> Optional[LoadedDL]:
    """Try to load a library using system search paths.

    Args:
        libname: The name of the library to load

    Returns:
        A LoadedDL object if successful, None if the library cannot be loaded

    Raises:
        RuntimeError: If the library is loaded but no expected symbol is found
    """
    candidate_sonames = list(SUPPORTED_LINUX_SONAMES.get(libname, ()))
    candidate_sonames.append(f"lib{libname}.so")
    for soname in candidate_sonames:
        try:
            handle = ctypes.CDLL(soname, CDLL_MODE)
            abs_path = abs_path_for_dynamic_library(libname, handle)
            if abs_path is None:
                raise RuntimeError(f"No expected symbol for {libname=!r}")
            return LoadedDL(abs_path, False, handle._handle)
        except OSError:
            pass
    return None


def _work_around_known_bugs(libname: str, found_path: str) -> None:
    if libname == "nvrtc":
        # Work around bug/oversight in
        #   nvidia_cuda_nvrtc-13.0.48-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl
        # Issue: libnvrtc.so.13 RUNPATH is not set.
        # This workaround is highly specific
        #   - for simplicity.
        #   - to not mask bugs in future nvidia-cuda-nvrtc releases.
        #   - because a more general workaround is complicated.
        dirname, basename = os.path.split(found_path)
        if basename == "libnvrtc.so.13":
            dep_basename = "libnvrtc-builtins.so.13.0"
            dep_path = os.path.join(dirname, dep_basename)
            if os.path.isfile(dep_path):
                # In case of failure, defer to primary load, which is almost certain to fail, too.
                with contextlib.suppress(OSError):
                    ctypes.CDLL(dep_path, CDLL_MODE)


def load_with_abs_path(libname: str, found_path: str) -> LoadedDL:
    """Load a dynamic library from the given path.

    Args:
        libname: The name of the library to load
        found_path: The absolute path to the library file

    Returns:
        A LoadedDL object representing the loaded library

    Raises:
        RuntimeError: If the library cannot be loaded
    """
    _work_around_known_bugs(libname, found_path)
    try:
        handle = ctypes.CDLL(found_path, CDLL_MODE)
    except OSError as e:
        raise RuntimeError(f"Failed to dlopen {found_path}: {e}") from e
    return LoadedDL(found_path, False, handle._handle)
