# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import ctypes
import ctypes.util
import os
from typing import Optional, cast

from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import (
    LIBNAMES_REQUIRING_RTLD_DEEPBIND,
    SUPPORTED_LINUX_SONAMES,
)

CDLL_MODE = os.RTLD_NOW | os.RTLD_GLOBAL


def _load_libdl() -> ctypes.CDLL:
    # In normal glibc-based Linux environments, find_library("dl") should return
    # something like "libdl.so.2". In minimal or stripped-down environments
    # (no ldconfig/gcc, incomplete linker cache), this can return None even
    # though libdl is present. In that case, we fall back to the stable SONAME.
    name = ctypes.util.find_library("dl") or "libdl.so.2"
    try:
        return ctypes.CDLL(name)
    except OSError as e:
        raise RuntimeError(f"Could not load {name!r} (required for dlinfo/dlerror on Linux)") from e


LIBDL = _load_libdl()

# dlinfo
LIBDL.dlinfo.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
LIBDL.dlinfo.restype = ctypes.c_int

# dlerror (thread-local error string; cleared after read)
LIBDL.dlerror.argtypes = []
LIBDL.dlerror.restype = ctypes.c_char_p

# First appeared in 2004-era glibc. Universally correct on Linux for all practical purposes.
RTLD_DI_LINKMAP = 2
RTLD_DI_ORIGIN = 6


class _LinkMapLNameView(ctypes.Structure):
    """
    Prefix-only view of glibc's `struct link_map` used **solely** to read `l_name`.

    Background:
      - `dlinfo(handle, RTLD_DI_LINKMAP, ...)` returns a `struct link_map*`.
      - The first few members of `struct link_map` (including `l_name`) have been
        stable on glibc for decades and are documented as debugger-visible.
      - We only need the offset/layout of `l_name`, not the full struct.

    Safety constraints:
      - This is a **partial** definition (prefix). It must only be used via a pointer
        returned by `dlinfo(...)`.
      - Do **not** instantiate it or pass it **by value** to any C function.
      - Do **not** access any members beyond those declared here.
      - Do **not** rely on `ctypes.sizeof(LinkMapPrefix)` for allocation.

    Rationale:
      - Defining only the leading fields avoids depending on internal/unstable
        tail members while keeping code more readable than raw pointer arithmetic.
    """

    _fields_ = (
        ("l_addr", ctypes.c_void_p),  # ElfW(Addr)
        ("l_name", ctypes.c_char_p),  # char*
    )


# Defensive assertions, mainly  to document the invariants we depend on
assert _LinkMapLNameView.l_addr.offset == 0
assert _LinkMapLNameView.l_name.offset == ctypes.sizeof(ctypes.c_void_p)


def _dl_last_error() -> Optional[str]:
    msg_bytes = cast(Optional[bytes], LIBDL.dlerror())
    if not msg_bytes:
        return None  # no pending error
    # Never raises; undecodable bytes are mapped to U+DC80..U+DCFF
    return msg_bytes.decode("utf-8", "surrogateescape")


def l_name_for_dynamic_library(libname: str, handle: ctypes.CDLL) -> str:
    lm_view = ctypes.POINTER(_LinkMapLNameView)()
    rc = LIBDL.dlinfo(ctypes.c_void_p(handle._handle), RTLD_DI_LINKMAP, ctypes.byref(lm_view))
    if rc != 0:
        err = _dl_last_error()
        raise OSError(f"dlinfo failed for {libname=!r} (rc={rc})" + (f": {err}" if err else ""))
    if not lm_view:  # NULL link_map**
        raise OSError(f"dlinfo returned NULL link_map pointer for {libname=!r}")

    l_name_bytes = lm_view.contents.l_name
    if not l_name_bytes:
        raise OSError(f"dlinfo returned empty link_map->l_name for {libname=!r}")

    path = os.fsdecode(l_name_bytes)
    if not path:
        raise OSError(f"dlinfo returned empty l_name string for {libname=!r}")

    return path


def l_origin_for_dynamic_library(libname: str, handle: ctypes.CDLL) -> str:
    l_origin_buf = ctypes.create_string_buffer(4096)
    rc = LIBDL.dlinfo(ctypes.c_void_p(handle._handle), RTLD_DI_ORIGIN, l_origin_buf)
    if rc != 0:
        err = _dl_last_error()
        raise OSError(f"dlinfo failed for {libname=!r} (rc={rc})" + (f": {err}" if err else ""))

    path = os.fsdecode(l_origin_buf.value)
    if not path:
        raise OSError(f"dlinfo returned empty l_origin string for {libname=!r}")

    return path


def abs_path_for_dynamic_library(libname: str, handle: ctypes.CDLL) -> str:
    l_name = l_name_for_dynamic_library(libname, handle)
    l_origin = l_origin_for_dynamic_library(libname, handle)
    return os.path.join(l_origin, os.path.basename(l_name))


def get_candidate_sonames(libname: str) -> list[str]:
    # Reverse tabulated names to achieve new → old search order.
    candidate_sonames = list(reversed(SUPPORTED_LINUX_SONAMES.get(libname, ())))
    candidate_sonames.append(f"lib{libname}.so")
    return candidate_sonames


def check_if_already_loaded_from_elsewhere(libname: str, _have_abs_path: bool) -> Optional[LoadedDL]:
    for soname in get_candidate_sonames(libname):
        try:
            handle = ctypes.CDLL(soname, mode=os.RTLD_NOLOAD)
        except OSError:
            continue
        else:
            return LoadedDL(abs_path_for_dynamic_library(libname, handle), True, handle._handle)
    return None


def _load_lib(libname: str, filename: str) -> ctypes.CDLL:
    cdll_mode = CDLL_MODE
    if libname in LIBNAMES_REQUIRING_RTLD_DEEPBIND:
        cdll_mode |= os.RTLD_DEEPBIND
    return ctypes.CDLL(filename, cdll_mode)


def load_with_system_search(libname: str) -> Optional[LoadedDL]:
    """Try to load a library using system search paths.

    Args:
        libname: The name of the library to load

    Returns:
        A LoadedDL object if successful, None if the library cannot be loaded

    Raises:
        RuntimeError: If the library is loaded but no expected symbol is found
    """
    for soname in get_candidate_sonames(libname):
        try:
            handle = _load_lib(libname, soname)
        except OSError:
            pass
        else:
            abs_path = abs_path_for_dynamic_library(libname, handle)
            if abs_path is None:
                raise RuntimeError(f"No expected symbol for {libname=!r}")
            return LoadedDL(abs_path, False, handle._handle)
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
        handle = _load_lib(libname, found_path)
    except OSError as e:
        raise RuntimeError(f"Failed to dlopen {found_path}: {e}") from e
    return LoadedDL(found_path, False, handle._handle)
