# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import struct
import sys

from cuda.pathfinder._dynamic_libs.find_nvidia_dynamic_lib import _FindNvidiaDynamicLib
from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL, load_dependencies
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import IS_WINDOWS

if IS_WINDOWS:
    from cuda.pathfinder._dynamic_libs.load_dl_windows import (
        check_if_already_loaded_from_elsewhere,
        load_with_abs_path,
        load_with_system_search,
    )
else:
    from cuda.pathfinder._dynamic_libs.load_dl_linux import (
        check_if_already_loaded_from_elsewhere,
        load_with_abs_path,
        load_with_system_search,
    )


def _load_lib_no_cache(libname: str) -> LoadedDL:
    # Check whether the library is already loaded into the current process by
    # some other component. This check uses OS-level mechanisms (e.g.,
    # dlopen on Linux, GetModuleHandle on Windows).
    loaded = check_if_already_loaded_from_elsewhere(libname)
    if loaded is not None:
        return loaded

    # Load dependencies first
    load_dependencies(libname, load_nvidia_dynamic_lib)

    # Find the library path
    found = _FindNvidiaDynamicLib(libname)
    if found.abs_path is None:
        loaded = load_with_system_search(libname, found.lib_searched_for)
        if loaded is not None:
            return loaded
        found.retry_with_cuda_home_priority_last()
        found.raise_if_abs_path_is_None()

    # Load the library from the found path
    assert found.abs_path is not None  # for mypy
    return load_with_abs_path(libname, found.abs_path)


@functools.cache
def load_nvidia_dynamic_lib(libname: str) -> LoadedDL:
    """Load a NVIDIA dynamic library by name.

    Args:
        libname: The name of the library to load (e.g. "cudart", "nvvm", etc.)

    Returns:
        A LoadedDL object containing the library handle and path

    Raises:
        RuntimeError: If the library cannot be found or loaded
    """
    pointer_size_bits = struct.calcsize("P") * 8
    if pointer_size_bits != 64:
        raise RuntimeError(
            f"cuda.pathfinder.load_nvidia_dynamic_lib() requires 64-bit Python."
            f" Currently running: {pointer_size_bits}-bit Python"
            f" {sys.version_info.major}.{sys.version_info.minor}"
        )
    return _load_lib_no_cache(libname)
