# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools

from cuda.pathfinder._dynamic_libs.find_nvidia_dynamic_library import (
    _find_nvidia_dynamic_library,
)
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
    load_dependencies(libname, load_lib)

    # Find the library path
    found = _find_nvidia_dynamic_library(libname)
    if found.abs_path is None:
        loaded = load_with_system_search(libname, found.lib_searched_for)
        if loaded is not None:
            return loaded
        found.retry_with_cuda_home_priority_last()
        found.raise_if_abs_path_is_None()

    # Load the library from the found path
    return load_with_abs_path(libname, found.abs_path)


@functools.cache
def load_lib(libname: str) -> LoadedDL:
    return _load_lib_no_cache(libname)
