# Copyright 2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import functools
import json

from cuda.bindings._path_finder.find_nvidia_dynamic_library import _find_nvidia_dynamic_library
from cuda.bindings._path_finder.load_dl_common import (
    LoadedDL,
    build_subprocess_failed_for_libname_message,
    load_dependencies,
    load_in_subprocess,
)
from cuda.bindings._path_finder.supported_libs import IS_WINDOWS

if IS_WINDOWS:
    from cuda.bindings._path_finder.load_dl_windows import (
        check_if_already_loaded_from_elsewhere,
        load_with_abs_path,
        load_with_system_search,
    )
else:
    from cuda.bindings._path_finder.load_dl_linux import (
        check_if_already_loaded_from_elsewhere,
        load_with_abs_path,
        load_with_system_search,
    )


def _load_other_in_subprocess(libname, error_messages):
    code = f"""\
from cuda.bindings._path_finder.load_nvidia_dynamic_library import load_nvidia_dynamic_library
import json
import sys
loaded = load_nvidia_dynamic_library({libname!r})
sys.stdout.write(json.dumps(loaded.abs_path, ensure_ascii=True))
"""
    result = load_in_subprocess(code)
    if result.returncode == 0:
        return json.loads(result.stdout)
    error_messages.extend(build_subprocess_failed_for_libname_message(libname, result).splitlines())
    return None


def _load_nvidia_dynamic_library_no_cache(libname: str) -> LoadedDL:
    # Check whether the library is already loaded into the current process by
    # some other component. This check uses OS-level mechanisms (e.g.,
    # dlopen on Linux, GetModuleHandle on Windows).
    loaded = check_if_already_loaded_from_elsewhere(libname)
    if loaded is not None:
        return loaded

    # Load dependencies first
    load_dependencies(libname, load_nvidia_dynamic_library)

    # Find the library path
    found = _find_nvidia_dynamic_library(libname)
    if found.abs_path is None:
        loaded = load_with_system_search(libname, found.lib_searched_for)
        if loaded is not None:
            return loaded
        if libname == "nvvm":
            # Use cudart as anchor point (libcudart.so.12 is only ~720K, cudart64_12.dll ~560K).
            loaded_cudart = check_if_already_loaded_from_elsewhere("cudart")
            if loaded_cudart is not None:
                found.retry_with_other_abs_path(loaded_cudart.abs_path)
            else:
                cudart_abs_path = _load_other_in_subprocess("cudart", found.error_messages)
                if cudart_abs_path is not None:
                    found.retry_with_other_abs_path(cudart_abs_path)
        found.raise_if_abs_path_is_None()

    # Load the library from the found path
    return load_with_abs_path(libname, found.abs_path)


@functools.cache
def load_nvidia_dynamic_library(libname: str) -> LoadedDL:
    """Load a NVIDIA dynamic library by name.

    Args:
        libname: The name of the library to load (e.g. "cudart", "nvvm", etc.)

    Returns:
        A LoadedDL object containing the library handle and path

    Raises:
        RuntimeError: If the library cannot be found or loaded
    """
    return _load_nvidia_dynamic_library_no_cache(libname)
