# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This helper is factored out so spawned child processes only import this
# lightweight module. That avoids re-importing the test module (and
# repeating its potentially expensive setup) in every child process.

import json
import os
import sys
import traceback


def build_child_process_failed_for_libname_message(libname, result):
    return (
        f"Child process failed for {libname=!r} with exit code {result.returncode}\n"
        f"--- stdout-from-child-process ---\n{result.stdout}<end-of-stdout-from-child-process>\n"
        f"--- stderr-from-child-process ---\n{result.stderr}<end-of-stderr-from-child-process>\n"
    )


def validate_abs_path(abs_path):
    assert abs_path, f"empty path: {abs_path=!r}"
    assert os.path.isabs(abs_path), f"not absolute: {abs_path=!r}"
    assert os.path.isfile(abs_path), f"not a file: {abs_path=!r}"


def child_process_func(libname):
    from cuda.pathfinder import DynamicLibNotFoundError, load_nvidia_dynamic_lib
    from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import _load_lib_no_cache
    from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import (
        SUPPORTED_LINUX_SONAMES,
        SUPPORTED_WINDOWS_DLLS,
    )
    from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

    try:
        loaded_dl_fresh = load_nvidia_dynamic_lib(libname)
    except DynamicLibNotFoundError:
        print("CHILD_LOAD_NVIDIA_DYNAMIC_LIB_HELPER_DYNAMIC_LIB_NOT_FOUND_ERROR:")  # noqa: T201
        traceback.print_exc(file=sys.stdout)
        return
    if loaded_dl_fresh.was_already_loaded_from_elsewhere:
        raise RuntimeError("loaded_dl_fresh.was_already_loaded_from_elsewhere")
    validate_abs_path(loaded_dl_fresh.abs_path)

    loaded_dl_from_cache = load_nvidia_dynamic_lib(libname)
    if loaded_dl_from_cache is not loaded_dl_fresh:
        raise RuntimeError("loaded_dl_from_cache is not loaded_dl_fresh")

    loaded_dl_no_cache = _load_lib_no_cache(libname)
    # check_if_already_loaded_from_elsewhere relies on these:
    supported_libs = SUPPORTED_WINDOWS_DLLS if IS_WINDOWS else SUPPORTED_LINUX_SONAMES
    if not loaded_dl_no_cache.was_already_loaded_from_elsewhere and libname in supported_libs:
        raise RuntimeError("not loaded_dl_no_cache.was_already_loaded_from_elsewhere")
    if not os.path.samefile(loaded_dl_no_cache.abs_path, loaded_dl_fresh.abs_path):
        raise RuntimeError(f"not os.path.samefile({loaded_dl_no_cache.abs_path=!r}, {loaded_dl_fresh.abs_path=!r})")
    validate_abs_path(loaded_dl_no_cache.abs_path)

    print(json.dumps(loaded_dl_fresh.abs_path))  # noqa: T201
