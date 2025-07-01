# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import pytest
import spawned_process_runner

from cuda.pathfinder import SUPPORTED_NVIDIA_LIBNAMES, load_nvidia_dynamic_lib
from cuda.pathfinder._dynamic_libs import supported_nvidia_libs

ALL_LIBNAMES = SUPPORTED_NVIDIA_LIBNAMES + supported_nvidia_libs.PARTIALLY_SUPPORTED_LIBNAMES_ALL
ALL_LIBNAMES_LINUX = SUPPORTED_NVIDIA_LIBNAMES + supported_nvidia_libs.PARTIALLY_SUPPORTED_LIBNAMES_LINUX
ALL_LIBNAMES_WINDOWS = SUPPORTED_NVIDIA_LIBNAMES + supported_nvidia_libs.PARTIALLY_SUPPORTED_LIBNAMES_WINDOWS
if os.environ.get("CUDA_PATHFINDER_TEST_ALL_LIBNAMES", False):
    if sys.platform == "win32":
        TEST_FIND_OR_LOAD_LIBNAMES = ALL_LIBNAMES_WINDOWS
    else:
        TEST_FIND_OR_LOAD_LIBNAMES = ALL_LIBNAMES_LINUX
else:
    TEST_FIND_OR_LOAD_LIBNAMES = SUPPORTED_NVIDIA_LIBNAMES


def test_all_libnames_linux_sonames_consistency():
    assert tuple(sorted(ALL_LIBNAMES_LINUX)) == tuple(sorted(supported_nvidia_libs.SUPPORTED_LINUX_SONAMES.keys()))


def test_all_libnames_windows_dlls_consistency():
    assert tuple(sorted(ALL_LIBNAMES_WINDOWS)) == tuple(sorted(supported_nvidia_libs.SUPPORTED_WINDOWS_DLLS.keys()))


@pytest.mark.parametrize("dict_name", ["SUPPORTED_LINUX_SONAMES", "SUPPORTED_WINDOWS_DLLS"])
def test_libname_dict_values_are_unique(dict_name):
    libname_dict = getattr(supported_nvidia_libs, dict_name)
    libname_for_value = {}
    for libname, values in libname_dict.items():
        for value in values:
            prev_libname = libname_for_value.get(value)
            if prev_libname is not None:
                raise RuntimeError(f"Multiple libnames for {value!r}: {prev_libname}, {libname}")
            libname_for_value[value] = libname


def test_all_libnames_libnames_requiring_os_add_dll_directory_consistency():
    assert not (set(supported_nvidia_libs.LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY) - set(ALL_LIBNAMES_WINDOWS))


def test_all_libnames_expected_lib_symbols_consistency():
    assert tuple(sorted(ALL_LIBNAMES)) == tuple(sorted(supported_nvidia_libs.EXPECTED_LIB_SYMBOLS.keys()))


def build_child_process_failed_for_libname_message(libname, result):
    return (
        f"Child process failed for {libname=!r} with exit code {result.returncode}\n"
        f"--- stdout-from-child-process ---\n{result.stdout}<end-of-stdout-from-child-process>\n"
        f"--- stderr-from-child-process ---\n{result.stderr}<end-of-stderr-from-child-process>\n"
    )


def child_process_func(libname):
    import os

    from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import _load_lib_no_cache

    loaded_dl_fresh = load_nvidia_dynamic_lib(libname)
    if loaded_dl_fresh.was_already_loaded_from_elsewhere:
        raise RuntimeError("loaded_dl_fresh.was_already_loaded_from_elsewhere")

    loaded_dl_from_cache = load_nvidia_dynamic_lib(libname)
    if loaded_dl_from_cache is not loaded_dl_fresh:
        raise RuntimeError("loaded_dl_from_cache is not loaded_dl_fresh")

    loaded_dl_no_cache = _load_lib_no_cache(libname)
    if not loaded_dl_no_cache.was_already_loaded_from_elsewhere:
        raise RuntimeError("loaded_dl_no_cache.was_already_loaded_from_elsewhere")
    if not os.path.samefile(loaded_dl_no_cache.abs_path, loaded_dl_fresh.abs_path):
        raise RuntimeError(f"not os.path.samefile({loaded_dl_no_cache.abs_path=!r}, {loaded_dl_fresh.abs_path=!r})")

    print(f"{loaded_dl_fresh.abs_path!r}")


@pytest.mark.parametrize("libname", TEST_FIND_OR_LOAD_LIBNAMES)
def test_find_or_load_nvidia_dynamic_lib(info_summary_append, libname):
    # We intentionally run each dynamic library operation in a child process
    # to ensure isolation of global dynamic linking state (e.g., dlopen handles).
    # Without child processes, loading/unloading libraries during testing could
    # interfere across test cases and lead to nondeterministic or platform-specific failures.
    result = spawned_process_runner.run_in_spawned_child_process(child_process_func, args=(libname,), timeout=30)
    if result.returncode == 0:
        info_summary_append(f"abs_path={result.stdout.rstrip()}")
    else:
        raise RuntimeError(build_child_process_failed_for_libname_message(libname, result))
