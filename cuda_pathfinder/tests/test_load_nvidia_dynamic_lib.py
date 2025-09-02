# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from unittest.mock import patch

import pytest
import spawned_process_runner
from child_load_nvidia_dynamic_lib_helper import build_child_process_failed_for_libname_message, child_process_func

from cuda.pathfinder import SUPPORTED_NVIDIA_LIBNAMES, load_nvidia_dynamic_lib
from cuda.pathfinder._dynamic_libs import supported_nvidia_libs
from cuda.pathfinder._utils.find_site_packages_dll import find_all_dll_files_via_metadata
from cuda.pathfinder._utils.find_site_packages_so import find_all_so_files_via_metadata

STRICTNESS = os.environ.get("CUDA_PATHFINDER_TEST_LOAD_NVIDIA_DYNAMIC_LIB_STRICTNESS", "see_what_works")
assert STRICTNESS in ("see_what_works", "all_must_work")


def test_supported_libnames_linux_sonames_consistency():
    assert tuple(sorted(supported_nvidia_libs.SUPPORTED_LIBNAMES_LINUX)) == tuple(
        sorted(supported_nvidia_libs.SUPPORTED_LINUX_SONAMES_CTK.keys())
    )


def test_supported_libnames_windows_dlls_consistency():
    assert tuple(sorted(supported_nvidia_libs.SUPPORTED_LIBNAMES_WINDOWS)) == tuple(
        sorted(supported_nvidia_libs.SUPPORTED_WINDOWS_DLLS_CTK.keys())
    )


def test_supported_libnames_linux_site_packages_libdirs_ctk_consistency():
    assert tuple(sorted(supported_nvidia_libs.SUPPORTED_LIBNAMES_LINUX)) == tuple(
        sorted(supported_nvidia_libs.SITE_PACKAGES_LIBDIRS_LINUX_CTK.keys())
    )


def test_supported_libnames_windows_site_packages_libdirs_ctk_consistency():
    assert tuple(sorted(supported_nvidia_libs.SUPPORTED_LIBNAMES_WINDOWS)) == tuple(
        sorted(supported_nvidia_libs.SITE_PACKAGES_LIBDIRS_WINDOWS_CTK.keys())
    )


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


def test_supported_libnames_windows_libnames_requiring_os_add_dll_directory_consistency():
    assert not (
        set(supported_nvidia_libs.LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY)
        - set(supported_nvidia_libs.SUPPORTED_LIBNAMES_WINDOWS)
    )


def test_runtime_error_on_non_64bit_python():
    with (
        patch("struct.calcsize", return_value=3),  # fake 24-bit pointer
        pytest.raises(RuntimeError, match=r"requires 64-bit Python\. Currently running: 24-bit Python"),
    ):
        load_nvidia_dynamic_lib("not_used")


@functools.cache
def _get_libnames_for_test_load_nvidia_dynamic_lib():
    result = list(SUPPORTED_NVIDIA_LIBNAMES)
    if supported_nvidia_libs.IS_WINDOWS:
        spld_other = supported_nvidia_libs.SITE_PACKAGES_LIBDIRS_WINDOWS_OTHER
        all_dyn_libs = find_all_dll_files_via_metadata()
        for libname in spld_other:
            for dll_name in all_dyn_libs:
                if dll_name.startswith(libname):
                    result.append(libname)
    else:
        spld_other = supported_nvidia_libs.SITE_PACKAGES_LIBDIRS_LINUX_OTHER
        all_dyn_libs = find_all_so_files_via_metadata()
        for libname in spld_other:
            so_basename = f"lib{libname}.so"
            if so_basename in all_dyn_libs:
                result.append(libname)

    return tuple(result)


@pytest.mark.parametrize("libname", _get_libnames_for_test_load_nvidia_dynamic_lib())
def test_load_nvidia_dynamic_lib(info_summary_append, libname):
    # We intentionally run each dynamic library operation in a child process
    # to ensure isolation of global dynamic linking state (e.g., dlopen handles).
    # Without child processes, loading/unloading libraries during testing could
    # interfere across test cases and lead to nondeterministic or platform-specific failures.
    timeout = 120 if supported_nvidia_libs.IS_WINDOWS else 30
    result = spawned_process_runner.run_in_spawned_child_process(child_process_func, args=(libname,), timeout=timeout)
    if result.returncode == 0:
        info_summary_append(f"abs_path={result.stdout.rstrip()}")
    elif STRICTNESS == "see_what_works" or "DynamicLibNotFoundError: Failure finding " in result.stderr:
        info_summary_append(f"Not found: {libname=!r}")
    else:
        raise RuntimeError(build_child_process_failed_for_libname_message(libname, result))
