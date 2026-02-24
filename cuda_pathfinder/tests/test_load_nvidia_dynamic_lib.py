# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import platform

import pytest
from child_load_nvidia_dynamic_lib_helper import build_child_process_failed_for_libname_message, child_process_func
from local_helpers import have_distribution

from cuda.pathfinder import DynamicLibNotAvailableError, DynamicLibUnknownError, load_nvidia_dynamic_lib
from cuda.pathfinder._dynamic_libs import load_nvidia_dynamic_lib as load_nvidia_dynamic_lib_module
from cuda.pathfinder._dynamic_libs import supported_nvidia_libs
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS, quote_for_shell
from cuda.pathfinder._utils.spawned_process_runner import run_in_spawned_child_process

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


def test_runtime_error_on_non_64bit_python(mocker):
    # Ensure this test is not affected by any prior cached calls.
    load_nvidia_dynamic_lib.cache_clear()
    mocker.patch("struct.calcsize", return_value=3)  # fake 24-bit pointer
    with pytest.raises(RuntimeError, match=r"requires 64-bit Python\. Currently running: 24-bit Python"):
        load_nvidia_dynamic_lib("cudart")


def test_unknown_libname_raises_dynamic_lib_unknown_error():
    with pytest.raises(DynamicLibUnknownError, match=r"Unknown library name: 'not_a_real_lib'.*cudart"):
        load_nvidia_dynamic_lib("not_a_real_lib")


def test_known_but_platform_unavailable_libname_raises_dynamic_lib_not_available_error(monkeypatch):
    load_nvidia_dynamic_lib.cache_clear()
    monkeypatch.setattr(load_nvidia_dynamic_lib_module, "_ALL_KNOWN_LIBNAMES", frozenset(("known_but_unavailable",)))
    monkeypatch.setattr(load_nvidia_dynamic_lib_module, "_ALL_SUPPORTED_LIBNAMES", frozenset())
    monkeypatch.setattr(load_nvidia_dynamic_lib_module, "_PLATFORM_NAME", "TestOS")
    with pytest.raises(
        DynamicLibNotAvailableError,
        match=r"known_but_unavailable.*not available on TestOS",
    ):
        load_nvidia_dynamic_lib("known_but_unavailable")


IMPORTLIB_METADATA_DISTRIBUTIONS_NAMES = {
    "cufftMp": r"^nvidia-cufftmp-.*$",
    "mathdx": r"^nvidia-libmathdx-.*$",
}


def _is_expected_load_nvidia_dynamic_lib_failure(libname):
    if libname == "nvpl_fftw" and platform.machine().lower() != "aarch64":
        return True
    dist_name_pattern = IMPORTLIB_METADATA_DISTRIBUTIONS_NAMES.get(libname)
    if dist_name_pattern is not None:
        return not have_distribution(dist_name_pattern)
    return False


@pytest.mark.parametrize(
    "libname",
    supported_nvidia_libs.SUPPORTED_WINDOWS_DLLS if IS_WINDOWS else supported_nvidia_libs.SUPPORTED_LINUX_SONAMES,
)
def test_load_nvidia_dynamic_lib(info_summary_append, libname):
    # We intentionally run each dynamic library operation in a child process
    # to ensure isolation of global dynamic linking state (e.g., dlopen handles).
    # Without child processes, loading/unloading libraries during testing could
    # interfere across test cases and lead to nondeterministic or platform-specific failures.
    timeout = 120 if IS_WINDOWS else 30
    result = run_in_spawned_child_process(child_process_func, args=(libname,), timeout=timeout)

    def raise_child_process_failed():
        raise RuntimeError(build_child_process_failed_for_libname_message(libname, result))

    if result.returncode != 0:
        raise_child_process_failed()
    assert not result.stderr
    if result.stdout.startswith("CHILD_LOAD_NVIDIA_DYNAMIC_LIB_HELPER_DYNAMIC_LIB_NOT_FOUND_ERROR:"):
        if STRICTNESS == "all_must_work" and not _is_expected_load_nvidia_dynamic_lib_failure(libname):
            raise_child_process_failed()
        info_summary_append(f"Not found: {libname=!r}")
    else:
        abs_path = json.loads(result.stdout.rstrip())
        info_summary_append(f"abs_path={quote_for_shell(abs_path)}")
        assert os.path.isfile(abs_path)  # double-check the abs_path
