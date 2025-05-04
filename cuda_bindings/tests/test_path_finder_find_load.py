# Copyright 2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import os
import subprocess  # nosec B404
import sys

import pytest

from cuda.bindings import path_finder
from cuda.bindings._path_finder import supported_libs

ALL_LIBNAMES = path_finder._SUPPORTED_LIBNAMES + supported_libs.PARTIALLY_SUPPORTED_LIBNAMES_ALL
ALL_LIBNAMES_LINUX = path_finder._SUPPORTED_LIBNAMES + supported_libs.PARTIALLY_SUPPORTED_LIBNAMES_LINUX
ALL_LIBNAMES_WINDOWS = path_finder._SUPPORTED_LIBNAMES + supported_libs.PARTIALLY_SUPPORTED_LIBNAMES_WINDOWS
if os.environ.get("CUDA_BINDINGS_PATH_FINDER_TEST_ALL_LIBNAMES", False):
    if sys.platform == "win32":
        TEST_FIND_OR_LOAD_LIBNAMES = ALL_LIBNAMES_WINDOWS
    else:
        TEST_FIND_OR_LOAD_LIBNAMES = ALL_LIBNAMES_LINUX
else:
    TEST_FIND_OR_LOAD_LIBNAMES = path_finder._SUPPORTED_LIBNAMES


def test_all_libnames_linux_sonames_consistency():
    assert tuple(sorted(ALL_LIBNAMES_LINUX)) == tuple(sorted(supported_libs.SUPPORTED_LINUX_SONAMES.keys()))


def test_all_libnames_windows_dlls_consistency():
    assert tuple(sorted(ALL_LIBNAMES_WINDOWS)) == tuple(sorted(supported_libs.SUPPORTED_WINDOWS_DLLS.keys()))


def test_all_libnames_libnames_requiring_os_add_dll_directory_consistency():
    assert not (set(supported_libs.LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY) - set(ALL_LIBNAMES_WINDOWS))


def test_all_libnames_expected_lib_symbols_consistency():
    assert tuple(sorted(ALL_LIBNAMES)) == tuple(sorted(supported_libs.EXPECTED_LIB_SYMBOLS.keys()))


def _build_subprocess_failed_for_libname_message(libname, result):
    return (
        f"Subprocess failed for {libname=!r} with exit code {result.returncode}\n"
        f"--- stdout-from-subprocess ---\n{result.stdout}<end-of-stdout-from-subprocess>\n"
        f"--- stderr-from-subprocess ---\n{result.stderr}<end-of-stderr-from-subprocess>\n"
    )


@pytest.mark.parametrize("api", ("find", "load"))
@pytest.mark.parametrize("libname", TEST_FIND_OR_LOAD_LIBNAMES)
def test_find_or_load_nvidia_dynamic_library(info_summary_append, api, libname):
    # We intentionally run each dynamic library operation in a subprocess
    # to ensure isolation of global dynamic linking state (e.g., dlopen handles).
    # Without subprocesses, loading/unloading libraries during testing could
    # interfere across test cases and lead to nondeterministic or platform-specific failures.
    #
    # Defining the subprocess code snippets as strings ensures each subprocess
    # runs a minimal, independent script tailored to the specific libname and API being tested.
    if api == "find":
        code = f"""\
from cuda.bindings._path_finder.find_nvidia_dynamic_library import find_nvidia_dynamic_library
abs_path = find_nvidia_dynamic_library({libname!r})
print(f"{{abs_path!r}}")
"""
    else:
        code = f"""\
from cuda.bindings.path_finder import _load_nvidia_dynamic_library
from cuda.bindings._path_finder.load_nvidia_dynamic_library import _load_nvidia_dynamic_library_no_cache

loaded_dl_fresh = _load_nvidia_dynamic_library({libname!r})
if loaded_dl_fresh.was_already_loaded_from_elsewhere:
    raise RuntimeError("loaded_dl_fresh.was_already_loaded_from_elsewhere")

loaded_dl_from_cache = _load_nvidia_dynamic_library({libname!r})
if loaded_dl_from_cache is not loaded_dl_fresh:
    raise RuntimeError("loaded_dl_from_cache is not loaded_dl_fresh")

loaded_dl_no_cache = _load_nvidia_dynamic_library_no_cache({libname!r})
if not loaded_dl_no_cache.was_already_loaded_from_elsewhere:
    raise RuntimeError("loaded_dl_no_cache.was_already_loaded_from_elsewhere")
if loaded_dl_no_cache.abs_path != loaded_dl_fresh.abs_path:
    raise RuntimeError(f"{{loaded_dl_no_cache.abs_path=!r}} != {{loaded_dl_fresh.abs_path=!r}}")

print(f"{{loaded_dl_fresh.abs_path!r}}")
"""
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        timeout=30,  # Ensure CI testing does not hang for an excessive amount of time.
    )
    if result.returncode == 0:
        info_summary_append(f"abs_path={result.stdout.rstrip()}")
    else:
        raise RuntimeError(_build_subprocess_failed_for_libname_message(libname, result))
