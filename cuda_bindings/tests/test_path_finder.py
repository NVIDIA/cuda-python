import os
import subprocess
import sys

import pytest

from cuda.bindings import path_finder
from cuda.bindings._path_finder import supported_libs

ALL_LIBNAMES = path_finder.SUPPORTED_LIBNAMES + supported_libs.PARTIALLY_SUPPORTED_LIBNAMES
if os.environ.get("CUDA_BINDINGS_PATH_FINDER_TEST_ALL_LIBNAMES", False):
    TEST_LIBNAMES = ALL_LIBNAMES
else:
    TEST_LIBNAMES = path_finder.SUPPORTED_LIBNAMES


def test_all_libnames_windows_dlls_consistency():
    assert tuple(sorted(ALL_LIBNAMES)) == tuple(sorted(path_finder.SUPPORTED_WINDOWS_DLLS.keys()))


def _build_subprocess_failed_for_libname_message(libname, result):
    return (
        f"Subprocess failed for {libname=!r} with exit code {result.returncode}\n"
        f"--- stdout-from-subprocess ---\n{result.stdout}<end-of-stdout-from-subprocess>\n"
        f"--- stderr-from-subprocess ---\n{result.stderr}<end-of-stderr-from-subprocess>\n"
    )


@pytest.mark.parametrize("algo", ("find", "load"))
@pytest.mark.parametrize("libname", TEST_LIBNAMES)
def test_find_or_load_nvidia_dynamic_library(algo, libname):
    if sys.platform == "win32" and not path_finder.SUPPORTED_WINDOWS_DLLS[libname]:
        pytest.skip(f"{libname=!r} not supported on {sys.platform=}")

    code = f"""\
from cuda.bindings import path_finder
path_finder.{algo}_nvidia_dynamic_library({libname!r})
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(_build_subprocess_failed_for_libname_message(libname, result))
