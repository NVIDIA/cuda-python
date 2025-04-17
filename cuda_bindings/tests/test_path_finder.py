import subprocess
import sys

import pytest

from cuda.bindings import path_finder


def test_supported_libnames_windows_dlls_consistency():
    assert list(sorted(path_finder.SUPPORTED_LIBNAMES)) == list(sorted(path_finder.SUPPORTED_WINDOWS_DLLS.keys()))


@pytest.mark.parametrize("algo", ("find", "load"))
@pytest.mark.parametrize("libname", path_finder.SUPPORTED_LIBNAMES)
def test_find_or_load_nvidia_dynamic_library(algo, libname):
    if sys.platform == "win32" and libname == "cufile":
        pytest.skip(f'test_find_and_load("{libname}") not supported on this platform')

    code = """\
from cuda.bindings import path_finder
"""
    if algo == "load" and libname == "cusolver":
        code += """\
path_finder.load_nvidia_dynamic_library("nvJitLink")
path_finder.load_nvidia_dynamic_library("cusparse")
path_finder.load_nvidia_dynamic_library("cublas")
"""
    code += f"""\
path_finder.load_nvidia_dynamic_library({libname!r})
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed for libname={libname!r} with exit code {result.returncode}\\n"
            f"--- stdout ---\\n{result.stdout}\\n"
            f"--- stderr ---\\n{result.stderr}"
        )
