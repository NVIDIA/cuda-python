import subprocess
import sys

import pytest

from cuda.bindings import path_finder


def test_supported_libnames_windows_dlls_consistency():
    assert list(sorted(path_finder.SUPPORTED_LIBNAMES)) == list(sorted(path_finder.SUPPORTED_WINDOWS_DLLS.keys()))


@pytest.mark.parametrize("algo", ("find", "load")[1:])
@pytest.mark.parametrize("libname", path_finder.SUPPORTED_LIBNAMES)
def test_find_or_load_nvidia_dynamic_library(algo, libname):
    if sys.platform == "win32" and not path_finder.SUPPORTED_WINDOWS_DLLS[libname]:
        pytest.skip(f'"{libname}" not supported on this platform')

    code = f"""\
from cuda.bindings import path_finder
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
