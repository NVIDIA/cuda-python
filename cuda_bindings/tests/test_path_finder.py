import sys

import pytest

from cuda.bindings import path_finder


def test_supported_libnames_windows_dlls_consistency():
    assert list(sorted(path_finder.SUPPORTED_LIBNAMES)) == list(sorted(path_finder.SUPPORTED_WINDOWS_DLLS.keys()))


@pytest.mark.parametrize("libname", path_finder.SUPPORTED_LIBNAMES)
def test_find_and_load(libname):
    if sys.platform == "win32" and libname == "cufile":
        pytest.skip(f'test_find_and_load("{libname}") not supported on this platform')
    print(f'\ntest_find_and_load("{libname}")')
    failures = []
    for algo, func in (
        ("find", path_finder.find_nvidia_dynamic_library),
        ("load", path_finder.load_nvidia_dynamic_library),
    ):
        try:
            out = func(libname)
        except Exception as e:
            out = f"EXCEPTION: {type(e)} {str(e)}"
            failures.append(algo)
        print(out)
    print()
    assert not failures
