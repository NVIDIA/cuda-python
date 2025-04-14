import pytest

from cuda.bindings import path_finder


def test_supported_libnames_windows_dlls_consistency():
    assert list(sorted(path_finder.SUPPORTED_LIBNAMES)) == list(sorted(path_finder.SUPPORTED_WINDOWS_DLLS.keys()))


@pytest.mark.parametrize("libname", path_finder.SUPPORTED_LIBNAMES)
def test_find_and_load(libname):
    print(f"\n{libname}")
    for fun in (path_finder.find_nvidia_dynamic_library, path_finder.load_nvidia_dynamic_library):
        try:
            out = fun(libname)
        except Exception as e:
            out = f"EXCEPTION: {type(e)} {str(e)}"
        print(out)
    print()
