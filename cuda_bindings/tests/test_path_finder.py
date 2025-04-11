import pytest

from cuda.bindings import path_finder


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
