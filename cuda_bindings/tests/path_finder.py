import sys

from cuda.bindings import path_finder
from cuda.bindings._path_finder import cuda_paths, supported_libs
from cuda.bindings._path_finder.find_nvidia_dynamic_library import find_nvidia_dynamic_library

ALL_LIBNAMES = path_finder.SUPPORTED_LIBNAMES + supported_libs.PARTIALLY_SUPPORTED_LIBNAMES


def run(args):
    assert len(args) == 0

    paths = cuda_paths.get_cuda_paths()

    for k, v in paths.items():
        print(f"{k}: {v}", flush=True)
    print()

    for libname in supported_libs.SUPPORTED_WINDOWS_DLLS:
        if libname not in ALL_LIBNAMES:
            print(f"MISSING IN SUPPORTED_LIBNAMES: {libname}")

    for libname in ALL_LIBNAMES:
        print(libname)
        dlls = supported_libs.SUPPORTED_WINDOWS_DLLS.get(libname)
        if dlls is None:
            print(f"MISSING IN SUPPORTED_WINDOWS_DLLS: {libname}")
        for fun in (find_nvidia_dynamic_library, path_finder.load_nvidia_dynamic_library):
            try:
                out = fun(libname)
            except Exception as e:
                out = f"EXCEPTION: {type(e)} {str(e)}"
            print(out)
        print()


if __name__ == "__main__":
    run(args=sys.argv[1:])
