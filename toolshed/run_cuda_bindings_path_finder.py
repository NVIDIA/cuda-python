import sys
import traceback

from cuda.bindings import path_finder
from cuda.bindings._path_finder import cuda_paths, supported_libs

ALL_LIBNAMES = (
    path_finder._SUPPORTED_LIBNAMES + supported_libs.PARTIALLY_SUPPORTED_LIBNAMES
)


def run(args):
    assert len(args) == 0

    paths = cuda_paths.get_cuda_paths()
    for k, v in paths.items():
        print(f"{k}: {v}", flush=True)
    print()

    for libname in ALL_LIBNAMES:
        print(f"{libname=}")
        try:
            loaded_dl = path_finder._load_nvidia_dynamic_library(libname)
        except Exception:
            print(f"EXCEPTION for {libname=}:")
            traceback.print_exc(file=sys.stdout)
        else:
            print(f"    {loaded_dl.abs_path=!r}")
            print(f"    {loaded_dl.was_already_loaded_from_elsewhere=!r}")
        print()


if __name__ == "__main__":
    run(args=sys.argv[1:])
