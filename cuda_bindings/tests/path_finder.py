import sys

from cuda.bindings import path_finder


def run(args):
    assert len(args) == 0

    paths = path_finder.get_cuda_paths()

    for k, v in paths.items():
        print(f"{k}: {v}", flush=True)
    print()

    for libname in path_finder.SUPPORTED_WINDOWS_DLLS:
        if libname not in path_finder.SUPPORTED_LIBNAMES:
            print(f"MISSING IN SUPPORTED_LIBNAMES: {libname}")

    for libname in path_finder.SUPPORTED_LIBNAMES:
        print(libname)
        dlls = path_finder.SUPPORTED_WINDOWS_DLLS.get(libname)
        if dlls is None:
            print(f"MISSING IN SUPPORTED_WINDOWS_DLLS: {libname}")
        for fun in (path_finder.find_nvidia_dynamic_library, path_finder.load_nvidia_dynamic_library):
            try:
                out = fun(libname)
            except Exception as e:
                out = f"EXCEPTION: {type(e)} {str(e)}"
            print(out)
        print()


if __name__ == "__main__":
    run(args=sys.argv[1:])
