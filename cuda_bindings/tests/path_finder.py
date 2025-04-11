from cuda.bindings import path_finder

paths = path_finder.get_cuda_paths()

for k, v in paths.items():
    print(f"{k}: {v}", flush=True)
print()

for libname in path_finder.SUPPORTED_LIBNAMES:
    print(libname)
    for fun in (path_finder.find_nvidia_dynamic_library, path_finder.load_nvidia_dynamic_library):
        try:
            out = fun(libname)
        except Exception as e:
            out = f"EXCEPTION: {type(e)} {str(e)}"
        print(out)
    print()
