from cuda.bindings import path_finder

paths = path_finder.get_cuda_paths()

for k, v in paths.items():
    print(f"{k}: {v}", flush=True)
print()

libnames = ("nvJitLink", "nvrtc", "nvvm")

for libname in libnames:
    print(path_finder.find_nvidia_dynamic_library(libname))
    print()

for libname in libnames:
    print(libname)
    print(path_finder.load_nvidia_dynamic_library(libname))
    print()
