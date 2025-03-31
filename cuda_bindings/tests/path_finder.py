from cuda.bindings import path_finder

paths = path_finder.get_cuda_paths()

for k, v in paths.items():
    print(f"{k}: {v}", flush=True)

print(path_finder.find_nvidia_dynamic_library("nvvm", "TEST"))
print(path_finder.find_nvidia_dynamic_library("nvJitLink", "TEST"))
