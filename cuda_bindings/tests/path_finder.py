from cuda.bindings import path_finder

paths = path_finder.get_cuda_paths()

for k, v in paths.items():
    print(f"{k}: {v}", flush=True)
