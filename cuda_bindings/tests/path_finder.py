from cuda.bindings import path_finder

for k, v in path_finder.get_cuda_paths().items():
    print(f"{k}: {v}", flush=True)
