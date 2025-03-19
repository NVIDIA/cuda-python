from cuda.bindings.ecosystem import cuda_paths

paths = cuda_paths.get_cuda_paths()
for k, v in cuda_paths.get_cuda_paths().items():
    print(f"{k}: {v}", flush=True)
