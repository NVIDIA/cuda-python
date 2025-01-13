# Environment Variables

## Build-Time Environment Variables

- `CUDA_HOME` or `CUDA_PATH`: Specifies the location of the CUDA Toolkit.

- `CUDA_PYTHON_PARSER_CACHING` : bool, toggles the caching of parsed header files during the cuda-bindings build process. If caching is enabled (`CUDA_PYTHON_PARSER_CACHING` is True), the cache path is set to ./cache_<library_name>, where <library_name> is derived from the cuda toolkit libraries used to build cuda-bindings.

- `CUDA_PYTHON_PARALLEL_LEVEL` (previously `PARALLEL_LEVEL`) : int, sets the number of threads used in the compilation of cython files. This is passed as the `nthreads` argument to :meth:`cython.cythonize` and as the parallel attribute for building extension modules.

## Runtime Environment Variables

- `CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM` : When set to 1, the default stream is the per-thread default stream. When set to 0, the default stream is the legacy default stream. This defaults to 0, for the legacy default stream. See [Stream Synchronization Behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html) for an explanation of the legacy and per-thread default streams.
