A source build of ``cuda.core`` must use a ``cuda.h`` of the same major.minor as its
``cuda-bindings``. If they differ, the build fails early.
(https://github.com/NVIDIA/cuda-python/issues/2783) (#2920)
