The C++ layer calls the driver through the entry points ``cuda-bindings`` resolves rather than
through its Cython wrappers. As a result, a driver function that the installed driver lacks can
no longer surface as ``SystemError`` from a C++ call.
(https://github.com/NVIDIA/cuda-python/issues/2783) (#2920)
