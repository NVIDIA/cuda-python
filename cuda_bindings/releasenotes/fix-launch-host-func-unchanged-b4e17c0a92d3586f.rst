:func:`driver.cuLaunchHostFunc` and :func:`runtime.cudaLaunchHostFunc` now pass the host
function and its user data to CUDA unchanged. They used to route the call through
an internal trampoline that freed its state after the first invocation, so a host
function captured into a CUDA graph crashed on the second launch of that graph.
(`Issue #790 <https://github.com/NVIDIA/cuda-python/issues/790>`_)
Callers must retain the callback object and user data storage for the lifetime
of the graph and its executables, including pending launches; see
:ref:`host-function-lifetime`. (#3061)
