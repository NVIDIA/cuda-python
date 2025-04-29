`cuda.bindings.path_finder`
===========================

Currently, the only two (semi-)public APIs are:

* `cuda.bindings.path_finder._SUPPORTED_LIBNAMES` (currently `('nvJitLink', 'nvrtc', 'nvvm')`)

* `cuda.bindings.path_finder._load_nvidia_dynamic_library(libname: str) -> LoadedDL`

These APIs are prefixed with an underscore because they are STILL A WORK IN PROGRESS,
although already fairly well tested.

`load_nvidia_dynamic_library()` is meant to become the one, central, go-to API for
loading NVIDIA shared libraries from Python.


Search Priority
---------------

The _intended_ search priority for locating NVIDIA dynamic libraries is:

* *site-packages* — Traversal of Python's `sys.path` (in order), which for all practical
  purposes amounts to a search for libraries installed from NVIDIA wheels.

* *Conda* — Currently mplemented via `get_cuda_paths()` as forked from numba/cuda/cuda_paths.py

* *System* — Also implemented via `get_cuda_paths()`

* *OS-provided search* — `dlopen()` (Linux) or `LoadLibraryW()` (Windows) mechanisms


Currently, our fork of cuda_paths.py is intentionally used as-is.
cuda_paths.py has a long and convoluted development history, but that also means
the product is time-tested. Our strategy for evolving the implementation is:

* Establish a minimal viable product as a baseline (current stage).

* Establish a comprehensive testing infrastructure (GitHub Actions / CI) to
  cover all sorts of environments that we want to support.

* Combine, refactor, and clean up find_nvidia_dynamic_library.py & cuda_paths.py
  to achieve a more maintainable and robust implementation of the intended
  dynamic library search priority.
