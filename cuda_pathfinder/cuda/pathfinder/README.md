# `cuda.pathfinder` Module

## Public API for loading NVIDIA Dynamic Libs

* `cuda.pathfinder.SUPPORTED_NVIDIA_LIBNAMES` (`tuple[str]`)

* `cuda.pathfinder.load_nvidia_dynamic_lib(libname: str) -> LoadedDL`

* `cuda.pathfinder.LoadedDL`:
    * `abs_path` (`str`)
    * `was_already_loaded_from_elsewhere` (`bool`)

* `cuda.pathfinder.DynamicLibNotFoundError` (inherits from `RuntimeError`)

## Dynamic Library Loading Search Priority

The `cuda.pathfinder.load_nvidia_dynamic_lib` function implements a
hierarchical search strategy for locating NVIDIA shared libraries:

0. **Check if a library was loaded into the process already by some other means.**
   - If yes, there is no alternative to skipping the rest of the search logic.
     The absolute path of the already loaded library will be returned, along
     with the handle to the library.

1. **NVIDIA Python wheels**
   - Scans all site-packages to find libraries installed via NVIDIA Python wheels.

2. **OS default mechanisms / Conda environments**
   - Falls back to native loader:
     - `dlopen()` on Linux
     - `LoadLibraryW()` on Windows
   - Conda installations are expected to be discovered:
     - Linux: Via `$ORIGIN/../lib` on `RPATH` (of the `python` binary;
       note that this preempts `LD_LIBRARY_PATH` and `/etc/ld.so.conf.d/`)
     - Windows: Via `%CONDA_PREFIX%\Library\bin` on system `PATH`
   - CTK installations with system config updates are expected to be discovered:
     - Linux: Via `/etc/ld.so.conf.d/*cuda*.conf`
     - Windows: Via `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\bin` on system `PATH`

3. **Environment variables**
   - Relies on `CUDA_HOME` or `CUDA_PATH` environment variables if set
     (in that order).

Note that the search is done on a per-library basis. Currently there is no
centralized mechanism that ensures all libraries are found in the same way.
