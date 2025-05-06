# `cuda.bindings.path_finder` Module

## Public API (Work in Progress)

Currently exposes two primary interfaces:

```
cuda.bindings.path_finder._SUPPORTED_LIBNAMES  # ('nvJitLink', 'nvrtc', 'nvvm')
cuda.bindings.path_finder._load_nvidia_dynamic_library(libname: str) -> LoadedDL
```

**Note:**
These APIs are prefixed with an underscore because they are considered
experimental while undergoing active development, although already
reasonably well-tested through CI pipelines.

## Library Loading Search Priority

The `load_nvidia_dynamic_library()` function implements a hierarchical search
strategy for locating NVIDIA shared libraries:

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
   - CTK installations with system config updates are expected to be discovered:
     - Linux: Via `/etc/ld.so.conf.d/*cuda*.conf`
     - Windows: Via `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\bin` on system `PATH`
   - Conda installations are expected to be discovered:
     - Linux: Via `$ORIGIN/../lib` on `RPATH` (of the `python` binary)
     - Windows: Via `%CONDA_PREFIX%\Library\bin` on system `PATH`

3. **Environment variables**
   - Relies on `CUDA_HOME` or `CUDA_PATH` environment variables if set
     (in that order).

Note that the search is done on a per-library basis. There is no centralized
mechanism that ensures all libraries are found in the same way.

## Maintenance Requirements

These key components must be updated for new CUDA Toolkit releases:

- `supported_libs.SUPPORTED_LIBNAMES`
- `supported_libs.SUPPORTED_WINDOWS_DLLS`
- `supported_libs.SUPPORTED_LINUX_SONAMES`
- `supported_libs.EXPECTED_LIB_SYMBOLS`
