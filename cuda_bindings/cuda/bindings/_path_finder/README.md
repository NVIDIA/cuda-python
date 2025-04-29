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

1. **Python Package Ecosystem**
   - Scans `sys.path` to find libraries installed via NVIDIA Python wheels

2. **Conda Environments**
   - Leverages Conda-specific paths through our fork of `get_cuda_paths()` from Numba

3. **System Installations**
   - Checks traditional system locations via the same `get_cuda_paths()` implementation

4. **OS Default Mechanisms**
   - Falls back to native loader:
     - `dlopen()` on Linux
     - `LoadLibraryW()` on Windows

## Implementation Philosophy

The current implementation balances stability and evolution:

- **Baseline Foundation:** Uses a fork of Numba's `cuda_paths.py` that has been
  battle-tested in production environments

- **Validation Infrastructure:** Comprehensive CI testing matrix being developed to cover:
  - Various Linux/Windows environments
  - Python packaging formats (wheels, conda)
  - CUDA Toolkit versions

- **Roadmap:** Planned refactoring to:
  - Unify library discovery logic
  - Improve maintainability
  - Better enforce search priority
  - Expand platform support

## Maintenance Requirements

These key components must be updated for new CUDA Toolkit releases:

- `supported_libs.SUPPORTED_LIBNAMES`
- `supported_libs.SUPPORTED_WINDOWS_DLLS`
- `supported_libs.SUPPORTED_LINUX_SONAMES`
- `supported_libs.EXPECTED_LIB_SYMBOLS`
