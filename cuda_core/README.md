# `cuda.core`: Pythonic CUDA module

Currently under active development; see [the documentation](https://nvidia.github.io/cuda-python/cuda-core/latest/) for more details.

## Installing

Please refer to the [Installation page](https://nvidia.github.io/cuda-python/cuda-core/latest/install.html) for instructions and required/optional dependencies.

## Developing

This subpackage adheres to the developing practices described in the parent metapackage [CONTRIBUTING.md](https://github.com/NVIDIA/cuda-python/blob/main/CONTRIBUTING.md).

## Debugging

Pass the `pip` / `uv` configuration option `-C="debug=True"` or
`--config-settings="debug=True"` to explicitly to build debuggable binaries.
Debuggable binaries are built by default for editable builds.

Debuggable builds are not supported on Windows.

## Testing

To run these tests:
* `python -m pytest tests/` with editable installations
* `pytest tests/` with installed packages

### Cython Unit Tests

Cython tests are located in `tests/cython` and need to be built. These builds have the same CUDA Toolkit header requirements as [those of cuda.bindings](https://nvidia.github.io/cuda-python/cuda-bindings/latest/install.html#requirements) where the major.minor version must match `cuda.bindings`. To build them:

1. Set up environment variable `CUDA_PATH` (or `CUDA_HOME`) with the path to the CUDA Toolkit installation. Note: If both are set, `CUDA_PATH` takes precedence.
2. Run `build_tests` script located in `tests/cython` appropriate to your platform. This will both cythonize the tests and build them.

To run these tests:
* `python -m pytest tests/cython/` with editable installations
* `pytest tests/cython/` with installed packages
