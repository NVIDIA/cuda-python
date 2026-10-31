# `cuda.bindings`: Low-level CUDA interfaces

`cuda.bindings` is a standard set of low-level interfaces, providing full coverage of and access to the CUDA host APIs from Python. Checkout the [Overview page](https://nvidia.github.io/cuda-python/cuda-bindings/latest/overview.html) for the workflow and performance results.

## Installing

Please refer to the [Installation page](https://nvidia.github.io/cuda-python/cuda-bindings/latest/install.html) for instructions and required/optional dependencies.

## Developing

This subpackage adheres to the developing practices described in the parent metapackage [CONTRIBUTING.md](https://github.com/NVIDIA/cuda-python/blob/main/CONTRIBUTING.md).

## Debugging

Pass the `pip` / `uv` configuration option `-C="debug=True"` or
`--config-settings="debug=True"` to explicitly to build debuggable binaries.
Debuggable binaries are built by default for editable builds.

Debuggable builds are not supported on Windows.

## Testing

Run the commands below from `cuda_bindings_13/`, in an environment containing
the CUDA 13 bindings. Install the package and its `test` dependency group with
`python -m pip install -v -e . --group test`.

Multiple testing options are available:

* Python Unit Tests
* Cython Unit Tests
* Samples
* Benchmark

### Python Unit Tests

Responsible for validating different binding usage patterns. Unit test `test_kernelParams.py` is particularly special since it demonstrates various approaches in setting up kernel launch parameters.

To run these tests:
* `python -m pytest tests/` against editable installations
* `pytest tests/` against installed packages

### Cython Unit Tests

Cython tests are located in `tests/cython` and need to be built. These builds have the same CUDA Toolkit header requirements as [Installing from Source](https://nvidia.github.io/cuda-python/cuda-bindings/latest/install.html#requirements) where the major.minor version must match `cuda.bindings`. To build them:

1. Setup environment variable `CUDA_PATH` (or `CUDA_HOME`) with the path to the CUDA Toolkit installation. Note: If both are set, `CUDA_PATH` takes precedence.
2. Run the `build_tests` script located in `tests/cython` appropriate to your platform. This will both cythonize the tests and build them.

To run these tests:
* `python -m pytest tests/cython/` against editable installations
* `pytest tests/cython/` against installed packages

### Samples

Various [CUDA Samples](https://github.com/NVIDIA/cuda-samples/tree/master) that were rewritten using CUDA Python are located in [samples/cuda_bindings](../samples/cuda_bindings/). Each sample directory has its own README with dependencies and instructions for running it.

In addition, extra examples are included:

* [jitProgram.py](../samples/cuda_bindings/extra/jitProgram/jitProgram.py): Demonstrates the use of the API to compile and
  launch a kernel on the device. Includes device memory allocation /
  deallocation, transfers between host and device, creation and usage of
  streams, and context management.

To run the automated sample coverage in `tests/example_tests/`, run
`pixi run test-samples` from this directory.

### Benchmark

Python and C++ benchmarks for analyzing binding overhead live in
[benchmarks/cuda_bindings](../benchmarks/cuda_bindings/). See the
[benchmark README](../benchmarks/cuda_bindings/README.md) for the Pixi environments,
benchmark commands, and result comparison instructions.
