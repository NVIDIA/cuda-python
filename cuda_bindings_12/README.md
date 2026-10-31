# `cuda.bindings`: Low-level CUDA interfaces

`cuda.bindings` is a standard set of low-level interfaces, providing full coverage of and access to the CUDA host APIs from Python. Checkout the [Overview page](https://nvidia.github.io/cuda-python/cuda-bindings/12.9.7/overview.html) for the workflow and performance results.

<!-- Keep documentation URLs aligned with DESCRIPTION.rst and build_hooks.py.
Use a published CUDA 12 snapshot; latest follows the current CUDA major. -->

## Installing

Please refer to the [Installation page](https://nvidia.github.io/cuda-python/cuda-bindings/12.9.7/install.html) for instructions and required/optional dependencies.

## Developing

Follow the repository [contribution guide](../CONTRIBUTING.md) and the
[maintenance guide](MAINTENANCE.md) for source builds and cross-major changes.

CUDA 12.9 sources are developed here on `main`. This root and
`cuda_bindings_13/` build the same distribution and Python namespace; use
separate environments for them. From the repository root:

```console
pixi run --manifest-path cuda_bindings_12 -e cu12 test
```

This builds bindings and Cython test extensions from source, then runs the
GPU test suite. `pixi run -e cu12 test-bindings` selects the same package.
For an existing Python environment with CUDA 12.9 headers and a compiler:

```console
cd cuda_bindings_12
python -m pip install -v -e . --group test
tests/cython/build_tests.sh
python -m pytest tests/ tests/cython/
```

Use `build_tests.bat` on Windows. Free-threaded Python also needs the
`test-ft` dependency group.

Versions follow reachable `v12.9.*` tags through standard `setuptools-scm`.
Before a retained milestone exists for the reserved preview release, CI uses
`ci.tools.bindings_scm_version` in its disposable checkout to create one
local ancestor tag. Never run that helper in a shared development repository
or push its synthetic tags. See the repository's
[version lifecycle](../AGENTS.md#git-derived-version-lifecycle).


## Testing

Run the commands below from `cuda_bindings_12/`, in an environment containing
the CUDA 12 bindings. Install the package and its `test` dependency group with
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

Cython tests are located in `tests/cython` and need to be built. These builds have the same CUDA Toolkit header requirements as [Installing from Source](https://nvidia.github.io/cuda-python/cuda-bindings/12.9.7/install.html#requirements) where the major.minor version must match `cuda.bindings`. To build them:

1. Setup environment variable `CUDA_HOME` with the path to the CUDA Toolkit installation.
2. Run the `build_tests` script located in `tests/cython` appropriate to your platform. This will both cythonize the tests and build them.

To run these tests:
* `python -m pytest tests/cython/` against editable installations
* `pytest tests/cython/` against installed packages

### Samples

Various [CUDA Samples](https://github.com/NVIDIA/cuda-samples/tree/master) that were rewritten using CUDA Python are located in `examples`.

In addition, extra examples are included:

* `examples/extra/jit_program_test.py`: Demonstrates the use of the API to compile and
  launch a kernel on the device. Includes device memory allocation /
  deallocation, transfers between host and device, creation and usage of
  streams, and context management.
* `examples/extra/numba_emm_plugin.py`: Implements a Numba External Memory Management
  plugin, showing that this CUDA Python Driver API can coexist with other
  wrappers of the driver API.

To run these samples:
* `python -m pytest examples/` against editable installations
* `pytest examples/` against installed packages

### Benchmark

See the shared [cuda.bindings benchmark harness](../benchmarks/cuda_bindings/README.md)
for performance benchmarks and usage instructions. Its configured environments
currently target CUDA 13; benchmarking CUDA 12 requires a compatible environment
configured separately.
