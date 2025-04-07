# `cuda.bindings`: Low-level CUDA interfaces

`cuda.bindings` is a standard set of low-level interfaces, providing full coverage of and access to the CUDA host APIs from Python. Checkout the [Overview page](https://nvidia.github.io/cuda-python/cuda-bindings/latest/overview.html) for the workflow and performance results.

`cuda.bindings` is a subpackage of `cuda-python`.

## Installing

Please refer to the [Installation page](https://nvidia.github.io/cuda-python/cuda-bindings/latest/install.html) for instructions and required/optional dependencies.

## Developing

We use `pre-commit` to manage various tools to help development and ensure consistency.
```shell
pip install pre-commit
```

### Code linting

Run this command before checking in the code changes
```shell
pre-commit run -a --show-diff-on-failure
```
to ensure the code formatting is in line of the requirements (as listed in [`pyproject.toml`](./pyproject.toml)).

### Code signing

This repository implements a security check to prevent the CI system from running untrusted code. A part of the
security check consists of checking if the git commits are signed. See
[here](https://docs.gha-runners.nvidia.com/apps/copy-pr-bot/faqs/#why-did-i-receive-a-comment-that-my-pull-request-requires-additional-validation)
and
[here](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification)
for more details, including how to sign your commits.

## Testing

Latest dependencies can be found in [requirements.txt](https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/requirements.txt).

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

1. Setup environment variable `CUDA_HOME` with the path to the CUDA Toolkit installation.
2. Run `build_tests` script located in `test/cython` appropriate to your platform. This will both cythonize the tests and build them.

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
* `python -m pytest tests/cython/` against editable installations
* `pytest tests/cython/` against installed packages

### Benchmark

Allows for analyzing binding performance using plugin [pytest-benchmark](https://github.com/ionelmc/pytest-benchmark).

To run these benchmarks:
* `python -m pytest --benchmark-only benchmarks/` against editable installations
* `pytest --benchmark-only benchmarks/` against installed packages
