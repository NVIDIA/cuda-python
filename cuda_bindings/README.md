# `cuda.bindings`: Low-level CUDA interfaces

CUDA Python is a standard set of low-level interfaces, providing full coverage of and access to the CUDA host APIs from Python. Checkout the [Overview](https://nvidia.github.io/cuda-python/cuda-bindings/latest/overview.html) for the workflow and performance results.

## Installing

CUDA Python can be installed from:

* PYPI
* Conda (nvidia channel)
* Source builds

Differences between these options are described in [Installation](https://nvidia.github.io/cuda-python/cuda-bindings/latest/install.html) documentation. Each package guarantees minor version compatibility.

## Runtime Dependencies

CUDA Python is supported on all platforms that CUDA is supported. Specific dependencies are as follows:

* Driver: Linux (450.80.02 or later) Windows (456.38 or later)
* CUDA Toolkit 12.0 to 12.6

Only the NVRTC redistributable component is required from the CUDA Toolkit. [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/index.html) Installation Guides can be used for guidance. Note that the NVRTC component in the Toolkit can be obtained via PYPI, Conda or Local Installer.

### Supported Python Versions

CUDA Python follows [NEP 29](https://numpy.org/neps/nep-0029-deprecation_policy.html) for supported Python version guarantee.

Before dropping support, an issue will be raised to look for feedback.

Source builds work for multiple Python versions, however pre-build PyPI and Conda packages are only provided for a subset:

* Python 3.9 to 3.12

## Testing

Latest dependencies can be found in [requirements.txt](https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/requirements.txt).

Multiple testing options are available:

* Cython Unit Tests
* Python Unit Tests
* Samples
* Benchmark

### Python Unit Tests

Responsible for validating different binding usage patterns. Unit test `test_kernelParams.py` is particularly special since it demonstrates various approaches in setting up kernel launch parameters.

To run these tests:
* `python -m pytest tests/` against local builds
* `pytest tests/` against installed packages

### Cython Unit Tests

Cython tests are located in `tests/cython` and need to be built. Furthermore they need CUDA Toolkit headers matching the major-minor of CUDA Python. To build them:

1. Setup environment variable `CUDA_HOME` with the path to the CUDA Toolkit installation.
2. Run `build_tests` script located in `test/cython` appropriate to your platform. This will both cythonize the tests and build them.

To run these tests:
* `python -m pytest tests/cython/` against local builds
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
* `python -m pytest tests/cython/` against local builds
* `pytest tests/cython/` against installed packages

### Benchmark (WIP)

Benchmarks were used for performance analysis during initial release of CUDA Python. Today they need to be updated the 12.x toolkit and are work in progress.

The intended way to run these benchmarks was:
* `python -m pytest --benchmark-only benchmark/` against local builds
* `pytest --benchmark-only benchmark/` against installed packages
