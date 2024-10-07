# CUDA-Python

CUDA Python is a standard set of low-level interfaces, providing full coverage of and access to the CUDA host APIs from Python. Checkout the [Overview](https://nvidia.github.io/cuda-python/overview.html) for the workflow and performance results.

## Installing

CUDA Python can be installed from:

* PYPI
* Conda (nvidia channel)
* Source builds

There're differences in each of these options that are described further in [Installation](https://nvidia.github.io/cuda-python/install.html) documentation. Each package will guarantee minor version compatibility.

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

### Requirements

Latest dependencies can be found in [requirements.txt](https://github.com/NVIDIA/cuda-python/blob/main/requirements.txt).

### Unit-tests

You can run the included tests with:

```
python -m pytest
```

### Benchmark

You can run benchmark only tests with:

```
python -m pytest --benchmark-only
```

### Samples

You can run the included tests with:

```
python -m pytest examples
```

## Examples

CUDA Samples rewriten using CUDA Python are found in `examples`.

Custom extra included examples:

- `examples/extra/jit_program_test.py`: Demonstrates the use of the API to compile and
  launch a kernel on the device. Includes device memory allocation /
  deallocation, transfers between host and device, creation and usage of
  streams, and context management.
- `examples/extra/numba_emm_plugin.py`: Implements a Numba External Memory Management
  plugin, showing that this CUDA Python Driver API can coexist with other
  wrappers of the driver API.
