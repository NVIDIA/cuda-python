# CUDA-Python

## Building

### Requirements

Dependencies of the CUDA-Python bindings and some versions that are known to
work are as follows:

* Driver: Linux (450.80.02 or later) Windows(456.38 or later)
* CUDA Toolkit 11.0 to 11.8
* Cython - e.g. 0.29.21

### Installing

Refer to documentation for installation options and requirements: [nvidia.github.io/cuda-python/](https://nvidia.github.io/cuda-python/install.html)

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
