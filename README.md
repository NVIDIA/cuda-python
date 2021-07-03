# Copyright 2021 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

# CUDA-Python

## Building

### Requirements

Dependencies of the CUDA-Python bindings and some versions that are known to
work are as follows:

* CUDA Toolkit 11.4 - e.g. 11.4.48
* Cython - e.g. 0.29.21

### Compilation

To compile the extension in-place, run:

```
python setup.py build_ext --inplace
```

To compile for debugging the extension modules with gdb, pass the `--debug`
argument to setup.py.

The CUDA location is assumed to be the parent directory of where `cuda-gdb` is
located - to suggest an alternative location, use the `CUDA_HOME` environment
variable, e.g.:

```
CUDA_HOME=/opt/cuda/11.4 python setup.py <args>
```


### Develop installation

You can use

```
python setup.py develop
```

to use the module in-place in your current Python environment (e.g. for testing
of porting other libraries to use the binding).


## Testing

### Requirements

Dependencies of the test execution and some versions that are known to
work are as follows:

* numpy-1.19.5
* numba-0.53.1
* matplotlib-3.3.4
* scipy-1.6.3
* pytest-benchmark-3.4.1

### Unit-tests

You can run the included tests with:

```
pytest
```

### Samples

You can run the included tests with:

```
pytest examples
```

### Benchmark

You can run benchmark only tests with:

```
pytest --benchmark-only
```

## Examples

The included examples are:

- `examples/extra/jit_program.py`: Demonstrates the use of the API to compile and
  launch a kernel on the device. Includes device memory allocation /
  deallocation, transfers between host and device, creation and usage of
  streams, and context management.
- `examples/extra/numba_emm_plugin.py`: Implements a Numba External Memory Management
  plugin, showing that this CUDA Python Driver API can coexist with other
  wrappers of the driver API.
