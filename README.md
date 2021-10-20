# CUDA-Python

## Building

### Requirements

Dependencies of the CUDA-Python bindings and some versions that are known to
work are as follows:

* Driver: Linux (450.80.02 or later) Windows(456.38 or later)
* CUDA Toolkit 11.0 to 11.5
* Cython - e.g. 0.29.21

### Compilation

To compile the extension in-place, run:

```
python setup.py build_ext --inplace
```

To compile for debugging the extension modules with gdb, pass the `--debug`
argument to setup.py.


### Develop installation

You can use

```
pip install -e .
```

to install the module  as editible in your current Python environment (e.g. for
testing of porting other libraries to use the binding).


### Build the Docs

```
conda env create -f docs_src/environment-docs.yml
conda activate cuda-python-docs
```
Then compile and install `cuda-python` following the steps above.

```
cd docs_src
make html
open build/html/index.html
```

### Publish the Docs

```
git checkout gh-pages
cd docs_src
make html
cp -a build/html/. ../docs/
```

## Testing

### Requirements

Dependencies of the test execution and some versions that are known to
work are as follows:

* numpy-1.19.5
* matplotlib-3.3.4
* scipy-1.6.3
* pytest-benchmark-3.4.1

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

The included examples are:

- `examples/extra/jit_program_test.py`: Demonstrates the use of the API to compile and
  launch a kernel on the device. Includes device memory allocation /
  deallocation, transfers between host and device, creation and usage of
  streams, and context management.
- `examples/extra/numba_emm_plugin.py`: Implements a Numba External Memory Management
  plugin, showing that this CUDA Python Driver API can coexist with other
  wrappers of the driver API.
