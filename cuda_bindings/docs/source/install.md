# Installation

## Runtime Requirements

`cuda-python` supports the same platforms as CUDA. Runtime dependencies are:

* Driver: Linux (450.80.02 or later) Windows (456.38 or later)
* CUDA Toolkit 12.0 to 12.6

```{note} Only the NVRTC redistributable component is required from the CUDA Toolkit. [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/index.html) Installation Guides can be used for guidance. Note that the NVRTC component in the Toolkit can be obtained via PYPI, Conda or local installers.
```

## Installing from PyPI

```console
$ pip install cuda-python
```

## Installing from Conda

```console
$ conda install -c nvidia cuda-python
```

Conda packages have dependencies to CUDA Toolkit components:

* cuda-cudart (Provides CUDA headers to enable writing NVRTC kernels with CUDA types)
* cuda-nvrtc (Provides NVRTC shared library)

## Installing from Source

### Requirements

* CUDA Toolkit headers[^1]
* [requirements.txt](https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/requirements.txt)

[^1]: User projects that `cimport` CUDA symbols in Cython must also use CUDA Toolkit (CTK) types as provided by the `cuda-python` major.minor version. This results in CTK headers becoming a transitive dependency of downstream projects through CUDA Python.

Source builds require that the provided CUDA headers are of the same major.minor version as the `cuda-python` bindings you're trying to build. Despite of this requirement, note that the minor version compatibility is still maintained. Use the `CUDA_HOME` (or `CUDA_PATH`) environment variable to specify the location of your headers. For example, if your headers are located in `/usr/local/cuda/include`, then you should set `CUDA_HOME` with:

```console
$ export CUDA_HOME=/usr/local/cuda
```

```{note} Only `cydriver`, `cyruntime` and `cynvrtc` are impacted by the header requirement.
```

### In-place

To compile the extension in-place, run:

```console
$ python setup.py build_ext --inplace
```

To compile for debugging the extension modules with gdb, pass the `--debug` argument to setup.py.

### Develop

You can use

```console
$ pip install -e .
```

to install the module as editable in your current Python environment (e.g. for testing of porting other libraries to use the binding).
