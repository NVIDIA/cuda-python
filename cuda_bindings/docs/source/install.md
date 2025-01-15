# Installation

## Runtime Requirements

`cuda.bindings` supports the same platforms as CUDA. Runtime dependencies are:

* Driver: Linux (450.80.02 or later) Windows (456.38 or later)
* CUDA Toolkit 12.x

```{note}
Only the NVRTC and nvJitLink redistributable components are required from the CUDA Toolkit, which can be obtained via PyPI, Conda, or local installers (as described in the CUDA Toolkit [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) and [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) Installation Guides).
```

## Installing from PyPI

```console
$ pip install cuda-python
```

## Installing from Conda

```console
$ conda install -c conda-forge cuda-python
```

## Installing from Source

### Requirements

* CUDA Toolkit headers[^1]

[^1]: User projects that `cimport` CUDA symbols in Cython must also use CUDA Toolkit (CTK) types as provided by the `cuda.bindings` major.minor version. This results in CTK headers becoming a transitive dependency of downstream projects through CUDA Python.

Source builds require that the provided CUDA headers are of the same major.minor version as the `cuda.bindings` you're trying to build. Despite this requirement, note that the minor version compatibility is still maintained. Use the `CUDA_HOME` (or `CUDA_PATH`) environment variable to specify the location of your headers. For example, if your headers are located in `/usr/local/cuda/include`, then you should set `CUDA_HOME` with:

```console
$ export CUDA_HOME=/usr/local/cuda
```

See [Environment Variables](environment_variables.md) for a description of other build-time environment variables.


```{note}
Only `cydriver`, `cyruntime` and `cynvrtc` are impacted by the header requirement.
```

### Editable Install

You can use

```console
$ pip install -v -e .
```

to install the module as editable in your current Python environment (e.g. for testing of porting other libraries to use the binding).
