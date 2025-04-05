# Installation

## Runtime Requirements

`cuda.bindings` supports the same platforms as CUDA. Runtime dependencies are:

* Driver: Linux (450.80.02 or later) Windows (456.38 or later)
* CUDA Toolkit 12.x

```{note}
Only the NVRTC and nvJitLink redistributable components are required from the CUDA Toolkit, which can be obtained via PyPI, Conda, or local installers (as described in the CUDA Toolkit [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) and [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) Installation Guides).
```

Starting from v12.8.0, `cuda-python` becomes a meta package which currently depends only on `cuda-bindings`; in the future more sub-packages will be added to `cuda-python`. In the instructions below, we still use `cuda-python` as example to serve existing users, but everything is applicable to `cuda-bindings` as well.


## Installing from PyPI

```console
$ pip install -U cuda-python
```

Install all optional dependencies with:
```{code-block} shell
pip install -U cuda-python[all]
```

Where the optional dependencies are:

* nvidia-cuda-nvrtc-cu12 (Provides NVRTC shared library)
* nvidia-nvjitlink-cu12>=12.3 (Provides nvJitLink shared library)
* nvidia-cuda-nvcc-cu12 (Provides NVVM shared library)


## Installing from Conda

```console
$ conda install -c conda-forge cuda-python
```


## Installing from Source

### Requirements

* CUDA Toolkit headers[^1]
* static CUDA runtime[^2]

[^1]: User projects that `cimport` CUDA symbols in Cython must also use CUDA Toolkit (CTK) types as provided by the `cuda.bindings` major.minor version. This results in CTK headers becoming a transitive dependency of downstream projects through CUDA Python.

[^2]: The static CUDA runtime (`libcudart_static.a` on Linux, `cudart_static.lib` on Windows) is part of CUDA Toolkit. If CUDA is installed from conda, it is contained in the `cuda-cudart-static` package.

Source builds require that the provided CUDA headers are of the same major.minor version as the `cuda.bindings` you're trying to build. Despite this requirement, note that the minor version compatibility is still maintained. Use the `CUDA_HOME` (or `CUDA_PATH`) environment variable to specify the location of your headers. For example, if your headers are located in `/usr/local/cuda/include`, then you should set `CUDA_HOME` with:

```console
$ export CUDA_HOME=/usr/local/cuda
$ export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
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
