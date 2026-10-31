# Installation

## Runtime Requirements

`cuda.bindings` supports the same platforms as CUDA. Runtime dependencies are:

* Linux (x86-64, arm64) and Windows (x86-64)
* Python 3.11 - 3.15
* Driver: Linux (525.60.13 or later), Windows (528.33 or later)
* Optionally, NVRTC, nvJitLink, NVVM, nvFatbin, and cuFile (Linux only) from CUDA Toolkit 12.x

These driver versions are the CUDA 12 minor-version compatibility baseline in the
[CUDA 12.9 release notes](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cuda-driver).
Features introduced with newer drivers and PTX compilation can require a newer
driver; see the
[minor-version compatibility limitations](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html#application-considerations-for-minor-version-compatibility).

```{note}
The optional CUDA Toolkit components can be installed via PyPI, Conda, OS-specific package managers, or local installers (as described in the CUDA Toolkit [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) and [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) Installation Guides).
```

The CUDA 12 releases of `cuda-python` are a metapackage that depends directly
on `cuda-bindings`. The examples below use `cuda-python` for existing users;
install `cuda-bindings` directly if you only need the low-level interfaces.

Both main-tree CUDA 12 packages require Python 3.11 or newer.


## Installing from PyPI

```console
$ pip install -U "cuda-python~=12.9.0"
```

Install all optional dependencies with:
```{code-block} shell
pip install -U "cuda-python[all]~=12.9.0"
```

Where the optional dependencies are:

* nvidia-cuda-nvrtc-cu12 (Provides NVRTC shared library)
* nvidia-nvjitlink-cu12>=12.3 (Provides nvJitLink shared library)
* nvidia-cuda-nvcc-cu12 (Provides NVVM shared library)
* nvidia-nvfatbin-cu12 (Provides nvFatbin shared library)
* nvidia-cufile-cu12 (Provides cuFile shared library, Linux only)


## Installing from Conda

```console
$ conda install -c conda-forge cuda-python=12.9
```


## Installing from Source

Start with the repository
[recommended clone](https://github.com/NVIDIA/cuda-python/blob/main/CONTRIBUTING.md#recommended-clone).
On native Windows outside WSL, complete its symlink setup before cloning.
CUDA 12 sources live in `cuda_bindings_12/`; CUDA 13 sources have a separate root.
Both build the same `cuda-bindings` distribution, so use a separate Python
environment for each major.

Versions are derived from reachable `v12.9.*` tags using `setuptools-scm`.
Historical tags from the old maintenance branch may not be ancestors of `main`.
Before a suitable tag is reachable, follow the
[maintenance preview-build guidance](https://github.com/NVIDIA/cuda-python/blob/main/ci/README.md#cuda-bindings-package-root-registry)
in an independent clone; worktrees share tags with the rest of their repository.

### Requirements

* CUDA Toolkit headers[^1]
* CUDA Runtime static library[^2]

[^1]: User projects that `cimport` CUDA symbols in Cython must also use CUDA Toolkit (CTK) types as provided by the `cuda.bindings` major.minor version. This results in CTK headers becoming a transitive dependency of downstream projects through CUDA Python.

[^2]: The CUDA Runtime static library (`libcudart_static.a` on Linux, `cudart_static.lib` on Windows) is part of the CUDA Toolkit. If using conda packages, it is contained in the `cuda-cudart-static` package.

Source builds require that the provided CUDA headers are of the same major.minor version as the `cuda.bindings` you're trying to build. Despite this requirement, note that the minor version compatibility is still maintained. Use the `CUDA_HOME` (or `CUDA_PATH`) environment variable to specify the location of your headers. For example, if your headers are located in `/usr/local/cuda/include`, then you should set `CUDA_HOME` with:

```console
$ export CUDA_HOME=/usr/local/cuda
```

See [Environment Variables](environment_variables.md) for a description of other build-time environment variables.

```{note}
Only `cydriver`, `cyruntime` and `cynvrtc` are impacted by the header requirement.
```


### Editable Install

From the repository root, run:

```console
$ cd cuda_bindings_12
$ python -m pip install -v -e .
```

This installs the CUDA 12 bindings as editable in your current Python environment
for development and testing.
