# Installation

## Runtime Requirements

CUDA Python is supported on all platforms that CUDA is supported. Specific
dependencies are as follows:

* Driver: Linux (450.80.02 or later) Windows (456.38 or later)
* CUDA Toolkit 12.0 to 12.5

```{note} Only the NVRTC redistributable component is required from the CUDA Toolkit. [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/index.html) Installation Guides can be used for guidance. Note that the NVRTC component in the Toolkit can be obtained via PYPI, Conda or Local Installer.
```

## Installing from PyPI

```{code-block} shell
pip install cuda-python
```

## Installing from Conda

```{code-block} shell
conda install -c nvidia cuda-python
```

Conda packages are assigned a dependency to CUDA Toolkit:

* cuda-cudart (Provides CUDA headers to enable writting NVRTC kernels with CUDA types)
* cuda-nvrtc (Provides NVRTC shared library)

## Installing from Source

### Build Requirements

* CUDA Toolkit headers
* Cython
* pyclibrary

Remaining build and test dependencies are outlined in [requirements.txt](https://github.com/NVIDIA/cuda-python/blob/main/requirements.txt)

The version of CUDA Toolkit headers must match the major.minor of CUDA Python. Note that minor version compatibility will still be maintained.

During the build process, environment variable `CUDA_HOME` or `CUDA_PATH` are used to find the location of CUDA headers. In particular, if your headers are located in path `/usr/local/cuda/include`, then you should set `CUDA_HOME` as follows:

```
export CUDA_HOME=/usr/local/cuda
```

### In-place

To compile the extension in-place, run:

```{code-block} shell
python setup.py build_ext --inplace
```

To compile for debugging the extension modules with gdb, pass the `--debug`
argument to setup.py.

### Develop

You can use

```{code-block} shell
pip install -e .
```

to install the module as editible in your current Python environment (e.g. for
testing of porting other libraries to use the binding).

## Build the Docs

```{code-block} shell
conda env create -f docs_src/environment-docs.yml
conda activate cuda-python-docs
```
Then compile and install `cuda-python` following the steps above.

```{code-block} shell
cd docs_src
make html
open build/html/index.html
```

### Publish the Docs

```{code-block} shell
git checkout gh-pages
cd docs_src
make html
cp -a build/html/. ../docs/
```
