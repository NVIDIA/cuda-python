# Installation

## Requirements

CUDA Python is supported on all platforms that CUDA is supported. Specific
dependencies are as follows:

* Driver: Linux (450.80.02 or later) Windows (456.38 or later)
* CUDA Toolkit 12.0 to 12.2
* Python 3.8 to 3.11

```{note} Only the NVRTC redistributable component is required from the CUDA Toolkit. [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/index.html) Installation Guides can be used for guidance. Note that the NVRTC component in the Toolkit can be obtained via PiPy, Conda or Local Installer.
```

## Installing from PyPI

```{code-block} shell
pip install cuda-python
```

## Installing from Conda

```{code-block} shell
conda install -c nvidia cuda-python
```

## Installing from Source

### Requirements

Installing from source requires the latest CUDA Toolkit (CTK), matching the major.minor of CUDA Python. The installed package will still be compatible with all minor CTK versions.

Environment variable CUDA_HOME must be set to CTK root directory:
```
export CUDA_HOME=/usr/local/cuda
```

Remaining build and test dependencies are outlined in [requirements.txt](https://github.com/NVIDIA/cuda-python/blob/main/requirements.txt)

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
