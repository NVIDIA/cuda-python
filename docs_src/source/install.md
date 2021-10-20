# Installation

```{note} Building from source is required for the EA release of CUDA Python. Look for PyPI and Conda packages soon!
```

## Requirements

CUDA Python is supported on all platforms that CUDA is supported. Specific
dependencies are as follows:

* Driver: Linux (450.80.02 or later) Windows(456.38 or later)
* CUDA Toolkit 11.0 to 11.4 - e.g. 11.4.48
* Cython - e.g. 0.29.21
* Versioneer - e.g. 0.20

## Compilation

To compile the extension in-place, run:

```{code-block} shell
python setup.py build_ext --inplace
```

To compile for debugging the extension modules with gdb, pass the `--debug`
argument to setup.py.

The CUDA location is assumed to be the parent directory of where `cuda-gdb` is
located - to suggest an alternative location, use the `CUDA_HOME` environment
variable, e.g.:

```{code-block} shell
CUDA_HOME=/opt/cuda/11.4 python setup.py <args>
```


## Develop installation

You can use

```{code-block} shell
python setup.py develop
```

to use the module in-place in your current Python environment (e.g. for testing
of porting other libraries to use the binding).

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

