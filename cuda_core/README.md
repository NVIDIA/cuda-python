# `cuda.core`: (experimental) pythonic CUDA module

Currently under active development; see [the documentation](https://nvidia.github.io/cuda-python/cuda-core/latest/) for more details.

## Installing

To build from source, just do:
```shell
$ git clone https://github.com/NVIDIA/cuda-python
$ cd cuda-python/cuda_core  # move to the directory where this README locates
$ pip install .
```
For now `cuda-python` is a required dependency.

## Developing

We use `pre-commit` to manage various tools to help development and ensure consistency.
```shell
pip install pre-commit
```

### Code linting

Run this command before checking in the code changes
```shell
pre-commit run -a --show-diff-on-failure
```
to ensure the code formatting is in line of the requirements (as listed in [`pyproject.toml`](./pyproject.toml)).

### Code signing

This repository implements a security check to prevent the CI system from running untrusted code. A part of the
security check consists of checking if the git commits are signed. See
[here](https://docs.gha-runners.nvidia.com/apps/copy-pr-bot/faqs/#why-did-i-receive-a-comment-that-my-pull-request-requires-additional-validation)
and
[here](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification)
for more details, including how to sign your commits.

## Testing

To run these tests:
* `python -m pytest tests/` against editable installations
* `pytest tests/` against installed packages

### Cython Unit Tests

Cython tests are located in `tests/cython` and need to be built. These builds have the same CUDA Toolkit header requirements as [those of cuda.bindings](https://nvidia.github.io/cuda-python/cuda-bindings/latest/install.html#requirements) where the major.minor version must match `cuda.bindings`. To build them:

1. Setup environment variable `CUDA_HOME` with the path to the CUDA Toolkit installation.
2. Run `build_tests` script located in `test/cython` appropriate to your platform. This will both cythonize the tests and build them.

To run these tests:
* `python -m pytest tests/cython/` against editable installations
* `pytest tests/cython/` against installed packages
