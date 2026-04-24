# `cuda.core`: Pythonic CUDA module

Currently under active development; see [the documentation](https://nvidia.github.io/cuda-python/cuda-core/latest/) for more details.

## Installing

Please refer to the [Installation page](https://nvidia.github.io/cuda-python/cuda-core/latest/install.html) for instructions and required/optional dependencies.

## Developing

This subpackage adheres to the developing practices described in the parent metapackage [CONTRIBUTING.md](https://github.com/NVIDIA/cuda-python/blob/main/CONTRIBUTING.md).

## Debugging

Pass the `pip` / `uv` configuration option `-C="debug=True"` or
`--config-settings="debug=True"` to explicitly to build debuggable binaries.
Debuggable binaries are built by default for editable builds.

Debuggable builds are not supported on Windows.

## Testing

To run these tests:
* `python -m pytest tests/` with editable installations
* `pytest tests/` with installed packages

Alternatively, from the repository root you can use a simple script:

* `./scripts/run_tests.sh core` to run only `cuda_core` tests
* `./scripts/run_tests.sh` to run all package tests (pathfinder → bindings → core)
* `./scripts/run_tests.sh smoke` to run meta-level smoke tests under `tests/integration`
