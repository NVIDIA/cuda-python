# QA Directory

> **This directory exists only on `ctk-next`.**

---

## Overview

The `ctk-next` branch of `cuda-python-private` is a staging branch for pre-release testing with upcoming CUDA Toolkit (CTK) releases.
It was originally **seeded from the public `cuda-python` repository’s `main` branch** and is periodically updated to stay in sync with it.

Within `ctk-next`, the `qa/` subdirectory contains scripts, helpers, and documentation that are *specific to internal or pre-release QA* activities.
This directory **does not exist in the public `main` branch**.

---

## Relationship between `ctk-next` and `main`

- The `qa/` subdirectory is the only directory unique to `ctk-next`.
  Everything else mirrors the public `main` branch.

- During **pre-CTK-release testing periods**, `ctk-next` may be *ahead of* `main`, containing changes specific to the upcoming CTK version.
  However, it will be **kept closely synchronized** with public `main` by regularly merging updates from the public repository.

- During **non-testing periods**, `ctk-next` will effectively be a clean copy of `main`.

- After each CTK release is publicly posted, we perform a **squash merge** of `ctk-next` back into the public `main` branch, **excluding** the `qa/` directory.

---

## What goes into `ctk-next`?

"Kitpicks" are CTK release candidates, sometimes referred to as, e.g., "13.2 RC 001". They are hosted here:

[**https://cuda-repo.nvidia.com/release-candidates/kitpicks/cuda-r13-2/13.2.0/**](
    https://cuda-repo.nvidia.com/release-candidates/kitpicks/cuda-r13-2/13.2.0/)

A helper script is available for downloading `linux-64`, `linux-aarch64`, and `win-64` release candidates:

`qa/helpers/download_from_kitpicks.py --help`

<img src="ctk-next.drawio.svg" width="400">

### cython-gen

The [`cuda-python-private:cython-gen`](https://github.com/NVIDIA/cuda-python-private/tree/cython-gen) branch contains
custom auto-generation scripts and configuration files for generating `driver`, `runtime`, and `nvrtc` bindings.
It includes scripts for:

* Fetching and processing CUDA headers.
* Generating documentation configurations.
* Creating Python bindings for CUDA driver, runtime, and NVRTC code.

To invoke the top-level script:
```
CUDA_HOME=/usr/local/cuda-13.2 python regenerate.py -o ../ctk-next
```

### cybind

All other library bindings provided by `cuda-bindings` are supported by [`cybind`](https://gitlab-master.nvidia.com/leof/cybind/).
After setting up the environment (see cybind documentation), the `cuda-python` bindings can be generated with:

```
CUDA_PATH=/usr/local/cuda-13.2 python -m cybind -vvv --generate cufile nvjitlink nvml nvvm --output-dir ../ctk-next/cuda_bindings
```

### Manual updates

#### cuda-pathfinder `supported_*.py`

These cuda-pathfinder files may need to be updated:

* `cuda/pathfinder/_headers/supported_nvidia_headers.py`
* `cuda/pathfinder/_dynamic_libs/supported_nvidia_libs.py`

No updates are required **prior** to a CTK release if all `cuda_pathfinder/tests/` pass.
PyPI wheels for new CTK releases may become available only a few days or even weeks after the main CTK release date.
Therefore it is likely that updates to the two files above will be required after the release, in the public repo.

#### `_ptx_to_cuda`

The `_ptx_to_cuda` dictionary in

* `cuda_bindings/cuda/bindings/utils/_ptx_utils.py`

may need to be updated, based on the table shown under the
[PTX Release Notes](http://sw-mobile-docs/CUDA/GPGPU/parallel-thread-execution/index.html#release-notes).

#### CUDA enums

Two files under `cuda/core/experimental/_utils/` need to be updated semi-manually:

* `toolshed/reformat_cuda_enums_as_py.py` → `cuda/core/experimental/_utils/driver_cu_result_explanations.py`
* `toolshed/reformat_cuda_enums_as_py.py` → `cuda/core/experimental/_utils/runtime_cuda_error_explanations.py`

See the instructions in the header of each of the files for details.

---

## How to Merge Public `main` into `ctk-next`

To keep `ctk-next` synchronized with the latest public changes, use the helper script:

```bash
qa/helpers/git_merge_public_main.sh
```

This script fetches and merges the current `main` branch from the public `cuda-python` repository into the private `ctk-next` branch.
It performs a clean merge without having to permanently add a remote.

## How to Squash-Merge ctk-next Back into Public main

When a new CTK release goes live:

1. Wait until the CTK release has been publicly posted.

2. Shortly after, create a PR in the public `cuda-python` repo that squash-merges all `ctk-next` changes into `main`, excluding the `qa/` directory.

For reference, the squash-merge PR for the CTK 13.1 release was:

[**https://github.com/NVIDIA/cuda-python/pull/1315/commits**](
    https://github.com/NVIDIA/cuda-python/pull/1315/commits)

Note the organization into three initial commits (commit messages slightly modernized here):

* `automatic code-gen changes: driver, runtime, nvrtc` (cython-gen)
* `automatic code-gen changes: cufile, nvjitlink, nvvm, nvml` (cybind)
* `accumulated changes from the QA testing period`

This was very helpful for keeping the large-scale changes reasonably easy to review.
