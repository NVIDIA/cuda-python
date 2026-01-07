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

**Branch naming convention**: For CTK 13.X, create a branch named `cython-gen-next-130X0` (e.g., `cython-gen-next-13020` for CTK 13.2).

**Reference previous versions**: When updating for a new CTK version, archived branches from previous versions are invaluable references:
* Check `archive/cython-gen-next-13010` (or similar) to see how new APIs were added
* Look at `configs/driver/_driver.py` in the previous branch to understand patterns for:
  - `signature_mapping`: Maps API names to parameter handling types (`SigType`)
  - `structure_mapping`: Maps structures with pointer members to parameter handling
  - `cuda_api_introduced_version`: Records the CTK version each API was introduced
* **Important**: Don't delete old branches—archive them (e.g., `archive/cython-gen-next-13020`) for future reference

To invoke the top-level script:
```
CUDA_HOME=/usr/local/cuda-13.2 python regenerate.py -o ../ctk-next
```

**Common issues when updating cython-gen**:
* Missing API signatures: The script will report missing entries in `signature_mapping`. Add them to `configs/driver/_driver.py` (or `runtime/_runtime.py`, `nvrtc/_nvrtc.py`).
* Missing `_ptsz` variants: Some APIs have per-thread stream zero variants (e.g., `cuMemcpyWithAttributesAsync_ptsz`). These also need entries in `cuda_api_introduced_version`.
* Structure mappings: If a structure contains pointer members, add it to `structure_mapping` with the appropriate parameter handling type.

### cybind

All other library bindings provided by `cuda-bindings` are supported by [`cybind`](https://gitlab-master.nvidia.com/leof/cybind/).

**Branch naming convention**: For CTK 13.X, create a branch named `cybind-next-130X0` (e.g., `cybind-next-13020` for CTK 13.2).

**Reference previous versions**: Archived branches from previous versions are invaluable:
* Check `archive/cybind-next-13010` (or similar) to see how headers were updated
* Look at config files (e.g., `cybind/assets/configs/config_cufile.py`) to see version update patterns
* **Important**: Don't delete old branches—archive them (e.g., `archive/cybind-next-13020`) for future reference

**Update process**:
1. Copy new headers from `/usr/local/cuda-13.X/include/` (or subdirectories) to `cybind/assets/headers/<library>/13.X.0/`
2. Update version lists in config files (e.g., `cybind/assets/configs/config_cufile.py`)
3. For `cufile.h`, manual docstring patches may be required (see previous version branches for examples)

After setting up the environment (see cybind documentation), the `cuda-python` bindings can be generated with:

```
CUDA_PATH=/usr/local/cuda-13.2 python -m cybind -vvv --generate cufile nvjitlink nvml nvvm --output-dir ../ctk-next/cuda_bindings
```

**Common issues when updating cybind**:
* Docstring parsing errors: Some headers may need manual patches to ensure proper docstring formatting (e.g., adding missing `@param` tags)
* Missing version entries: Don't forget to update the `versions` list in each library's config file

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

Two files under `cuda/core/_utils/` need to be updated semi-manually:

* `toolshed/reformat_cuda_enums_as_py.py` → `cuda/core/_utils/driver_cu_result_explanations.py`
* `toolshed/reformat_cuda_enums_as_py.py` → `cuda/core/_utils/runtime_cuda_error_explanations.py`

**Update process**:
1. Run the script on the appropriate header:
   ```bash
   python toolshed/reformat_cuda_enums_as_py.py /usr/local/cuda-13.2/include/driver_types.h > /tmp/runtime_enum_output.py
   python toolshed/reformat_cuda_enums_as_py.py /usr/local/cuda-13.2/include/cuda.h > /tmp/driver_enum_output.py
   ```
2. Replace the dictionary in each target file with the generated output (preserve the header comments)
3. Update the CUDA Toolkit version number in the comment (e.g., from `v13.1.0` to `v13.2.0`)
4. Verify the files compile: `python -m py_compile <file>`

**Note**: The script automatically detects which enum it's parsing based on the header file. The generated output includes only the dictionary—you need to preserve the file header manually.

See the instructions in the header of each of the files for additional details.

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

---

## Tips for Future CTK Updates

### Reference Previous Version Branches

**Highly recommended**: When updating for a new CTK version, archived branches from previous versions are invaluable references:

* **cython-gen**: Check `archive/cython-gen-next-13010` (or similar) to see:
  - How new APIs were added to `signature_mapping`
  - How `_ptsz` variants were handled
  - Patterns for `structure_mapping` and `cuda_api_introduced_version`

* **cybind**: Check `archive/cybind-next-13010` (or similar) to see:
  - Which headers were updated
  - Manual patches that were applied (e.g., `cufile.h` docstring fixes)
  - Config file version update patterns

**Branch archiving**: After completing a CTK release:
1. Archive the working branches instead of deleting them
2. Use naming like `archive/cython-gen-next-13020` and `archive/cybind-next-13020`
3. These archived branches serve as templates for future updates

### Common Workflow Patterns

1. **cython-gen updates**:
   - Run `regenerate.py` to identify missing configurations
   - Reference previous version branch to understand patterns
   - Add missing entries to config files
   - Re-run until generation succeeds

2. **cybind updates**:
   - Copy headers from new CTK installation
   - Update config file version lists
   - Apply any necessary manual patches (check previous version)
   - Run generator and commit results

3. **Enum updates**:
   - Run reformat script on new headers
   - Replace dictionaries and update version numbers
   - Verify compilation and test health checks pass
