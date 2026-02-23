<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Release Process

This document provides detailed guidance for each step of the
[Release Checklist](ISSUE_TEMPLATE/release_checklist.yml). To start a
release, create a new issue from that template and work through it item by
item, referring back here as needed.

---

## File an internal nvbug

Create an nvbug from the SWQA template to request pre-release validation.
To find the template, search for a previous release's nvbug (e.g. by
title "Release of cuda.core") and create a new bug from the same template.

Example (from the cuda.core v0.6.0 release,
[nvbug 5910741](https://nvbugspro.nvidia.com/bug/5910741)):

> **Title:** Release of cuda.core v0.6.0
>
> **Description:**
>
> Requesting SWQA validation for the cuda.core v0.6.0 release. Please test
> the following SW combinations on all listed platforms and report any
> issues found.
>
> **SW Combinations**
> - cuda.core 0.6.0 / cuda.bindings 12.9 / CTK 12.9 / CUDA 12.9 driver
> - cuda.core 0.6.0 / cuda.bindings 13.0 / CTK 13.0 / CUDA 13.0 driver
> - cuda.core 0.6.0 / cuda.bindings 13.1 / CTK 13.1 / CUDA 13.1 driver
>
> **Platforms**
> - Linux x86-64
> - Linux arm64
> - Windows x86-64 (TCC and WDDM)
> - WSL
>
> **Test Plan**
>
> Functional tests as described in the cuda.core test plan.
>
> **Release Milestones**
> - Pre-release QA (this request)
> - GitHub release tag and posting
> - PyPI wheel upload
> - Post-release validation

**How to determine the SW combinations:**

- **cuda.core version**: The version being released.
- **cuda.bindings, CTK, and CUDA driver versions**: Check with the release owner.

Update the version, SW combinations, and platforms as appropriate for each
release.

---

## Check (or update if needed) the dependency requirements

Review `cuda_core/pyproject.toml` and verify the following are current:

- `requires-python` — supported Python version range
- `dependencies` — runtime dependencies (e.g. `numpy`)
- `[project.optional-dependencies]` — `cuda-bindings` version pins for
  `cu12` / `cu13` extras
- `[build-system] requires` — Cython and setuptools version pins
- `[dependency-groups]` — test dependencies (`ml-dtypes`, `cupy`,
  `cuda-toolkit` version pins)
- Python version classifiers in `[project]`

---

## Finalize the doc update, including release notes

Review every PR included in the release. For each one, check whether new
functions, classes, or features were added and whether they have complete
docstrings. Add or edit docstrings as needed — touching docstrings and
type annotations in code is OK during code freeze.

Write the release notes in `cuda_core/docs/source/release/`. Look at
historical release notes for guidance on format and structure. Balance all
entries for length, specificity, tone, and consistency. Highlight a few
notable items in the highlights section, keeping their full entries in the
appropriate sections below.

---

## Update the docs for the new version

Add the new version to
`cuda_core/docs/nv-versions.json`. This file drives the version
switcher on the documentation site. Add an entry for the new version
after `"latest"`, following the existing pattern. The docs themselves are
built and deployed automatically by the release workflow.

---

## Create a public release tag

**Warning:** Pushing a tag is a potentially irrevocable action.  Be absolutely
certain the tag points to the correct commit before pushing.

Tags should be GPG-signed. The tag name format is `cuda-core-v<VERSION>`
(e.g. `cuda-core-v0.6.0`). The tag must point to a commit on `main`.

```bash
git checkout main
git pull origin main
git tag -s cuda-core-v0.6.0 -m "cuda-core v0.6.0"
git push origin cuda-core-v0.6.0
```

---

## Wait for the tag-triggered CI run to complete

Pushing the tag triggers a CI run automatically. Monitor it in the
**Actions** tab on GitHub.

- **The docs build is expected to fail** on tag-triggered runs. This is
  normal — docs are built during the release workflow instead.
- **All CI tests should succeed.** If any fail, investigate and rerun as
  needed.
- Note the **run ID** of the successful tag-triggered run. The release
  workflow can auto-detect it from the tag, but you can also provide it
  explicitly.

---

## Upload wheels to PyPI

This is a two-stage process: first publish to TestPyPI, verify, then
publish to PyPI.

### Stage 1: TestPyPI

1. Go to **Actions > CI: Release** and run the workflow with:
   - **Component**: `cuda-core`
   - **The release git tag**: `cuda-core-v0.6.0`
   - **The GHA run ID that generated validated artifacts**: This is the
     run ID of the successful tag-triggered CI run from the previous step.
     You can find it in the URL when viewing the run in the Actions tab
     (e.g. `https://github.com/NVIDIA/cuda-python/actions/runs/123456789`
     — the run ID is `123456789`).
   - **build-ctk-ver**: the `cuda.build.version` from
     [`ci/versions.yml`](../ci/versions.yml) (e.g. `13.1.1`)
   - **Which wheel index to publish to**: `testpypi`

2. Wait for the workflow to complete. The docs build step will fail on
   forks — this is expected and does not block the wheel upload.

3. Verify the TestPyPI upload by installing and running tests locally:

   ```bash
   pip install -i https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ \
       cuda-core==0.6.0
   cd cuda_core/tests && pytest
   ```

### Stage 2: PyPI

Once TestPyPI verification passes, rerun the same workflow with:
- **Which wheel index to publish to**: `pypi`

After completion, verify:

```bash
pip install cuda-core==0.6.0
```

---

## Update the conda recipe & release conda packages

The conda-forge feedstock builds from the GitHub Release source archive
(not from PyPI). The bot (`regro-cf-autotick-bot`) does not always pick up
new releases automatically, so you may need to open the PR manually.

### Fork and clone the feedstock

```bash
gh repo fork conda-forge/cuda-core-feedstock --clone
cd cuda-core-feedstock
```

Optional: Set up remotes so your fork is named after your GitHub username:

```bash
git remote rename origin <your-github-username>
git remote add origin https://github.com/conda-forge/cuda-core-feedstock.git
git fetch origin
```

### Update `recipe/meta.yaml`

Create a branch and edit `recipe/meta.yaml`:

```bash
git checkout -b update-v0.6.0 origin/main
```

Update the following fields:

1. **`version`**: Set to the new version (e.g. `0.6.0`).
2. **`number`** (build number): Reset to `0` for a new version.
3. **`sha256`**: The SHA-256 of the source archive from the GitHub
   Release. Download it and compute the hash:

   ```bash
   curl -sL https://github.com/NVIDIA/cuda-python/releases/download/cuda-core-v0.6.0/cuda-python-cuda-core-v0.6.0.tar.gz \
       | sha256sum
   ```

4. **Host dependencies**: Ensure all headers needed at build time are
   listed. For example, v0.6.0 added a Cython C++ dependency on
   `nvrtc.h`, requiring `cuda-nvrtc-dev` to be added to both `host`
   requirements and `ignore_run_exports_from`.

5. **Test commands and descriptions**: Update any import paths or
   descriptions that changed (e.g. `cuda.core.experimental` ->
   `cuda.core`).

### Open a PR

```bash
git add recipe/meta.yaml
git commit -m "Update cuda-core to 0.6.0"
git push <your-github-username> update-v0.6.0

gh pr create \
    --repo conda-forge/cuda-core-feedstock \
    --head <your-github-username>:update-v0.6.0 \
    --title "Update cuda-core to 0.6.0" \
    --body "Update cuda-core to version 0.6.0."
```

The feedstock CI (Azure Pipelines) triggers automatically on the PR.
Monitor it for build failures — common issues include missing
build-time header dependencies (see host dependencies above).

---

## Post-release QA

*TBD*

---

## Finalize the announcement update

*TBD*

---

## Send out the announcement internally

*TBD*

---

## Send out the announcement externally (GitHub Release -> Announcement)

*TBD*
