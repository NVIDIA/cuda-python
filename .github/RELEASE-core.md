<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# cuda.core Release Process

This document covers the `cuda.core` release process. For other packages:
`cuda-bindings` and `cuda-python` involve a private repository and are not
documented here; `cuda-pathfinder` is largely automated by the
[release-cuda-pathfinder.yml](workflows/release-cuda-pathfinder.yml)
workflow.

Each section below provides detailed guidance for a step in the
[Release Checklist](ISSUE_TEMPLATE/release_checklist.yml). To start a
release, create a new issue from that template and work through it item by
item, referring back here as needed.

---

## File an internal nvbug

Create an nvbug to request that SWQA begin post-release validation.  Issues
identified by that process are typically addressed in a patch release.  To find
the template, search for a previous release's nvbug (e.g. by title "Release of
cuda.core") and create a new bug from the same template.

Example:

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

Update the version, SW combinations (check with the release owner), and
platforms as appropriate for each release.

---

## Check (or update if needed) the dependency requirements

Review `cuda_core/pyproject.toml` and verify that all dependency
requirements are current.

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

2. Wait for the workflow to complete.

3. Verify the TestPyPI upload by installing and running tests from a
   checked-out copy of the repository:

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
(not from PyPI). There are three approaches to updating the feedstock,
from least effort to most control.

### Approach A: Wait for the bot

The `regro-cf-autotick-bot` periodically scans for new releases and opens
a PR automatically. If nothing has changed in the build requirements, the
bot's PR may be sufficient — review it and ask a feedstock maintainer
to merge. However, the bot only
updates the version and sha256. If build dependencies, import paths, or
other recipe fields have changed, the bot's PR will be incomplete and CI
will fail.

### Approach B: Request a bot update

If the bot hasn't opened a PR, you can request one explicitly. Go to the
feedstock's Issues tab and create a new "Bot commands" issue:

- **Title**: `@conda-forge-admin, please update version`
- **Body**: (leave empty)

This triggers the bot to create a version-bump PR. As with approach A,
review the PR and push additional fixes if needed.

### Approach C: Manual PR

For full control — or when the bot's PR needs extensive fixes — open a
PR manually from a fork.

**Fork and clone** (one-time setup):

```bash
gh repo fork conda-forge/cuda-core-feedstock --clone
cd cuda-core-feedstock
```

**Create a branch and edit `recipe/meta.yaml`:**

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

4. **Host dependencies**: Ensure all build-time dependencies are listed.
   For example, v0.6.0 added a Cython C++ dependency on `nvrtc.h`,
   requiring `cuda-nvrtc-dev` in both `host` requirements and
   `ignore_run_exports_from`.

5. **Test commands and descriptions**: Update any import paths or
   descriptions that changed (e.g. `cuda.core.experimental` ->
   `cuda.core`).

**Open a PR:**

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

### Notes

The feedstock CI (Azure Pipelines) triggers automatically on the PR.
Monitor it for build failures — common issues include missing build-time
header dependencies. Feedstock maintainers (listed in
`recipe/meta.yaml` under `extra.recipe-maintainers`) can merge the PR.

---

## Post-release QA

*TBD*

---

## Finalize the announcement update

The release workflow creates a draft GitHub Release. To publish it:

1. Go to the repository on GitHub, click **Tags**, then switch to the
   **Releases** tab.
2. Find the draft release for the new tag and click **Edit**.
3. Copy the body from a previous release as a starting point. It
   typically links to the release notes in the documentation (e.g.
   `https://nvidia.github.io/cuda-python/cuda-core/latest/release/0.6.0-notes.html`).
4. Update the link and any version references, then click
   **Publish release**.

---

## Send out the announcement internally

The release owner will prepare and send the announcement.

---

## Send out the announcement externally (GitHub Release -> Announcement)

*TBD*
