# Contributing to CUDA Python

Thank you for your interest in contributing to CUDA Python! Based on the type of contribution, it will fall into two categories:

1. You want to report a bug, feature request, or documentation issue:
    - File an [issue](https://github.com/NVIDIA/cuda-python/issues/new/choose)
    describing what you encountered or what you want to see changed.
    - The NVIDIA team will evaluate the issues and triage them, scheduling
    them for a release. If you believe the issue needs priority attention
    comment on the issue to notify the team.
2. You want to implement a feature, improvement, or bug fix:
   - Before starting work on an existing issue, please comment on the issue to express your interest and wait to be assigned by a maintainer. This helps avoid redundant effort in case the issue is already being worked on by another contributor or an NVIDIA team member.
   - Please refer to each component's guideline:
       - [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/contribute.html)
       - [`cuda.bindings`](https://nvidia.github.io/cuda-python/cuda-bindings/latest/contribute.html)<sup>[1](#footnote1)</sup>
       - [`cuda.pathfinder`](https://nvidia.github.io/cuda-python/cuda-pathfinder/latest/contribute.html)

## Table of Contents

- [Contributing to CUDA Python](#contributing-to-cuda-python)
  - [Table of Contents](#table-of-contents)
  - [Cloning the repository](#cloning-the-repository)
    - [Recommended clone](#recommended-clone)
    - [Updating your clone](#updating-your-clone)
    - [Why the clone matters](#why-the-clone-matters)
    - [Symptoms of a bad clone](#symptoms-of-a-bad-clone)
  - [Development on Windows](#development-on-windows)
    - [Enabling git symlinks](#enabling-git-symlinks)
  - [Type stubs for cuda.core](#type-stubs-for-cudacore)
  - [Pre-commit](#pre-commit)
  - [CodeRabbit](#coderabbit)
    - [Review Guidance](#review-guidance)
    - [Pull Request Reviews](#pull-request-reviews)
  - [Pixi lockfiles](#pixi-lockfiles)
  - [Signing Your Work](#signing-your-work)
  - [Code signing](#code-signing)
  - [Developer Certificate of Origin (DCO)](#developer-certificate-of-origin-dco)
  - [CI infrastructure overview](#ci-infrastructure-overview)
    - [CI Pipeline Flow](#ci-pipeline-flow)
    - [Pipeline Execution Details](#pipeline-execution-details)
    - [Source and Artifact Flow](#source-and-artifact-flow)
    - [Key Infrastructure Details](#key-infrastructure-details)
  - [Code coverage](#code-coverage)


## Cloning the repository

### Recommended clone

**On native Windows (not WSL), first [enable symlink creation](#enabling-git-symlinks),
including the Developer Mode registry check.** Complete that setup before
running the clone command.

From the directory where you want to create the checkout, run:

```console
$ git clone --config core.symlinks=true --config submodule.recurse=true --recurse-submodules https://github.com/NVIDIA/cuda-python.git
$ cd cuda-python
```

This recipe sets up the repository for source builds and development:

- `core.symlinks=true` materializes symlinks during checkout, including the
  build-critical `cuda_core/_build_shared.py`.
- `--recurse-submodules` initializes the CCCL headers needed to build `cuda.core`.
- `submodule.recurse=true` keeps initialized submodules in sync on `git pull`,
  `git checkout`, and `git switch`.
- The clone retains history and release tags for correct package versions.
  Do not add `--depth` or `--no-tags`.

Both configuration settings are stored in the new clone. Next, follow the
build and test instructions for
[`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/install.html),
[`cuda.bindings`](https://nvidia.github.io/cuda-python/cuda-bindings/latest/install.html), or
[`cuda.pathfinder`](https://nvidia.github.io/cuda-python/cuda-pathfinder/latest/install.html).
The bindings `latest` guide covers CUDA 13. For CUDA 12 development, start
with [`cuda_bindings_12/`](cuda_bindings_12/). Both bindings roots require
Python 3.11 or newer; use a separate environment for each CUDA major.

### Updating your clone

From the repository root, run:

```console
$ git pull --recurse-submodules
$ git submodule update --init
```

The recommended clone configures `submodule.recurse=true`, so you can omit
`--recurse-submodules` from `git pull`. The second command initializes any
newly added submodules; it is usually a no-op.

If you work from a fork, periodically fetch release tags from NVIDIA upstream
so package versions stay current:

```console
$ git fetch --tags https://github.com/NVIDIA/cuda-python.git
```

### Why the clone matters

Every package in this repository derives its version from git tags using
[`setuptools-scm`](https://setuptools-scm.readthedocs.io/), so **how you clone
determines whether you can build at all, and whether the version you build is
correct.** Packages select tags for their own release line:

| Package | Tag pattern |
| --- | --- |
| `cuda-bindings` from `cuda_bindings_12/` | `v12.9.*` (e.g. `v12.9.10`) |
| `cuda-bindings` from `cuda_bindings_13/` | `v13.4.*` (e.g. `v13.4.2`) |
| `cuda-python` | The matching bindings selector, chosen with `CUDA_PYTHON_BUILD_MAJOR=12` or `13` (default) |
| `cuda-core` | `cuda-core-v*` (e.g. `cuda-core-v1.1.0`) |
| `cuda-pathfinder` | `cuda-pathfinder-v*` (e.g. `cuda-pathfinder-v1.6.0`) |

Git source builds of `cuda-python` default to CUDA 13. Set
`CUDA_PYTHON_BUILD_MAJOR=12` to select CUDA 12, or `13` explicitly for CUDA 13.
The resolved version determines dependencies: CUDA 12 depends on matching
bindings, while CUDA 13 also depends on Core and Pathfinder. A standalone
sdist retains its version and dependency major without requiring this variable.

The selectors are recorded in the bindings roots' `pyproject.toml` files and
the metapackage's `pyproject.toml`; the metadata check keeps them aligned with
`ci/versions.yml`. All packages read versions from the *repository root*,
rather than their package directory. Builds from a Git checkout therefore
need all of the following:

1. **A real git clone.** Source zips and GitHub "Download ZIP" archives have no
   git metadata and the build fails outright. Tarballs produced by
   `git archive` still resolve a version via the `.git_archival.txt`
   substitutions in `.gitattributes`, but they do not include submodule
   contents, so they cannot compile `cuda.core`.
2. **The full repository**, not just the package subdirectory, because the
   version lookup walks up to the repository root.
3. **Tags, reaching back at least as far as the most recent tag** matching the
   package you are building. `git describe` needs to find that tag; the history
   between it and your checkout must be present too.
4. **The CCCL submodule** when building `cuda.core`. That package compiles
   against the headers pinned in `cuda_core/third_party/cccl`.

Copying CUDA 12 sources into `main` does not make their historical branch tags
reachable. For maintenance preview builds without an appropriate reachable
tag, follow the [SCM bootstrap guidance](ci/README.md#cuda-bindings-package-root-registry)
in a disposable clone; do not create preview tags in a shared worktree repository.

### Symptoms of a bad clone

Incomplete history can fail version detection or produce a plausible but
incorrect version that surfaces later as a dependency-resolution failure:

1. **No matching tags reachable.** Depending on the package's SCM configuration,
   the build can fail or produce a placeholder such as `0.1.dev...`. Fetching
   tags is insufficient if the checkout lacks the ancestry connecting them to
   the build. Check the selected release line, not just whether any tag exists.
2. **Stale tags** (a fork that has not fetched upstream in a while): you get a
   plausible-looking but wrong version, e.g. `13.4.1.dev7+g0d22cb44` when the
   real latest tag is `v13.4.2`. Nothing warns you. Note there is no leading
   `v` — `setuptools-scm` strips the tag prefix.
3. **No git metadata** (source zip): the build fails with
   `LookupError: setuptools-scm was unable to detect version`.

As a last resort — for example when building inside a container that has no git
history — you can bypass the lookup entirely:

```console
$ SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUDA_CORE=1.1.0 pip install ./cuda_core
```

The environment variable is suffixed with the distribution name, uppercased with
hyphens replaced by underscores: `..._FOR_CUDA_BINDINGS`, `..._FOR_CUDA_CORE`,
`..._FOR_CUDA_PATHFINDER`, `..._FOR_CUDA_PYTHON`. Use this only when you
genuinely cannot provide tags; it is not a substitute for a correct clone.


## Transferring fixes between bindings roots

For a handwritten fix that applies to both majors, a path-relative patch can
transfer the source-root portion of a commit. From the repository root, with a
clean worktree, replace `COMMIT` with the reviewed source commit:

```console
$ git format-patch -1 --stdout --relative=cuda_bindings_13 COMMIT -- cuda_bindings_13 > bindings-fix.patch
$ git am --directory=cuda_bindings_12 bindings-fix.patch
```

Inspect the patch before applying it. The path restriction excludes changes
outside the selected root; assess shared files separately. Resolve conflicts
and review the result against the destination API and toolkit, then run the
relevant tests. Swap the two roots for the reverse direction, and remove the
temporary patch when finished. These commands copy handwritten changes;
generated bindings must be regenerated from their generation source for each
toolkit. Record matching changes or a concrete reason they do not apply.

## Development on Windows

This section collects Windows-specific guidance for contributors working
outside of WSL. WSL contributors can follow the Linux flow in the rest of this
document.

### Enabling git symlinks

The repository contains documentation and metadata symlinks, including
`CLAUDE.md`, `cuda_python/README.md`, and the per-package `.git_archival.txt`
files. It also currently tracks the build-critical
`cuda_core/_build_shared.py` as a symlink to the canonical
`cuda_bindings_13/_build_shared.py`. The core path may instead be maintained as a
byte-for-byte identical copy, but the current tree uses a symlink. With
`core.symlinks=false`, Git checks it out as a plain text file containing the
target path, and `cuda-core` source builds and tests fail.

[Activate Developer Mode](https://learn.microsoft.com/en-us/windows/advanced-settings/developer-mode)
so Git can create symlinks without Administrator privileges. Then verify
that Windows recorded Developer Mode in the machine-wide registry:

```console
$ reg.exe query "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock" /v AllowDevelopmentWithoutDevLicense
```

The output must show `AllowDevelopmentWithoutDevLicense` as a `REG_DWORD`
with value `0x1`. If the value is missing or differs, Git may report
`unable to create symlink ...: Permission denied` even when Windows Settings
shows Developer Mode as enabled. In that case, run the following command
once from an Administrator shell:

```console
$ reg.exe add "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock" /v AllowDevelopmentWithoutDevLicense /t REG_DWORD /d 1 /f
```

Then return to [Recommended clone](#recommended-clone).

## Type stubs for cuda.core

`cuda.core` is a PEP 561-compliant package: it ships a `py.typed` marker and
`.pyi` stub files alongside the Cython extensions.  The stubs
are checked into the repository.

**You do not need to run stubgen-pyx manually.**  A pre-commit hook
regenerates the corresponding `.pyi` files automatically when you commit.
The results are then also tested with `mypy`.

A few things to keep in mind:

- **Do not edit `.pyi` files by hand.**  They are regenerated from the Cython
  sources on every commit that touches those sources; manual edits will be
  overwritten.
- **Type annotations belong in the `.pyx`/`.pxd` source.**  stubgen-pyx reads
  Cython type annotations and docstrings to build the stubs, so keeping the
  source well-annotated is the right way to improve stub quality.
- **To run mypy manually (outside of pre-commit)**: `python -m mypy
  --config-file cuda_core/pyproject.toml

## Pre-commit
GitHub Actions runs pre-commit checks on Linux and Windows and checks
documentation links for pull requests. Check failures must be resolved
before merging.

To set yourself up for running pre-commit checks locally and to catch issues before pushing your changes, follow these steps:

* Install pre-commit with: `pip install pre-commit`
* Run this once per checkout: `pre-commit install`
* You can manually check all files at any time by running: `pre-commit run --all-files`

On Windows, these commands work from PowerShell, cmd, or Git Bash.

This command runs all configured hooks (such as linters and formatters) across your repository, letting you review and address issues before committing.

Installing the hook is required, not optional. Some of the automated checks
(the SPDX header updater and the `.pyi` stub generator for `cuda_core`) only
keep the tree consistent if they run on *every* commit. Relying on manual
`pre-commit run --all-files` invocations means these checks can be skipped
between commits, leaving stale headers or out-of-date stubs in the history.
If the hook isn't installed, `pre-commit run` will print a visible warning
reminding you to run `pre-commit install`.

C, C++ and CUDA sources are formatted with `clang-format` using the
[`.clang-format`](.clang-format) at the repository root, which follows PEP 7
except for a 120-column limit. The
pre-commit hook installs its own pinned `clang-format` (from PyPI) into the
hook environment, so no system-wide installation is required. Vendored code
(for example `dlpack.h` and `cuda_core/third_party/`) is excluded in
`.pre-commit-config.yaml`. To format in your editor, install the same version
with `pip install clang-format==23.1.3`.

## CodeRabbit

This section explains how to use CodeRabbit for CUDA Python pull request review.
For complete product documentation, see the
[CodeRabbit documentation](https://docs.coderabbit.ai/).

### Review Guidance

We treat CodeRabbit feedback as review assistance, not as a merge requirement by
itself. Maintainers remain responsible for deciding whether comments are
actionable and whether a pull request has adequate tests and CI coverage.

### Pull Request Reviews

Automatic reviews may be disabled or restricted by the repository
configuration. PR authors and maintainers can request a review explicitly by
posting a comment on the pull request:

```text
@coderabbitai review
```

Use a full review when the pull request should be reviewed again from scratch:

```text
@coderabbitai full review
```

Other useful commands:

- `@coderabbitai help` shows the current command reference.
- `@coderabbitai configuration` shows the active configuration.
- `@coderabbitai summary` can be placed in the pull request description as a
  placeholder for the generated summary.
- `@coderabbitai pause` and `@coderabbitai resume` pause or resume automatic
  review behavior. This is useful when a pull request is still being updated
  frequently and should not be reviewed again until it is ready.
- `@coderabbitai ignore` can be placed in the pull request description to
  disable automatic reviews.

## Pixi lockfiles

The repository checks in a `pixi.lock` next to each `pixi.toml`. Those lockfiles
pin the solved dependency graph used by local pixi workflows and by CI, so they
must stay in sync with their manifests and easy to review.

Contributor expectations:

- If a PR changes dependencies in a `pixi.toml`, update and validate the
  corresponding `pixi.lock` in the same PR. Use the Pixi version pinned in
  `ci/pixi-version.env`; review the affected solution rather than performing
  an unrelated dependency refresh.
- If a PR does not intentionally change pixi dependencies or metadata, do not
  include unrelated lockfile churn. `pixi run` can refresh a stale lockfile
  implicitly; revert that noise unless the refresh is the point of the change.
- If a lockfile changes, the PR description should briefly say why.
- Isolate large dependency refreshes from feature work when possible. Prefer a
  dedicated lockfile-only PR over mixing solver churn into an unrelated change.

Freshness enforcement and automated refreshes are temporarily suspended.
The freshness workflow succeeds with a warning; it does not verify
manifest/lock consistency. The refresh workflow has no schedule, and manual
dispatch reports the suspension without changing dependencies or opening a PR.
Source-build CI uses `PIXI_FROZEN=true` to consume committed lockfiles without
solving them. Necessary dependency changes require manual review and validation.
See the [maintenance pause and restoration criteria](ci/README.md#temporary-pixi-lockfile-maintenance-pause)
before attempting a lockfile repair or refresh.

## Secret Scanning

The `secret-scan-trufflehog` pre-commit hook scans staged files and installs TruffleHog into its own environment on first run, on Linux, macOS, and Windows. If it flags a secret, remove it before committing, or contact a maintainer if it's a false positive. Secrets are also scanned server-side in CI.


## Signing Your Work

Contributions to files licensed under Apache 2.0 must be certified under the
[Developer Certificate of Origin (DCO)](#developer-certificate-of-origin-dco).
Sign off every commit with the `-s` option:

```console
git commit -s -m "Describe your change"
```

Git uses your configured name and email address to add a trailer like this to
the commit message:

```text
Signed-off-by: Your Name <your.email@example.com>
```

Use your real name and an email address associated with your contribution. The
sign-off certifies that you have the right to submit the contribution under the
DCO below. DCO sign-off is separate from the cryptographic commit signing
described in the next section; both requirements apply.


## Code signing

This repository implements a security check to prevent the CI system from running untrusted code. A part of the security check consists of checking if the git commits are signed. Please ensure that your commits are signed [following GitHub’s instruction](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification).


## Developer Certificate of Origin (DCO)
```
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## CI infrastructure overview

The CUDA Python project uses a comprehensive CI pipeline that builds, tests, and releases multiple components across different platforms. This section provides a visual overview of our CI infrastructure to help contributors understand the build and release process.

### CI Pipeline Flow

The CI pipeline diagram is maintained as Mermaid source in [`ci/ci-pipeline.mmd`](ci/ci-pipeline.mmd).

### Pipeline Execution Details

**Parallel Execution**: The CI pipeline leverages parallel execution to optimize build and test times:
- **Build Stage**: Different architectures/operating systems (linux-64, linux-aarch64, win-64) are built in parallel across their respective runners
- **Test Stage**: Different architectures/operating systems/CUDA versions are tested in parallel; documentation preview is also built in parallel with testing

### Source and Artifact Flow

- Pushes to `main` build and test both registered bindings lines and their
  dependent packages from the same source revision.
- Pull requests build affected packages and test their consumers. Unchanged
  wheels and test artifacts can reuse a successful exact-baseline CI run;
  without a complete usable baseline, CI builds and tests everything.
- Bindings tags select their matching bindings root and metapackage, using
  published Pathfinder. They do not require a Core release.
- Other branches retain their own release procedure and compatible tooling.
  Main-tree CI no longer fetches CUDA 12 bindings artifacts from `12.9.x`.

See [Reading CI Jobs and Summaries](ci/README.md#reading-ci-jobs-and-summaries)
for package/major names, local versus published bindings, and reuse details.

### Key Infrastructure Details

- **Self-hosted runners**: Used for Linux builds and GPU testing (more resources, faster builds)
- **GitHub-hosted runners**: Used for Windows builds and general tasks
- **Artifact retention**: Set per workflow for wheels, documentation, and test results
- **Cache retention**: GitHub Cache for build dependencies and environments
- **Security**: All commits must be signed, untrusted code blocked
- **Parallel execution**: Matrix builds across Python versions and platforms
- **Component isolation**: Each component (core, bindings, pathfinder, python) can be built/released independently

## Code coverage

Code coverage reports are produced nightly and posted to [GitHub Pages](https://nvidia.github.io/cuda-python/coverage).

Known limitations: Code coverage is only run on Linux x86_64 with an a100 GPU.  We plan to add more platform and GPU coverage in the future.

---

<a>1</a>: The `cuda-python` meta package shares the same license and the contributing guidelines as those of `cuda-bindings`.
