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
    - [Fixing an existing clone](#fixing-an-existing-clone)
    - [Symptoms of a bad clone](#symptoms-of-a-bad-clone)
  - [Development on Windows](#development-on-windows)
    - [Enabling git symlinks](#enabling-git-symlinks)
    - [Pre-commit lychee workaround](#pre-commit-lychee-workaround)
  - [Type stubs for cuda.core](#type-stubs-for-cudacore)
  - [Pre-commit](#pre-commit)
  - [Release notes](#release-notes)
    - [Writing a release note](#writing-a-release-note)
    - [Known issues](#known-issues)
    - [How release notes are assembled](#how-release-notes-are-assembled)
  - [Pixi lockfiles](#pixi-lockfiles)
  - [Secret Scanning](#secret-scanning)
  - [Signing Your Work](#signing-your-work)
  - [Code signing](#code-signing)
  - [Developer Certificate of Origin (DCO)](#developer-certificate-of-origin-dco)
  - [CI infrastructure overview](#ci-infrastructure-overview)
    - [CI Pipeline Flow](#ci-pipeline-flow)
    - [Pipeline Execution Details](#pipeline-execution-details)
    - [Branch-specific Artifact Flow](#branch-specific-artifact-flow)
      - [Main Branch](#main-branch)
      - [Backport Branches](#backport-branches)
    - [Key Infrastructure Details](#key-infrastructure-details)
  - [Code coverage](#code-coverage)


## Cloning the repository

> **Windows contributors (not WSL):** configure Git for symlinks *before*
> cloning, or the shared PEP 517 build-hook file lands as a text stub instead
> of a working symlink. See [Enabling git symlinks](#enabling-git-symlinks)
> under Development on Windows.

Every package in this repository derives its version from git tags using
[`setuptools-scm`](https://setuptools-scm.readthedocs.io/), so **how you clone
determines whether you can build at all, and whether the version you build is
correct.** Each package matches its own tag prefix:

| Package | Tag pattern |
| --- | --- |
| `cuda-bindings`, `cuda-python` | `v*` (e.g. `v13.4.2`) |
| `cuda-core` | `cuda-core-v*` (e.g. `cuda-core-v1.1.0`) |
| `cuda-pathfinder` | `cuda-pathfinder-v*` (e.g. `cuda-pathfinder-v1.6.0`) |

Each package sets `root = ".."` in its `[tool.setuptools_scm]` table, meaning the
version is read from the *repository root* rather than the package directory. A
working build therefore needs all of the following:

1. **A real git clone.** Source zips and GitHub "Download ZIP" archives have no
   git metadata and the build fails outright. (Tarballs produced by
   `git archive` do work, thanks to the `.git_archival.txt` substitutions
   configured in `.gitattributes`.)
2. **The full repository**, not just the package subdirectory, because the
   version lookup walks up to the repository root.
3. **Tags, reaching back at least as far as the most recent tag** matching the
   package you are building. `git describe` needs to find that tag; the history
   between it and your checkout must be present too.

### Recommended clone

The default `git clone` gives you everything you need:

```console
$ git clone https://github.com/NVIDIA/cuda-python.git
```



### Fixing an existing clone

If you already have a shallow clone:

```console
$ git fetch --unshallow --tags
```

If you are working from a personal fork, your fork's tags stop tracking upstream
the moment new releases are cut, which silently yields a stale version. Fetch
tags from upstream directly:

```console
$ git remote add upstream https://github.com/NVIDIA/cuda-python.git
$ git fetch --tags upstream
```

Keep doing this periodically — a fork that was correct when you created it will
drift.

### Symptoms of a bad clone

Only case 3 below reports an error. The first two fail *silently*, producing a
wrong version that surfaces much later as a confusing dependency-resolution or
version-check failure:

1. **No tags reachable.** The build succeeds and produces a version starting at
   `0.1.dev`: a `--depth 1` clone yields `0.1.dev1+g0d22cb444`, a full clone made
   with `--no-tags` yields `0.1.dev2114+g0d22cb444`. Installing `cuda-python`
   built this way then fails, because its `install_requires` pins
   `cuda-bindings` to that same bogus version.
2. **Stale tags** (a fork that has not fetched upstream in a while): you get a
   plausible-looking but wrong version, e.g. `13.0.4.dev650+g0d22cb44` when the
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


## Development on Windows

This section collects the Windows-specific setup a contributor needs when
working outside of WSL. WSL contributors can follow the Linux flow in the rest
of this document.

### Enabling git symlinks

The `cuda_core` PEP 517 backend shares source-of-truth helper files with
`cuda_bindings` via symbolic links. Git materializes symlinks by default on
Linux and macOS, but on Windows it needs to be configured before cloning,
otherwise the "symlinks" land in your working tree as plain text files that
contain the target path — enough to look right in `git status`, but not enough
to actually build.

1. **[Activate Developer Mode](https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development#activate-developer-mode)**
   so Git can create symlinks without Administrator privileges.

2. **Enable Git symlink support globally** so newly-cloned repositories inherit
   the setting:

   ```console
   $ git config --global core.symlinks true
   ```

Then clone as usual (see [Cloning the repository](#cloning-the-repository)).

If you already cloned without these settings, note that `git clone` probes
symlink support at clone time and writes `core.symlinks=false` into the
repo-local config when the probe fails. Repo-local config overrides
`--global`, so you must clear it *inside the existing clone* — the global
setting alone won't take effect:

```console
$ git config core.symlinks true          # no --global — clears the repo-local override
$ git rm --cached cuda_core/_build_shared.py
$ git checkout HEAD -- cuda_core/_build_shared.py
```

In practice, deleting the checkout and re-cloning after the two steps at
the top of this section (Developer Mode + `git config --global core.symlinks
true`) is usually simpler and less error-prone than repairing an existing
clone in place.

### Pre-commit lychee workaround

For development on Windows (not WSL), the `lychee` pre-commit task will not
work when running `pre-commit run --all-files`. This problem does not occur
if you install the pre-commit hook and run it automatically as part of your
`git commit` workflow. To resolve this, you can either:

1. Run `pre-commit` in Git Bash, rather than directly in PowerShell or cmd

2. Skip it by setting the environment variable `SKIP` to `lychee`. This would
   be `$env:SKIP = "lychee"` in PowerShell or `set SKIP=lychee` in cmd.


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
This project uses [pre-commit.ci](https://pre-commit.ci/) with GitHub Actions. All pull requests are automatically checked for pre-commit compliance, and any pre-commit failures will block merging until resolved.

To set yourself up for running pre-commit checks locally and to catch issues before pushing your changes, follow these steps:

* Install pre-commit with: `pip install pre-commit`
* Run this once per checkout: `pre-commit install`
* You can manually check all files at any time by running: `pre-commit run --all-files`

This command runs all configured hooks (such as linters and formatters) across your repository, letting you review and address issues before committing.

Installing the hook is required, not optional. Some of the automated checks
(the SPDX header updater and the `.pyi` stub generator for `cuda_core`) only
keep the tree consistent if they run on *every* commit. Relying on manual
`pre-commit run --all-files` invocations means these checks can be skipped
between commits, leaving stale headers or out-of-date stubs in the history.
If the hook isn't installed, `pre-commit run` (and CI) will print a visible
warning reminding you to run `pre-commit install`.

Windows contributors: see [Pre-commit lychee workaround](#pre-commit-lychee-workaround) under Development on Windows.

## Release notes

Release notes are one small YAML file per change, kept next to the code in the
PR that makes the change. (The design is inspired by
[reno](https://docs.openstack.org/reno/latest/), but behaves quite differently.)
All four packages work this way: `cuda-bindings` and
`cuda-python` (13.5.0 and later), `cuda-core` (1.3.0 and later) and
`cuda-pathfinder` (1.9.0 and later), with notes in
`<package>/releasenotes/`.  Earlier releases keep their hand-written pages
under `<package>/docs/source/release/`.

### Writing a release note

Every pull request that changes the sources of one of these packages needs a
release note for it, or an edit to an existing one (a PR touching multiple
packages needs one for each).  Changes to docs, tests, examples, and CI do not
need one. A CI check enforces this. If a PR touches the sources but truly needs
no note, a maintainer can apply the `skip-release-note` label.

Create a note with the `add_note.py` script. It can be run from any directory,
and takes the package and a few words describing the change:

```bash
python toolshed/add_note.py cuda-core "Fix the thing" [--edit]
```

The package is one of `cuda-bindings`, `cuda-core`, `cuda-pathfinder` and
`cuda-python` (the `cuda-` prefix is optional). The new file is created in
`<package>/releasenotes/`. `--edit` opens it in `$VISUAL` or `$EDITOR`.

Edit the new file: keep the sections that apply and delete the rest, and replace
the `TODO` text. Each entry is reStructuredText, written for users of the
package. A pre-commit hook checks the notes for obvious mistakes, such as a
misspelled section or leftover `TODO` text.

| Section | Use for |
|---|---|
| `features` | New functionality, including experimental APIs (say so in the text) |
| `upgrade` | Changes users may need to act on, including behavior changes |
| `deprecations` | Newly deprecated functionality |
| `critical` | Critical issues users must know about |
| `security` | Security issues |
| `fixes` | Bug fixes |
| `issues` | Known issues (see below) |
| `other` | Anything else |
| `prelude` | An introduction to a release's notes. Useful to highlight important themes in a release. |

Some details:

* You do not need to add a link to the pull request: one is added automatically,
  taken from the `(#N)` in the squash-merge commit that introduced the note.
* To name the pull request(s) yourself instead, write `(#1234)` or
  `(#1234, #1235)` anywhere in an entry, usually at the end. Each marker is
  replaced with links, and the automatic lookup is skipped for that entry. This
  is rarely needed. It is for a note whose file is added by a different PR than
  the one that made the change, for example a note written afterwards or notes
  moved in from another place. The marker can go anywhere in ordinary text. A
  `(#N)` inside backquoted text, such as an inline literal, is left alone, so
  it can be used in an example. Do not put a marker inside a literal block, or
  the rows of a table or other directive content, where the link text could
  break the markup.
* Do not rename or move a note after it is merged.
* Which releases list a note is decided from git, so a note shows up in a local
  docs build only once it is committed.
* You can fix a note that is already part of a release (a typo, or a clearer
  wording). The docs are published from `main`, and every page shows a note's
  current text, so the fix appears on the pages of the releases that list it
  the next time the docs are published. Editing a note never moves it to
  another release.

### Known issues

Unlike the other sections, which appear only on the release that first contains
them, known issues are listed on **every** release until they are resolved. To
record one, add an `issues` entry to a note. To resolve it, delete the entry (or
the whole note, if it has nothing else) in the PR that fixes the problem, and
describe the fix in a new `fixes` entry. The pages of releases that were already
tagged keep listing the issue, with the text it had when its note was deleted.

### How release notes are assembled

The docs build generates a page for every release from the notes in git,
with a final "In development" page for notes that are not in any release yet.
Only plain `vX.Y.Z` tags count as releases. Pre-release tags (`rc`, `a`, `b`)
are ignored, and their notes roll into the final release.

* A minor release (`X.Y.0`) lists the notes added since the previous minor
  release.
* A patch release lists the notes added since the previous release of the same
  `X.Y` line.
* A note that is backported appears on the patch release page and again on the
  page of the next minor release that contains it. This is intentional.

Building the docs needs the history of the checked-out commit and of every release
tag (no branches), and the build fails on a shallow clone rather than produce incomplete
pages. In CI, the `.github/actions/fetch-release-history` action does this. The
rules are documented in `cuda_python/docs/exts/release_notes.py`.

## Pixi lockfiles

The repository checks in a `pixi.lock` next to each `pixi.toml`. Those lockfiles
pin the solved dependency graph used by local pixi workflows and by CI, so they
must stay in sync with their manifests and easy to review.

Contributor expectations:

- If a PR changes a `pixi.toml`, update the corresponding `pixi.lock` in the
  same PR. Regenerating one lockfile:

  ```console
  $ pixi lock --manifest-path cuda_core
  ```

  Use `--manifest-path .` for the repository-root environment. Regenerate with
  the pixi version pinned in `ci/pixi-version.env`: different pixi versions
  write different canonical forms, such as the `pixi.lock` format version or
  the generated platform alias names, and CI requires the committed bytes to
  match what the pinned version produces. The pin must remain at least 0.71.0,
  which supplies the content-addressed source-build cache used by CI and writes
  the repository's version 7 lockfiles.
- If a PR does not intentionally change pixi dependencies or metadata, do not
  include unrelated lockfile churn. `pixi run` can refresh a stale lockfile
  implicitly; revert that noise unless the refresh is the point of the change.
- If a lockfile changes, the PR description should briefly say why.
- Isolate large dependency refreshes from feature work when possible. Prefer a
  dedicated lockfile-only PR over mixing solver churn into an unrelated change.

CI enforces the contract: `pixi lock --check` fails when a committed lockfile is
stale, and pixi source-build jobs run with `PIXI_LOCKED=true` so they install from
the committed lock rather than updating it during the job. If either check
fails, regenerate and commit the affected lockfile.

The freshness check additionally fails when the check itself rewrote a lockfile.
`pixi lock --check` accepts a lock whose solution is current but whose bytes are
not canonical for the pinned pixi version, quietly normalizing the file instead,
which leaves every later pixi run rewriting the committed lockfile.

A scheduled workflow (`CI: pixi lockfile refresh`) runs
`pixi update --no-install` for every workspace in one job and opens one PR with
all changed lockfiles, so broad dependency churn is reviewed as maintenance
rather than landing inside unrelated feature work. The workflow can also be
dispatched manually. Both lockfile workflows resolve their workspace lists
through `ci/tools/list_pixi_workspaces.py`, which derives the inventory from the
committed manifests, so a newly added workspace is picked up without editing a
workflow.

Refresh PRs use `GITHUB_TOKEN`. After one opens, a maintainer with write access
must first select **Approve workflows to run** in the merge box, then assign
themselves to the PR. Approval starts the queued `pull_request` runs; the
human-generated `assigned` event creates the required
**PR has assignee, labels, and milestone** `pull_request_target` check and gives
the PR a clear owner.

A future GitHub App integration could trigger both `pull_request` and
`pull_request_target` workflows automatically.
See GitHub's
[token event documentation](https://docs.github.com/en/actions/concepts/security/github_token#when-github_token-triggers-workflow-runs)
for the current behavior.

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

### Branch-specific Artifact Flow

#### Main Branch
- **Build** → **Test** → **Documentation** → **Potential Release**
- Artifacts stored as `{component}-python{version}-{platform}-{sha}`
- Full test coverage across all platforms and CUDA versions
- **Artifact flow out**: `cuda-pathfinder` artifacts → backport branches

#### Backport Branches
- **Build** → **Test** → **Backport PR Creation**
- Artifacts used for validation before creating backport pull requests
- Maintains compatibility with older CUDA versions
- **Artifact flow in**: `cuda-pathfinder` artifacts ← main branch
- **Artifact flow out**: older `cuda-bindings` artifacts → main branch

### Key Infrastructure Details

- **Self-hosted runners**: Used for Linux builds and GPU testing (more resources, faster builds)
- **GitHub-hosted runners**: Used for Windows builds and general tasks
- **Artifact retention**: 30 days for GitHub Artifacts (wheels, docs, tests)
- **Cache retention**: GitHub Cache for build dependencies and environments
- **Security**: All commits must be signed, untrusted code blocked
- **Parallel execution**: Matrix builds across Python versions and platforms
- **Component isolation**: Each component (core, bindings, pathfinder, python) can be built/released independently

## Code coverage

Code coverage reports are produced nightly and posted to [GitHub Pages](https://nvidia.github.io/cuda-python/coverage).

Known limitations: Code coverage is only run on Linux x86_64 with an a100 GPU.  We plan to add more platform and GPU coverage in the future.

---

<a>1</a>: The `cuda-python` meta package shares the same license and the contributing guidelines as those of `cuda-bindings`.
