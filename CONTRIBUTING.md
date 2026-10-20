# Contributing to CUDA Python

Thank you for your interest in contributing to CUDA Python! Bug reports,
reproducible examples, documentation improvements, and code changes are all
welcome. Our review capacity is limited, so for anything beyond a small fix we
ask you to check in with us before you start; the sections below explain when
and how. That way your effort goes into changes we can take, and our review
time goes where it helps most.

## Table of Contents

- [Contributing to CUDA Python](#contributing-to-cuda-python)
  - [Table of Contents](#table-of-contents)
  - [Report a problem or suggest a change](#report-a-problem-or-suggest-a-change)
  - [Coordinate before implementing](#coordinate-before-implementing)
    - [Working on an existing issue](#working-on-an-existing-issue)
    - [Small standalone fixes](#small-standalone-fixes)
    - [Larger changes and shared infrastructure](#larger-changes-and-shared-infrastructure)
    - [Readiness notices on issues](#readiness-notices-on-issues)
    - [What you can expect from us](#what-you-can-expect-from-us)
  - [Using coding agents](#using-coding-agents)
  - [Cloning the repository](#cloning-the-repository)
    - [Recommended clone](#recommended-clone)
    - [Updating your clone](#updating-your-clone)
    - [Why the clone matters](#why-the-clone-matters)
    - [Symptoms of a bad clone](#symptoms-of-a-bad-clone)
  - [Development on Windows](#development-on-windows)
    - [Enabling git symlinks](#enabling-git-symlinks)
  - [Type stubs for cuda.core](#type-stubs-for-cudacore)
  - [Pre-commit](#pre-commit)
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


## Report a problem or suggest a change

If you run into a bug, please [file an
issue](https://github.com/NVIDIA/cuda-python/issues/new/choose). Include a
minimal reproducer, the package and CUDA versions involved, your platform, and
what you expected versus what happened. Please also tell us how the problem
affects you: regressions, incorrect results, and bugs that block your work help
us set priorities, and a comment on the issue is the right way to flag
something that needs urgent attention.

Feature requests, documentation corrections, measurements, and suggested
regression tests are welcome too. Reporting a problem or adding evidence to an
issue never requires an assignment. For questions, please use [GitHub
Discussions](https://github.com/NVIDIA/cuda-python/discussions). The NVIDIA
team triages issues by user impact, release plans, and the review and
maintenance capacity we have.

## Coordinate before implementing

The short version:

- **Found a bug or have an idea?** Open an issue. No assignment needed.
- **Fixing a small, self-contained bug or documentation error?** Open a PR
  directly. Say what was wrong and how you checked the fix.
- **Want to work on an existing issue?** Comment on it and wait for a
  maintainer to assign it to you before you start.
- **Planning a new feature, a design change, or anything touching CI or
  release infrastructure?** Open an issue first and agree on scope and testing
  with a maintainer before writing code.

This section is written for external contributors, which in this guide means
anyone contributing from outside NVIDIA. If you are an NVIDIA colleague, see
the note at the end of the section.

Why we ask: an open issue may record an idea, an investigation, or a proposed
approach that nobody has agreed to yet, and even an issue with a detailed
suggested implementation is not necessarily ready for a PR. Review is also
more than reading a diff. We need to understand the problem, check behavior
and tests, and think about long-term maintenance. Checking in first protects
you from investing in a change we cannot take, protects other contributors
from duplicate work, and lets us put review time where it helps most.

### Working on an existing issue

Comment on the issue to say you would like to take it, and wait for a
maintainer to assign it to you before you start implementing. If a maintainer
has already asked for the change, in the issue or in a review, that counts as
agreement and no separate assignment is needed. Issues labeled
`good first issue` are ones we have already judged suitable for a newcomer;
claiming one works the same way.

Before you start, confirm the scope and how the change will be tested,
especially if part of the testing needs hardware or CI you do not have. Agree
on which checks you can run yourself and which need help from us, and mention
those limits in the PR.

### Small standalone fixes

Small, self-contained bug fixes and documentation corrections do not need an
issue or an assignment: open a PR directly. As a rule of thumb, a standalone
fix changes a few lines, involves no API or behavior design decisions, and
does not touch CI or release infrastructure. In the PR, explain the problem
and how you validated the correction. If an issue already exists, link it and
leave a short comment there so nobody else starts on the same fix.

### Larger changes and shared infrastructure

For a new feature, a change that involves design decisions, or anything that
touches workflows, shared caches, runners, or release infrastructure, open an
issue to discuss the problem and the approach first, and agree on scope and
validation with a maintainer before writing code. Infrastructure changes in
particular often need operational context and validation that are not visible
from the issue alone. If you are not sure who the right maintainer is, just
ask on the issue; the team will route it.

Keep each PR focused and independently reviewable. If the work needs several
dependent PRs, agree on the sequence with a maintainer first. You do not need
access to the upstream repository for any of this: develop in a fork and open
PRs from there.

### Readiness notices on issues

Some issues carry a notice at the top, like this one:

> **Maintainer investigation; not yet open for external implementation
> PRs.** The approach touches shared CI infrastructure and is still being
> decided. Reproducers, measurements, and suggestions are very welcome here;
> please check with a maintainer before starting on an implementation.

The notice means the approach is still being decided. It is not a request to
stay away: reproducers, measurements, and discussion are exactly what helps at
that stage. Please hold off on an implementation PR until a maintainer
confirms the plan; the notice is then updated or removed. A notice starting
with "Proposal awaiting maintainer agreement" means the same for an idea that
is not an active investigation yet, and says what remains to be decided.

For maintainers: add the notice to investigations and proposals that are not
ready for implementation, pick the status that matches the issue, adapt the
reason, and update the notice when the direction changes. Routine bug reports
do not need one just because a fix has not been designed yet.

### What you can expect from us

- We read every issue and PR. Triage weighs user impact, release plans, and
  the review and maintenance capacity we have.
- When you ask to take an issue or propose an approach, we will answer, even
  when the answer is "not now". If you have not heard back after a week or
  two, a short ping on the issue is welcome.
- If a PR arrives without the coordination described here, we will point you
  to this section and may put the PR on hold until the approach is agreed,
  rather than reviewing it ahead of agreed work.
- When we decline a change, we try to say why, so the effort is not wasted
  next time.

Please also read the relevant component's guidelines:

- [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/contribute.html)
- [`cuda.bindings`](https://nvidia.github.io/cuda-python/cuda-bindings/latest/contribute.html)<sup>[1](#footnote1)</sup>
- [`cuda.pathfinder`](https://nvidia.github.io/cuda-python/cuda-pathfinder/latest/contribute.html)

**A note for NVIDIA colleagues.** You coordinate through the team's internal
planning and maintainer direction rather than through the public assignment
step. Everything else in this guide, including the quality, validation, and
review expectations, applies to everyone.

## Using coding agents

Coding agents are welcome here, and the same expectations apply as for any
other contribution: the coordination step for external contributors, and the
quality bar for everyone. A patch that was quick to generate still takes the
same care to review and maintain, so the useful question is not whether you
can produce a change but whether it is one we can take.

Give your agent [AGENTS.md](AGENTS.md) and this guide. AGENTS.md asks the agent
to check an issue's readiness notice, assignment, and discussion, and to tell
you, before it starts on an implementation, whether maintainer agreement
already exists, whether the change is a small standalone fix, or what the next
step is. If your agent does not do this on its own, ask it to.

Before submitting an agent-assisted PR:

- Give the agent the relevant repository and component instructions, the issue
  discussion, the surrounding code, and the tests. Treat a fix suggested in an
  issue, or by the agent, as a proposal to evaluate, not as a design that has
  been accepted.
- Read and understand the complete diff. You should be able to explain the
  problem, the approach, and the resulting behavior, and you are the one who
  responds to review feedback and corrects the change.
- Validate the change against the reported problem and run the relevant
  checks. Report what actually ran and what you could not run. Mark new tests
  with the authorship markers described in
  [AGENTS.md](AGENTS.md#test-authorship-markers).
- Keep the change within the agreed scope, and please mention in the PR
  description that an agent helped and roughly how; it helps reviewers
  calibrate.
- One PR at a time, please. Batches of speculative PRs picked from the backlog
  are costly to review and rarely land. If you would like to do more, ask on
  an issue and we will point you at work that is ready.

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

This section collects Windows-specific guidance for contributors working
outside of WSL. WSL contributors can follow the Linux flow in the rest of this
document.

### Enabling git symlinks

The repository contains documentation and metadata symlinks, including
`CLAUDE.md`, `cuda_python/README.md`, and the per-package `.git_archival.txt`
files. It also currently tracks the build-critical
`cuda_core/_build_shared.py` as a symlink to the canonical
`cuda_bindings/_build_shared.py`. The core path may instead be maintained as a
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

<a name="footnote1">1</a>: The `cuda-python` meta package shares the same license and the contributing guidelines as those of `cuda-bindings`.
