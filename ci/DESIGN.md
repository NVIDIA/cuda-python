# CI design principles

One engineer should be able to develop and maintain CI part time across several projects. Neither we nor most open-source projects can afford full-time DevOps. Our mission is to deliver high-quality libraries; CI is a means to that end. Judge infrastructure choices by the ongoing work they create—upgrades, debugging and changes across projects—as well as execution cost.

This design is a reference implementation for testing CUDA and Python on real GPUs wherever CUDA runs: Linux and Windows on x86-64 and aarch64, and WSL. It is used by [cuda-python](https://github.com/NVIDIA/cuda-python), [numba-cuda](https://github.com/NVIDIA/numba-cuda), [numba-cuda-mlir](https://github.com/NVIDIA/numba-cuda-mlir) and [CuPy](https://github.com/cupy/cupy). Any community project with a GPU runner, self-hosted or GitHub-hosted, should be able to adopt it. This document explains the principles, not today's workflows.

The design predates widespread agentic development. Keeping it explicit and small should also help agents understand it with less context and fewer tokens.

## Use community tooling and existing resources

Use public, open-source community tooling wherever it fits: GitHub and third-party actions, cibuildwheel, sccache, pip and PyPI, and pixi (conda). These tools are maintained and scrutinized by others, often before we need them. When a tool blocks us, help fix it upstream where possible. Write a custom helper only when nothing fits and we cannot afford to wait for the upstream fix, and keep it small and exercised by ordinary workflows.

Use GitHub's hosted runners, artifacts, caches, Pages and Releases so adopting the design requires little additional infrastructure. Use self-hosted runners only to fill gaps, such as GPUs.

## Use container images minimally

Maintain no images, and use no CUDA images. Use containers only when another requirement calls for them, such as PyPA's manylinux images for glibc compatibility. Layering tools on CUDA images adds time, bandwidth and storage costs.

Get CUDA from PyPI wheels or from NVIDIA's public redistributable archives, which our small `fetch_ctk` action unpacks and caches. The archives are the first format published in a CUDA release and the source of truth for other formats, including apt, yum, conda and pip.

Community tooling's support for Python pre-releases and direct access to CUDA archives enable Day 1 releases without waiting for custom CI infrastructure to catch up.

## Share one implementation across platforms

Use Bash for shared build and test logic, including Git Bash on Windows. A Linux-centric team maintains one implementation, and a fix lands once and runs everywhere. Keep short snippets inline in workflow steps. They read GitHub Actions state through expressions and contexts, often to decide whether a step runs, and passing that state back and forth to a script is awkward. Move longer logic shared across workflows into scripts. Use PowerShell only for Windows-specific operations, such as driver installation.

## Separate build runners from test runners

Build on CPU runners and pass artifacts to GPU test runners, so GPU time is spent on tests. Prebuild everything requiring a host compiler, including test extensions and binaries, to keep test environments lean and clean.

On Windows, this separation is a necessity. MSVC comes only from Microsoft's slow Visual Studio installer, not from any package manager. Without a pre-built image, GitHub-hosted Windows runners, which have it preinstalled, are the only practical place to build, and GPU test runners have no reasonable way to get it.

## Test what users install

Test built wheels in a fresh environment with explicitly installed dependencies. Build the docs from the same wheels, and publish the tested wheels without rebuilding. This checks packaging as well as library behavior, without relying on the build environment to supply runtime dependencies.

Most bugs users hit come from their environment, so cover the environments users have:

- CUDA installed from PyPI wheels or locally, across supported CUDA versions.
- Different drivers, Windows driver modes, and GPU configurations.
- Supported Python versions, free-threaded builds and pre-releases.

## Declare once, use everywhere

Keep pipelines, build and test versions, dependency groups and test configurations declarative. Give each shared setting a single source of truth that all pipelines look up and honor. A version change should not require finding and synchronizing copies across workflows.

The test matrix is an explicit list of rows. Limited runners may force coverage compromises; the design makes those choices visible and reviewable. A project can change coverage by editing rows rather than rewriting workflows.

## Reuse workflows and actions

Reuse stage workflows and composite actions through inputs. Extend an existing implementation when a new variant fits, rather than adding a near-copy that can drift. For example, `fetch_ctk` can overlay an additional component on an existing toolkit instead of defining another toolkit.

When porting to another repository, copy the reference files verbatim first and adapt them in later changes. This preserves a clear basis for review and helps distinguish deliberate differences from accidental drift.
