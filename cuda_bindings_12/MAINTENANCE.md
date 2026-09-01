<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Maintaining the CUDA 12 bindings line

## Ownership model

This directory makes CUDA 12.9 bindings buildable from `main` alongside the
CUDA 13 bindings in `cuda_bindings/`. The two package roots are an intentional
transitional design. Moving target-specific generated output into overlays and
sharing more handwritten implementation is a separate architectural change.

The imported tree comes from
`NVIDIA/cuda-python@238955935bd903ac72817c0dfdfe4f6a54ee6bb1:cuda_bindings`.
Cybind commit `95d8bb525de46a9ff7ae40d759a98cbe50cf8391`
reproduces 52 generated paths in that snapshot, including all 40 generated
paths modified by #2604. It does not reproduce the snapshot's legacy
runtime/v2 storage layout, so generation provenance for the complete imported
root remains partial.

## File classes

- Generated bindings carry an embedded content seal. A seal proves file
  integrity, not the cybind revision, target, or cross-root synchronization.
  Change the cybind source first, generate into a clean temporary output
  directory for CUDA 12.9, and copy the generated result only after reviewing
  the diff. Record the cybind commit, toolkit inputs, and command in the pull
  request.
- Files listed in `ci/cuda-bindings-shared-files.json` are intentionally shared
  by the public bindings lines and must remain byte-identical. Pre-commit and CI
  apply the list to the public source roots declared in `ci/versions.yml`.
- Cybind-owned support files, including `_internal/_fast_enum.py` and
  `_internal/utils.pxd`, are not in the shared-file list. They follow cybind's
  snippets and templates and may legitimately change with generated output.
- Files absent from that list may differ because the CUDA 12 line retains the
  legacy runtime-generation layout, APIs, tests, documentation, packaging, and
  toolkit pins. Absence from the list does not mean fixes may be ignored.

Cybind's target behavior is intentionally mixed. Megaheader-backed libraries
generated with `need_headers_at_build=False` expose the latest public APIs
known to the generator across its supported toolkit range. Header-consuming
libraries such as cuFile and the legacy bindings remain target-filtered.
The megaheader files happen to match the CUDA 13 tree while 13.3 is the latest
public release, but that is not a cross-major invariant: a CUDA 13 prerelease
target may legitimately diverge while CUDA 12 remains capped at the latest
public release.

## Contributor checklist

For every handwritten fix in either bindings root, state one of the following
in the pull request:

- the corresponding file in the other root was updated;
- the shared-file manifest already proves the roots match; or
- the change is not applicable to the other root, with a concrete reason.

For generated changes, validate all modified seals with
`python toolshed/check_generated_file_seals.py <paths...>` and retain the
generation provenance in the pull request. Run
`python ci/tools/check_cuda_bindings_shared_files.py` after changing either
bindings tree.
