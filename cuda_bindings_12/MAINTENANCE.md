<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Maintaining the CUDA 12 bindings line

`cuda_bindings_12/` is the CUDA 12.9 maintenance package root. The current
CUDA 13 bindings live in `cuda_bindings/`. The package roots and their release
statuses are configured in [`ci/versions.yml`](../ci/versions.yml).

Integrate both release lines on `main`. The former `12.9.x` branch is a record
of past releases; new CUDA 12 fixes, tests, and documentation belong in
`cuda_bindings_12/`. Short release branches may stabilize a release or carry
an urgent fix; return such fixes to `main` and assess the CUDA 13 root too.

Follow the [bindings release guide](../.github/RELEASE-bindings.md) for branch
selection, patch-release scope, backports, and release validation.

Follow the [shared regeneration and review policy](../ci/README.md#maintaining-both-bindings-roots)
when changing implementation in either package root.
