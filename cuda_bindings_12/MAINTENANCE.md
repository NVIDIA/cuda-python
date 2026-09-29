<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Maintaining the CUDA 12 bindings line

`cuda_bindings_12/` is the CUDA 12.9 maintenance package root. The current
CUDA 13 bindings live in `cuda_bindings/`. The package roots and their release
statuses are configured in [`ci/versions.yml`](../ci/versions.yml).

Maintain both release lines on `main`. The former `12.9.x` branch is a record
of past releases; new CUDA 12 fixes, tests, documentation, and releases belong
in `cuda_bindings_12/` on `main`.

Follow the [shared regeneration and review policy](../ci/README.md#maintaining-both-bindings-roots)
when changing implementation in either package root.
