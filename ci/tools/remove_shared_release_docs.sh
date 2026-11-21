#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

payload=$1
# The main docs build owns these paths at each component root.
for root in "$payload" "$payload"/cuda-{bindings,core,pathfinder}; do
    rm -rf -- "$root/latest"
    rm -f -- "$root"/{nv-versions.json,versions.json,objects.inv}
done
