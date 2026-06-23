#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Build .o test fixtures. Invoked at CI build stage with the oldest test-matrix
# CTK (prev-cuda-version, currently 12.9.x) so nvJitLink on 12.9/13.0/13.3
# test jobs can consume the embedded device code.

SCRIPTPATH=$(dirname "$(realpath "$0")")

NVCC_EXTRA_FLAGS=(-std=c++17)
if [[ "${OS:-}" == "Windows_NT" ]]; then
    NVCC_EXTRA_FLAGS+=(-Xcompiler /Zc:preprocessor)
fi

GENCODE=()
for cc in 70 75 80 89 90 120; do
    if nvcc --list-gpu-code | grep -qx "sm_${cc}"; then
        GENCODE+=(-gencode "arch=compute_${cc},code=sm_${cc}")
    fi
done

nvcc -dc "${NVCC_EXTRA_FLAGS[@]}" "${GENCODE[@]}" -o "${SCRIPTPATH}/saxpy.o" "${SCRIPTPATH}/saxpy.cu"

ls -lah "${SCRIPTPATH}/saxpy.o"
