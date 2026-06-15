#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Build .o test fixtures. Invoked at CI build stage

SCRIPTPATH=$(dirname "$(realpath "$0")")

NVCC_EXTRA_FLAGS=()
if [[ "${OS:-}" == "Windows_NT" ]]; then
    # CCCL headers (e.g. cuda/std/cstddef) require MSVC's conforming preprocessor.
    NVCC_EXTRA_FLAGS+=(-Xcompiler /Zc:preprocessor)
fi

nvcc -dc "${NVCC_EXTRA_FLAGS[@]}" -o "${SCRIPTPATH}/saxpy.o" "${SCRIPTPATH}/saxpy.cu"

ls -lah "${SCRIPTPATH}/saxpy.o"
