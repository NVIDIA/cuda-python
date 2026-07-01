#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Build .o test fixtures. Invoked at CI build stage

SCRIPTPATH=$(dirname "$(realpath "$0")")

NVCC_EXTRA_FLAGS=(-std=c++17)
if [[ "${OS:-}" == "Windows_NT" ]]; then
    NVCC_EXTRA_FLAGS+=(-Xcompiler /Zc:preprocessor)
fi

NVCC="${NVCC:-nvcc}"

"${NVCC}" -dc "${NVCC_EXTRA_FLAGS[@]}" -arch=all-major \
    -o "${SCRIPTPATH}/saxpy.o" "${SCRIPTPATH}/saxpy.cu"

if [[ "${OS:-}" == "Windows_NT" ]]; then
    nvcc -lib -o "${SCRIPTPATH}/saxpy.lib" "${SCRIPTPATH}/saxpy.o"
    ls -lah "${SCRIPTPATH}/saxpy.o" "${SCRIPTPATH}/saxpy.lib"
else
    nvcc -lib -o "${SCRIPTPATH}/saxpy.a" "${SCRIPTPATH}/saxpy.o"
    ls -lah "${SCRIPTPATH}/saxpy.o" "${SCRIPTPATH}/saxpy.a"
fi
