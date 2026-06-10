#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Build .o test fixtures. Invoked at CI build stage

SCRIPTPATH=$(dirname "$(realpath "$0")")

nvcc -dc -o "${SCRIPTPATH}/saxpy.o" "${SCRIPTPATH}/saxpy.cu"

ls -lah "${SCRIPTPATH}/saxpy.o"
