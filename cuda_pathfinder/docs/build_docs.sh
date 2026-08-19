#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${SCRIPT_DIR}/../../cuda_python/docs/build_component_docs.sh" cuda-pathfinder "$@"
