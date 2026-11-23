#!/bin/bash
set -eo pipefail

# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# BEGIN SYNCED CYTHON TEST BUILD BASH
script_dir=$(dirname "$(realpath -- "$0")")
case "$(uname)" in
  Linux) ;;
  CYGWIN*|MINGW*|MSYS*) script_dir=$(cygpath -w "$script_dir") ;;
  *) exit 1 ;;
esac

# Legacy callers select the build toolkit with CUDA_HOME.
if [[ -n "${CUDA_HOME:-}" ]]; then
  export CUDA_PATH="$CUDA_HOME"
fi
exec python "$script_dir/build_tests.py"
# END SYNCED CYTHON TEST BUILD BASH
