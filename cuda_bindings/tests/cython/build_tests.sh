#!/bin/bash
set -eo pipefail

# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

UNAME=$(uname)
if [ "$UNAME" == "Linux" ] ; then
  SCRIPTPATH=$(dirname $(realpath "$0"))
  export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}
elif [[ "$UNAME" == CYGWIN* || "$UNAME" == MINGW* || "$UNAME" == MSYS* ]] ; then
  SCRIPTPATH="$(dirname $(cygpath -w $(realpath "$0")))"
  export CL="/I\"${CUDA_HOME}\\include\" ${CL:-}"
else
  exit 1
fi

# Use a Python driver so the cuda.bindings source root is resolved at
# runtime and passed via Cython's include_path -- avoids platform-specific
# PYTHONPATH separator handling and surfaces import errors as exceptions.
# nthreads=1 inside the driver mirrors the previous `-j 1` to side-step
# any process-pool issues and keep builds deterministic.
python "${SCRIPTPATH}/build_tests.py"
