#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

UNAME=$(uname)
if [ "$UNAME" == "Linux" ] ; then
  SCRIPTPATH=$(dirname $(realpath "$0"))
  export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
elif [[ "$UNAME" == CYGWIN* || "$UNAME" == MINGW* || "$UNAME" == MSYS* ]] ; then
  SCRIPTPATH="$(dirname $(cygpath -w $(realpath "$0")))"
  export CL="/I\"${CUDA_HOME}\\include\" ${CL}"
else
  exit 1
fi

cythonize -3 -i -Xfreethreading_compatible=True ${SCRIPTPATH}/test_*.pyx
