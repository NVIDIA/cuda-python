#!/bin/bash

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0

UNAME=$(uname)
if [ "$UNAME" == "Linux" ] ; then
  SCRIPTPATH=$(dirname $(realpath "$0"))
  export CPLUS_INCLUDE_PATH=${SCRIPTPATH}/../../cuda/core/experimental/include:$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
elif [[ "$UNAME" == CYGWIN* || "$UNAME" == MINGW* || "$UNAME" == MSYS* ]] ; then
  SCRIPTPATH="$(dirname $(cygpath -w $(realpath "$0")))"
  export CL="/I\"${SCRIPTPATH}\\..\\..\\cuda_core\\cuda\\core\\experimental\\include\" /I\"${CUDA_HOME}\\include\" ${CL}"
else
  exit 1
fi

cythonize -3 -i ${SCRIPTPATH}/test_*.pyx
