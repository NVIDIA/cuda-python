#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

UNAME=$(uname)
if [ "$UNAME" == "Linux" ] ; then
  SCRIPTPATH=$(dirname $(realpath "$0"))
  export CPLUS_INCLUDE_PATH=${SCRIPTPATH}/../../cuda/core/_include:$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
elif [[ "$UNAME" == CYGWIN* || "$UNAME" == MINGW* || "$UNAME" == MSYS* ]] ; then
  SCRIPTPATH="$(dirname $(cygpath -w $(realpath "$0")))"
  CUDA_CORE_INCLUDE_PATH=$(echo "${SCRIPTPATH}\..\..\cuda\core\_include" | sed 's/\\/\\\\/g')
  export CL="/I\"${CUDA_CORE_INCLUDE_PATH}\" /I\"${CUDA_HOME}\\include\" ${CL}"
else
  exit 1
fi

# pixi-build's editable install exposes the cuda namespace package via a
# finder hook that Cython's filesystem .pxd resolver does not consult.
# Surface the package's parent directory on PYTHONPATH so `cimport
# cuda.bindings.*` can locate the .pxd files.
CUDA_PKG_PARENT=$(python -c "import cuda.bindings as m, os; print(os.path.dirname(os.path.dirname(os.path.dirname(m.__file__))))")
export PYTHONPATH="${CUDA_PKG_PARENT}${PYTHONPATH:+:${PYTHONPATH}}"

cythonize -3 -i -Xfreethreading_compatible=True ${SCRIPTPATH}/test_*.pyx
