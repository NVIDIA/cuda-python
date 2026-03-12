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

python - <<'PY'
from pathlib import Path
import sysconfig

site_packages = Path(sysconfig.get_path("purelib"))
for pth_file in site_packages.glob("__editable__.cuda_core-*.pth"):
    lines = pth_file.read_text(encoding="utf-8").splitlines()
    if not lines:
        continue
    # Older editable installs appended the local cuda_bindings checkout here.
    # Keep the editable import hook, but drop stale path injections so this
    # test build only sees the bindings path selected by the current env.
    if len(lines) > 1:
        pth_file.write_text(lines[0] + "\n", encoding="utf-8")
PY

find "${SCRIPTPATH}" -maxdepth 1 -type f \( -name 'test_*.c' -o -name 'test_*.cpp' -o -name 'test_*.so' \) -delete

cythonize -3 -i -Xfreethreading_compatible=True ${SCRIPTPATH}/test_*.pyx
