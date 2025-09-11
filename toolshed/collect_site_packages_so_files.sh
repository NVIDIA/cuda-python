#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Usage:
#     cd cuda-python
#     ./toolshed/collect_site_packages_so_files.sh
#     ./toolshed/make_site_packages_libdirs.py linux site_packages_so.txt

set -euo pipefail
fresh_venv() {
    python3 -m venv "$1"
    . "$1/bin/activate"
    pip install --upgrade pip
}
cd cuda_pathfinder/
fresh_venv ../TmpCp12Venv
set -x
pip install --only-binary=:all: -e .[test,test_nvidia_wheels_cu12,test_nvidia_wheels_host]
set +x
deactivate
fresh_venv ../TmpCp13Venv
set -x
pip install --only-binary=:all: -e .[test,test_nvidia_wheels_cu13,test_nvidia_wheels_host]
set +x
deactivate
cd ..
set -x
find TmpCp12Venv TmpCp13Venv -name 'lib*.so*' | grep -e nvidia -e nvpl >site_packages_so.txt
