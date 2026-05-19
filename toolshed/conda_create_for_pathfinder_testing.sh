#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $(basename "$0") python_major_minor cuda_major_minor_patch" 1>&2
    exit 1
fi

python_major_minor="$1"
cuda_major_minor_patch="$2"
cuda_major="${cuda_major_minor_patch%%.*}"
uname_m="$(uname -m)"

eval "$(conda shell.bash hook)"

conda create --yes -n "pathfinder_testing_cu$cuda_major_minor_patch" "python=$python_major_minor" cuda-toolkit="$cuda_major_minor_patch"
set +u
conda activate "pathfinder_testing_cu$cuda_major_minor_patch"
set -u

# Keep this list aligned with the Linux-installable subset of
# cuda_pathfinder/pyproject.toml.
cpkgs=(
    "cusparselt-dev"
    "cutensor"
    "cutlass"
    "libcublasmp-dev"
    "libcudss-dev"
    "libcufftmp-dev"
    "libcusolvermp-dev"
    "libmathdx-dev"
    "libnvshmem3"
    "libnvshmem-dev"
)

# Keep the conda environment aligned with platform-scoped pyproject groups.
if [[ "$uname_m" == "aarch64" ]]; then
    cpkgs+=("libnvpl-fft-dev")
    if (( cuda_major >= 13 )); then
        cpkgs+=("libcudla-dev")
    fi
fi

echo "CONDA INSTALL: ${cpkgs[*]}"
set +u
conda install -y -c conda-forge "${cpkgs[@]}"
set -u
