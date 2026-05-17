#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $(basename "$0") ctk-major-minor-patch" 1>&2
    exit 1
fi

cuda_version="$1"
cuda_major="${cuda_version%%.*}"
uname_m="$(uname -m)"
case "$cuda_major" in
    12)
        python_version=3.12
        ;;
    13)
        python_version=3.14
        ;;
    *)
        echo "Unsupported CUDA major version for this helper: $cuda_major" 1>&2
        echo "Expected a 12.x or 13.x toolkit version." 1>&2
        exit 1
        ;;
esac

eval "$(conda shell.bash hook)"

conda create --yes -n "pathfinder_testing_cu$cuda_version" "python=$python_version" cuda-toolkit="$cuda_version"
set +u
conda activate "pathfinder_testing_cu$cuda_version"
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
    if [[ "$cuda_major" == "13" ]]; then
        cpkgs+=("libcudla-dev")
    fi
fi

for cpkg in "${cpkgs[@]}"; do
    echo "CONDA INSTALL: $cpkg"
    set +u
    conda install -y -c conda-forge "$cpkg"
    set -u
done
