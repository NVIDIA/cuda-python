#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if [[ $# -ne 1 ]]; then
    echo "Usage: $(basename "$0") ctk-major-minor-patch" 1>&2
    exit 1
fi

eval "$(conda shell.bash hook)"

conda create --yes -n "pathfinder_testing_cu$1" python=3.13 cuda-toolkit="$1"
conda activate "pathfinder_testing_cu$1"

for cpkg in \
    cusparselt \
    cutensor \
    libcublasmp-dev \
    libcudss-dev \
    libcufftmp-dev \
    libmathdx-dev \
    libnvshmem3 \
    libnvshmem-dev \
    libnvpl-fft-dev; do
    echo "CONDA INSTALL: $cpkg"
    conda install -y -c conda-forge "$cpkg"
done
