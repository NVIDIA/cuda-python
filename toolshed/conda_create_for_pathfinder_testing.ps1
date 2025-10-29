# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0

param(
    [Parameter(Mandatory = $true)]
    [string]$CudaVersion
)

$ErrorActionPreference = "Stop"

& "$env:CONDA_EXE" "shell.powershell" "hook" | Out-String | Invoke-Expression

conda create --yes -n "pathfinder_testing_cu$CudaVersion" python=3.13 "cuda-toolkit=$CudaVersion"
conda activate "pathfinder_testing_cu$CudaVersion"

$cpkgs = @(
    "cutensor",
    "libcublasmp-dev",
    "libcudss-dev",
    "libcufftmp-dev",
    "libmathdx-dev",
    "libnvshmem3",
    "libnvshmem-dev",
    "libnvpl-fft-dev"
)

foreach ($cpkg in $cpkgs) {
    Write-Host "CONDA INSTALL: $cpkg"
    conda install -y -c conda-forge $cpkg
}
