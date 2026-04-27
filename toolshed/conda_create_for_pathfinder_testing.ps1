# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0

param(
    [Parameter(Mandatory = $true)]
    [string]$CudaVersion
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

& "$env:CONDA_EXE" "shell.powershell" "hook" | Out-String | Invoke-Expression

conda create --yes -n "pathfinder_testing_cu$CudaVersion" python=3.14 "cuda-toolkit=$CudaVersion"
conda activate "pathfinder_testing_cu$CudaVersion"

$cpkgs = @(
    "pytest>=6.2.4",
    "pytest-mock",
    "pytest-repeat",
    "pytest-randomly",
    "cusparselt-dev",
    "cutensor",
    "cutlass",
    "libcudss-dev",
    "libmathdx-dev"
)

# Keep the PowerShell environment aligned with the Windows-relevant
# cuda_pathfinder dependency groups; Linux-only deps stay in the .sh script.
foreach ($cpkg in $cpkgs) {
    Write-Host "CONDA INSTALL: $cpkg"
    conda install -y -c conda-forge $cpkg
}
