# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0

param(
    [Parameter(Mandatory = $true)]
    [string]$CudaVersion
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$cudaMajor = $CudaVersion.Split(".", 2)[0]
switch ($cudaMajor) {
    "12" { $pythonVersion = "3.12" }
    "13" { $pythonVersion = "3.14" }
    default {
        throw "Unsupported CUDA major version for this helper: $cudaMajor. Expected a 12.x or 13.x toolkit version."
    }
}

& "$env:CONDA_EXE" "shell.powershell" "hook" | Out-String | Invoke-Expression

conda create --yes -n "pathfinder_testing_cu$CudaVersion" "python=$pythonVersion" "cuda-toolkit=$CudaVersion"
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
