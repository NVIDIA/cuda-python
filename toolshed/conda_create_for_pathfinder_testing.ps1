# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$PythonMajorMinor,
    [Parameter(Mandatory = $true, Position = 1)]
    [string]$CudaMajorMinorPatch
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

& "$env:CONDA_EXE" "shell.powershell" "hook" | Out-String | Invoke-Expression

conda create --yes -n "pathfinder_testing_cu$CudaMajorMinorPatch" "python=$PythonMajorMinor" "cuda-toolkit=$CudaMajorMinorPatch"
conda activate "pathfinder_testing_cu$CudaMajorMinorPatch"

# Keep this list aligned with the Windows-installable subset of
# cuda_pathfinder/pyproject.toml.
$cpkgs = @(
    "cusparselt-dev",
    "cutensor",
    "cutlass",
    "libcudss-dev",
    "libmathdx-dev"
)

Write-Host "CONDA INSTALL: $($cpkgs -join ' ')"
conda install -y -c conda-forge @cpkgs
