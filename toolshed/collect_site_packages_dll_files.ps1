# collect_site_packages_dll_files.ps1

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Usage:
#     cd cuda-python
#     powershell -File toolshed\collect_site_packages_dll_files.ps1
#     python .\toolshed\make_site_packages_libdirs.py windows site_packages_dll.txt

$ErrorActionPreference = 'Stop'

function Fresh-Venv {
    param(
        [Parameter(Mandatory=$true)]
        [string] $Path
    )
    & python3 -m venv $Path
    . (Join-Path $Path 'Scripts\Activate.ps1')
    python -m pip install --upgrade pip
}

Set-Location -Path 'cuda_pathfinder'

Fresh-Venv -Path '..\TmpCp12Venv'
pip install --only-binary=:all: -e '.[test,test_nvidia_wheels_cu12,test_nvidia_wheels_host]'
deactivate

Fresh-Venv -Path '..\TmpCp13Venv'
pip install --only-binary=:all: -e '.[test,test_nvidia_wheels_cu13,test_nvidia_wheels_host]'
deactivate

Set-Location -Path '..'

$venvs = @('TmpCp12Venv', 'TmpCp13Venv')

$matches =
    Get-ChildItem -Path $venvs -Recurse -File -Include '*.dll' |
    Where-Object { $_.FullName -match '(?i)(nvidia|nvpl)' } |
    Select-Object -ExpandProperty FullName |
    Sort-Object -Unique

$outFile = 'site_packages_dll.txt'
$matches | Set-Content -Path $outFile -Encoding utf8
