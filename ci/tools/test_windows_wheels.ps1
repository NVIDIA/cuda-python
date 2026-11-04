# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

param(
    [Parameter(Mandatory = $true)][string]$PythonExecutable,
    [Parameter(Mandatory = $true)][string]$PythonVersion,
    [Parameter(Mandatory = $true)][string]$CudaVersion,
    [string]$SourceDirectory = '.',
    [string]$Wheelhouse = 'wheelhouse',
    [string]$ArtifactNames = ''
)

$ErrorActionPreference = 'Stop'
$source = (Resolve-Path -LiteralPath $SourceDirectory).Path
$wheelhousePath = (Resolve-Path -LiteralPath $Wheelhouse).Path
$resultsDirectory = Join-Path $wheelhousePath 'test-results'
New-Item -ItemType Directory -Force -Path $resultsDirectory | Out-Null
$python = $PythonExecutable

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $Arguments"
    }
}

if ($ArtifactNames) {
    $names = $ArtifactNames | ConvertFrom-Json
    $pythonArtifacts = $names.python.$PythonVersion
    if (-not $names.pathfinder -or -not $pythonArtifacts.bindings -or -not $pythonArtifacts.core) {
        throw "Missing artifact names for Python $PythonVersion."
    }
    $pathfinderDirectory = Join-Path $wheelhousePath $names.pathfinder
    $bindingsDirectory = Join-Path $wheelhousePath $pythonArtifacts.bindings
    $coreDirectory = Join-Path $wheelhousePath $pythonArtifacts.core
    $pathfinder = @(Get-ChildItem -LiteralPath $pathfinderDirectory -Filter '*.whl' -File)
    $bindings = @(Get-ChildItem -LiteralPath $bindingsDirectory -Filter '*.whl' -File)
    $core = @(Get-ChildItem -LiteralPath $coreDirectory -Filter '*.whl' -File)
}
else {
    $pathfinder = @(Get-ChildItem -LiteralPath $wheelhousePath -Filter 'cuda_pathfinder-*.whl' -File)
    $bindings = @(Get-ChildItem -LiteralPath $wheelhousePath -Filter 'cuda_bindings-*.whl' -File)
    $core = @(Get-ChildItem -LiteralPath $wheelhousePath -Filter 'cuda_core-*.whl' -File)
}
if ($pathfinder.Count -ne 1 -or $bindings.Count -ne 1 -or $core.Count -ne 1) {
    throw "Expected exactly one pathfinder, bindings, and core wheel for Python $PythonVersion."
}
$wheelPaths = @($pathfinder[0].FullName, $bindings[0].FullName, $core[0].FullName)
$wheelPaths | ForEach-Object { Write-Host "Installing wheel: $_" }
$venv = Join-Path $wheelhousePath '.venv'
Invoke-CheckedPython -Arguments @('-m', 'venv', $venv)
$python = Join-Path $venv 'Scripts\python.exe'
$env:VIRTUAL_ENV = $venv
$env:PATH = "$venv\Scripts;$env:CUDA_PATH\bin;$env:PATH"
$env:EXPECTED_PYTHON_VERSION = $PythonVersion
$env:EXPECTED_CUDA_VERSION = $CudaVersion

Push-Location $source
try {
    Invoke-CheckedPython -Arguments @('-m', 'pip', 'install', '--no-cache-dir', '--upgrade', 'pip')
    $installArguments = @('-m', 'pip', 'install', '--no-cache-dir') + $wheelPaths + @(
        '--group', './cuda_pathfinder/pyproject.toml:test',
        '--group', './cuda_bindings_13/pyproject.toml:test',
        '--group', './cuda_core/pyproject.toml:test'
    )
    Invoke-CheckedPython -Arguments $installArguments
    Invoke-CheckedPython -Arguments @('-m', 'pip', 'check')

    $verifyEnvironment = @'
import ctypes
import os
import sys
import sysconfig
from pathlib import Path

import cuda.bindings
import cuda.core
import cuda.pathfinder

expected_python = os.environ["EXPECTED_PYTHON_VERSION"]
expected_pair = tuple(map(int, expected_python.rstrip("t").split(".")[:2]))
print(f"Python: {sys.executable}; {sys.version}", flush=True)
assert sys.version_info[:2] == expected_pair, (sys.version_info, expected_python)
assert sysconfig.get_platform() == os.environ["PYTHON_PLATFORM"], sysconfig.get_platform()
assert bool(sysconfig.get_config_var("Py_GIL_DISABLED")) == expected_python.endswith("t")
if expected_python.endswith("t"):
    assert not sys._is_gil_enabled(), "The GIL is unexpectedly enabled"
for module in (cuda.pathfinder, cuda.bindings, cuda.core):
    module_path = Path(module.__file__).resolve()
    print(f"{module.__name__}: {module_path}", flush=True)
    assert module_path.is_relative_to(Path(sys.prefix).resolve()), module_path

cuda_root = Path(os.environ["CUDA_PATH"]).resolve()
expected_cuda = tuple(map(int, os.environ["EXPECTED_CUDA_VERSION"].split(".")[:2]))
for name, query in (("cudart", "cudaRuntimeGetVersion"), ("nvrtc", "nvrtcVersion")):
    loaded = cuda.pathfinder.load_nvidia_dynamic_lib(name)
    assert loaded.abs_path, (name, loaded)
    library_path = Path(loaded.abs_path).resolve()
    assert library_path.is_relative_to(cuda_root), (name, library_path, cuda_root)
    library = ctypes.WinDLL(str(library_path))
    if query == "cudaRuntimeGetVersion":
        version = ctypes.c_int()
        assert library.cudaRuntimeGetVersion(ctypes.byref(version)) == 0
        actual_cuda = (version.value // 1000, version.value % 1000 // 10)
    else:
        major, minor = ctypes.c_int(), ctypes.c_int()
        assert library.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)) == 0
        actual_cuda = (major.value, minor.value)
    print(f"{library_path}: {actual_cuda}", flush=True)
    assert actual_cuda == expected_cuda, (library_path, actual_cuda, expected_cuda)
'@
    $verifyPath = Join-Path $venv 'verify-environment.py'
    [IO.File]::WriteAllText($verifyPath, $verifyEnvironment, [Text.UTF8Encoding]::new($false))
    Invoke-CheckedPython -Arguments @($verifyPath)

    cmd /d /c 'call cuda_bindings_13\tests\cython\build_tests.bat'
    if ($LASTEXITCODE -ne 0) {
        throw "Cython test build failed with exit code $LASTEXITCODE."
    }

    $script:testResult = 0
    function Invoke-Pytest {
        param([string]$TestPath, [string]$Suite)
        $report = Join-Path $resultsDirectory "junit-$Suite.xml"
        & $python -m pytest -ra -s -vv --junitxml $report $TestPath
        if ($LASTEXITCODE -ne 0) {
            $script:testResult = 1
        }
    }

    Invoke-Pytest 'cuda_pathfinder/tests' 'pathfinder'
    Invoke-Pytest 'cuda_bindings_13/tests' 'bindings'
    $previousStreamMode = $env:CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM
    $env:CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM = '1'
    try {
        Invoke-Pytest 'cuda_bindings_13/tests' 'bindings-ptds'
    }
    finally {
        $env:CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM = $previousStreamMode
    }
    Invoke-Pytest 'cuda_bindings_13/tests/cython' 'bindings-cython'
    Invoke-Pytest 'cuda_core/tests' 'core'
    if ($script:testResult -ne 0) {
        throw 'One or more installed-wheel test suites failed.'
    }
}
finally {
    Pop-Location
}
