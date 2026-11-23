@echo off

REM SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
REM SPDX-License-Identifier: Apache-2.0

REM BEGIN SYNCED CYTHON TEST BUILD CMD
setlocal
REM Legacy callers select the build toolkit with CUDA_HOME.
if defined CUDA_HOME set "CUDA_PATH=%CUDA_HOME%"
python "%~dp0build_tests.py"
set "BUILD_RESULT=%ERRORLEVEL%"
endlocal & exit /b %BUILD_RESULT%
REM END SYNCED CYTHON TEST BUILD CMD
