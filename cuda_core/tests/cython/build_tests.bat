@echo off

REM SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
REM SPDX-License-Identifier: Apache-2.0

setlocal
set "INCLUDE=%~dp0..\..\cuda\core\_include;%CUDA_HOME%\include;%INCLUDE%"
python "%~dp0build_tests.py"
set "BUILD_RESULT=%ERRORLEVEL%"
endlocal & exit /b %BUILD_RESULT%
