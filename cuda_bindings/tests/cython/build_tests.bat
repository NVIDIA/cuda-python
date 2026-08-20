@echo off

REM SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
REM SPDX-License-Identifier: Apache-2.0

setlocal
set CL=%CL% /I"%CUDA_HOME%\include"
REM The Python driver provides Cython's .pxd include path and builds in this
REM directory so Windows does not duplicate the checkout path in link outputs.
python "%~dp0build_tests.py"
set "BUILD_RESULT=%ERRORLEVEL%"
endlocal & exit /b %BUILD_RESULT%
