@echo off

REM SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
REM SPDX-License-Identifier: Apache-2.0

setlocal
	set CL=%CL% /I"%CUDA_HOME%\include"
	REM Use -j 1 to side-step any process-pool issues and ensure deterministic single-threaded builds
	cythonize -3 -j 1 -i -Xfreethreading_compatible=True %~dp0test_*.pyx
endlocal
