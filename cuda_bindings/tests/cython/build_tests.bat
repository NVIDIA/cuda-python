@echo off

REM SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
REM SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

setlocal
	set CL=%CL% /I"%CUDA_HOME%\include"
	cythonize -3 -i -Xfreethreading_compatible=True %~dp0test_*.pyx
endlocal
