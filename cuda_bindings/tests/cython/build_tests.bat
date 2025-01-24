@echo off
setlocal
	set CL=%CL% /I"%CUDA_HOME%\include"
	cythonize -3 -i %~dp0test_*.pyx
endlocal
