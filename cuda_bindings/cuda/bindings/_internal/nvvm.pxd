# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 11.0.3 to 12.8.0. Do not modify it directly.

from ..cynvvm cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef nvvmResult _nvvmVersion(int* major, int* minor) except* nogil
cdef nvvmResult _nvvmIRVersion(int* majorIR, int* minorIR, int* majorDbg, int* minorDbg) except* nogil
cdef nvvmResult _nvvmCreateProgram(nvvmProgram* prog) except* nogil
cdef nvvmResult _nvvmDestroyProgram(nvvmProgram* prog) except* nogil
cdef nvvmResult _nvvmAddModuleToProgram(nvvmProgram prog, const char* buffer, size_t size, const char* name) except* nogil
cdef nvvmResult _nvvmLazyAddModuleToProgram(nvvmProgram prog, const char* buffer, size_t size, const char* name) except* nogil
cdef nvvmResult _nvvmCompileProgram(nvvmProgram prog, int numOptions, const char** options) except* nogil
cdef nvvmResult _nvvmVerifyProgram(nvvmProgram prog, int numOptions, const char** options) except* nogil
cdef nvvmResult _nvvmGetCompiledResultSize(nvvmProgram prog, size_t* bufferSizeRet) except* nogil
cdef nvvmResult _nvvmGetCompiledResult(nvvmProgram prog, char* buffer) except* nogil
cdef nvvmResult _nvvmGetProgramLogSize(nvvmProgram prog, size_t* bufferSizeRet) except* nogil
cdef nvvmResult _nvvmGetProgramLog(nvvmProgram prog, char* buffer) except* nogil
