# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.0.1 to 13.0.1. Do not modify it directly.

from ..cynvvm cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef const char* _nvvmGetErrorString(nvvmResult result) except?NULL nogil
cdef nvvmResult _nvvmVersion(int* major, int* minor) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult _nvvmIRVersion(int* majorIR, int* minorIR, int* majorDbg, int* minorDbg) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult _nvvmCreateProgram(nvvmProgram* prog) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult _nvvmDestroyProgram(nvvmProgram* prog) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult _nvvmAddModuleToProgram(nvvmProgram prog, const char* buffer, size_t size, const char* name) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult _nvvmLazyAddModuleToProgram(nvvmProgram prog, const char* buffer, size_t size, const char* name) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult _nvvmCompileProgram(nvvmProgram prog, int numOptions, const char** options) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult _nvvmVerifyProgram(nvvmProgram prog, int numOptions, const char** options) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult _nvvmGetCompiledResultSize(nvvmProgram prog, size_t* bufferSizeRet) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult _nvvmGetCompiledResult(nvvmProgram prog, char* buffer) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult _nvvmGetProgramLogSize(nvvmProgram prog, size_t* bufferSizeRet) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult _nvvmGetProgramLog(nvvmProgram prog, char* buffer) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
