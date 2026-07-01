# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.0.1 to 13.3.0, generator version 0.3.1.dev1779+ga8cc71818.d20260626. Do not modify it directly.


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
cdef extern from 'nvvm.h':
    ctypedef enum nvvmResult:
        NVVM_SUCCESS
        NVVM_ERROR_OUT_OF_MEMORY
        NVVM_ERROR_PROGRAM_CREATION_FAILURE
        NVVM_ERROR_IR_VERSION_MISMATCH
        NVVM_ERROR_INVALID_INPUT
        NVVM_ERROR_INVALID_PROGRAM
        NVVM_ERROR_INVALID_IR
        NVVM_ERROR_INVALID_OPTION
        NVVM_ERROR_NO_MODULE_IN_PROGRAM
        NVVM_ERROR_COMPILATION
        NVVM_ERROR_CANCELLED
cdef enum: _NVVMRESULT_INTERNAL_LOADING_ERROR = -42


# types
cdef extern from 'nvvm.h':
    ctypedef void* nvvmProgram 'nvvmProgram'



###############################################################################
# Functions
###############################################################################

cdef const char* nvvmGetErrorString(nvvmResult result) except?NULL nogil
cdef nvvmResult nvvmVersion(int* major, int* minor) except?<nvvmResult>_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmIRVersion(int* majorIR, int* minorIR, int* majorDbg, int* minorDbg) except?<nvvmResult>_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmCreateProgram(nvvmProgram* prog) except?<nvvmResult>_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmDestroyProgram(nvvmProgram* prog) except?<nvvmResult>_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmAddModuleToProgram(nvvmProgram prog, const char* buffer, size_t size, const char* name) except?<nvvmResult>_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmLazyAddModuleToProgram(nvvmProgram prog, const char* buffer, size_t size, const char* name) except?<nvvmResult>_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmCompileProgram(nvvmProgram prog, int numOptions, const char** options) except?<nvvmResult>_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmVerifyProgram(nvvmProgram prog, int numOptions, const char** options) except?<nvvmResult>_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmGetCompiledResultSize(nvvmProgram prog, size_t* bufferSizeRet) except?<nvvmResult>_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmGetCompiledResult(nvvmProgram prog, char* buffer) except?<nvvmResult>_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmGetProgramLogSize(nvvmProgram prog, size_t* bufferSizeRet) except?<nvvmResult>_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmGetProgramLog(nvvmProgram prog, char* buffer) except?<nvvmResult>_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmLLVMVersion(const char* arch, int* major) except?<nvvmResult>_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
