# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.0.1 to 13.0.1. Do not modify it directly.


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum nvvmResult "nvvmResult":
    NVVM_SUCCESS "NVVM_SUCCESS" = 0
    NVVM_ERROR_OUT_OF_MEMORY "NVVM_ERROR_OUT_OF_MEMORY" = 1
    NVVM_ERROR_PROGRAM_CREATION_FAILURE "NVVM_ERROR_PROGRAM_CREATION_FAILURE" = 2
    NVVM_ERROR_IR_VERSION_MISMATCH "NVVM_ERROR_IR_VERSION_MISMATCH" = 3
    NVVM_ERROR_INVALID_INPUT "NVVM_ERROR_INVALID_INPUT" = 4
    NVVM_ERROR_INVALID_PROGRAM "NVVM_ERROR_INVALID_PROGRAM" = 5
    NVVM_ERROR_INVALID_IR "NVVM_ERROR_INVALID_IR" = 6
    NVVM_ERROR_INVALID_OPTION "NVVM_ERROR_INVALID_OPTION" = 7
    NVVM_ERROR_NO_MODULE_IN_PROGRAM "NVVM_ERROR_NO_MODULE_IN_PROGRAM" = 8
    NVVM_ERROR_COMPILATION "NVVM_ERROR_COMPILATION" = 9
    NVVM_ERROR_CANCELLED "NVVM_ERROR_CANCELLED" = 10
    _NVVMRESULT_INTERNAL_LOADING_ERROR "_NVVMRESULT_INTERNAL_LOADING_ERROR" = -42


# types
ctypedef void* nvvmProgram 'nvvmProgram'


###############################################################################
# Functions
###############################################################################

cdef const char* nvvmGetErrorString(nvvmResult result) except?NULL nogil
cdef nvvmResult nvvmVersion(int* major, int* minor) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmIRVersion(int* majorIR, int* minorIR, int* majorDbg, int* minorDbg) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmCreateProgram(nvvmProgram* prog) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmDestroyProgram(nvvmProgram* prog) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmAddModuleToProgram(nvvmProgram prog, const char* buffer, size_t size, const char* name) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmLazyAddModuleToProgram(nvvmProgram prog, const char* buffer, size_t size, const char* name) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmCompileProgram(nvvmProgram prog, int numOptions, const char** options) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmVerifyProgram(nvvmProgram prog, int numOptions, const char** options) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmGetCompiledResultSize(nvvmProgram prog, size_t* bufferSizeRet) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmGetCompiledResult(nvvmProgram prog, char* buffer) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmGetProgramLogSize(nvvmProgram prog, size_t* bufferSizeRet) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvvmResult nvvmGetProgramLog(nvvmProgram prog, char* buffer) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil
