# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 11.0.3 to 12.8.0. Do not modify it directly.


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


# types
ctypedef void* nvvmProgram 'nvvmProgram'


###############################################################################
# Functions
###############################################################################

cdef nvvmResult nvvmVersion(int* major, int* minor) except* nogil
cdef nvvmResult nvvmIRVersion(int* majorIR, int* minorIR, int* majorDbg, int* minorDbg) except* nogil
cdef nvvmResult nvvmCreateProgram(nvvmProgram* prog) except* nogil
cdef nvvmResult nvvmDestroyProgram(nvvmProgram* prog) except* nogil
cdef nvvmResult nvvmAddModuleToProgram(nvvmProgram prog, const char* buffer, size_t size, const char* name) except* nogil
cdef nvvmResult nvvmLazyAddModuleToProgram(nvvmProgram prog, const char* buffer, size_t size, const char* name) except* nogil
cdef nvvmResult nvvmCompileProgram(nvvmProgram prog, int numOptions, const char** options) except* nogil
cdef nvvmResult nvvmVerifyProgram(nvvmProgram prog, int numOptions, const char** options) except* nogil
cdef nvvmResult nvvmGetCompiledResultSize(nvvmProgram prog, size_t* bufferSizeRet) except* nogil
cdef nvvmResult nvvmGetCompiledResult(nvvmProgram prog, char* buffer) except* nogil
cdef nvvmResult nvvmGetProgramLogSize(nvvmProgram prog, size_t* bufferSizeRet) except* nogil
cdef nvvmResult nvvmGetProgramLog(nvvmProgram prog, char* buffer) except* nogil
