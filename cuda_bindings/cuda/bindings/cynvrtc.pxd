# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.9.0 to 13.2.0, generator version 0.3.1.dev1630+gadce055ea.d20260422. Do not modify it directly.

from libc.stdint cimport uint32_t, uint64_t


# ENUMS
cdef extern from '<nvrtc.h>':
    ctypedef enum nvrtcResult:
        NVRTC_SUCCESS
        NVRTC_ERROR_OUT_OF_MEMORY
        NVRTC_ERROR_PROGRAM_CREATION_FAILURE
        NVRTC_ERROR_INVALID_INPUT
        NVRTC_ERROR_INVALID_PROGRAM
        NVRTC_ERROR_INVALID_OPTION
        NVRTC_ERROR_COMPILATION
        NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
        NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
        NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
        NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID
        NVRTC_ERROR_INTERNAL_ERROR
        NVRTC_ERROR_TIME_FILE_WRITE_FAILED
        NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED
        NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED
        NVRTC_ERROR_PCH_CREATE
        NVRTC_ERROR_CANCELLED
        NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED
cdef enum: _NVRTCRESULT_INTERNAL_LOADING_ERROR = -42


# TYPES
cdef extern from '<nvrtc.h>':
    ctypedef void* nvrtcProgram 'nvrtcProgram'



# FUNCTIONS
cdef const char* nvrtcGetErrorString(nvrtcResult result) except?NULL nogil
cdef nvrtcResult nvrtcVersion(int* major, int* minor) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcGetNumSupportedArchs(int* numArchs) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcGetSupportedArchs(int* supportedArchs) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name, int numHeaders, const char* const* headers, const char* const* includeNames) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcDestroyProgram(nvrtcProgram* prog) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char* const* options) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* ptx) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, size_t* cubinSizeRet) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char* cubin) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcGetLTOIRSize(nvrtcProgram prog, size_t* LTOIRSizeRet) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcGetLTOIR(nvrtcProgram prog, char* LTOIR) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcGetOptiXIRSize(nvrtcProgram prog, size_t* optixirSizeRet) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcGetOptiXIR(nvrtcProgram prog, char* optixir) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char* log) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcAddNameExpression(nvrtcProgram prog, const char* const name_expression) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcGetLoweredName(nvrtcProgram prog, const char* const name_expression, const char** lowered_name) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcGetPCHHeapSize(size_t* ret) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcSetPCHHeapSize(size_t size) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcGetPCHCreateStatus(nvrtcProgram prog) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcGetPCHHeapSizeRequired(nvrtcProgram prog, size_t* size) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcSetFlowCallback(nvrtcProgram prog, void * callback, void* payload) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcGetTileIRSize(nvrtcProgram prog, size_t* TileIRSizeRet) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult nvrtcGetTileIR(nvrtcProgram prog, char* TileIR) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
