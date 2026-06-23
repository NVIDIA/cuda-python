# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.9.0 to 13.3.0, generator version 0.3.1.dev1630+gadce055ea.d20260422. Do not modify it directly.

from libc.stdint cimport uint32_t, uint64_t


# ENUMS
cdef extern from 'nvrtc.h':
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
        NVRTC_ERROR_BUSY
cdef enum: _NVRTCRESULT_INTERNAL_LOADING_ERROR = -42


cdef enum: NVRTC_INSTALL_HEADERS_SKIP_IF_EXISTS = 0
cdef enum: NVRTC_INSTALL_HEADERS_FORCE_OVERWRITE = 1
cdef enum: NVRTC_INSTALL_HEADERS_NO_WAIT = 2


# TYPES
cdef extern from 'nvrtc.h':
    ctypedef struct _nvrtcProgram:
        pass
    ctypedef _nvrtcProgram* nvrtcProgram 'nvrtcProgram'


cdef extern from 'nvrtc.h':
    ctypedef struct nvrtcBundledHeadersInfo 'nvrtcBundledHeadersInfo':
        int available
        size_t compressedSize
        size_t uncompressedSize
        int cudaVersionMajor
        int cudaVersionMinor
        unsigned int numFiles


# FUNCTIONS
cdef const char* nvrtcGetErrorString(nvrtcResult result) except?NULL nogil
cdef nvrtcResult nvrtcVersion(int* major, int* minor) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcGetNumSupportedArchs(int* numArchs) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcGetSupportedArchs(int* supportedArchs) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcDestroyProgram(nvrtcProgram* prog) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char** options) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* ptx) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, size_t* cubinSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char* cubin) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcGetLTOIRSize(nvrtcProgram prog, size_t* LTOIRSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcGetLTOIR(nvrtcProgram prog, char* LTOIR) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcGetOptiXIRSize(nvrtcProgram prog, size_t* optixirSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcGetOptiXIR(nvrtcProgram prog, char* optixir) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char* log) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcAddNameExpression(nvrtcProgram prog, const char* name_expression) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcGetLoweredName(nvrtcProgram prog, const char* name_expression, const char** lowered_name) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcGetPCHHeapSize(size_t* ret) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcSetPCHHeapSize(size_t size) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcGetPCHCreateStatus(nvrtcProgram prog) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcGetPCHHeapSizeRequired(nvrtcProgram prog, size_t* size) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcSetFlowCallback(nvrtcProgram prog, void * callback, void* payload) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcGetTileIRSize(nvrtcProgram prog, size_t* TileIRSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcGetTileIR(nvrtcProgram prog, char* TileIR) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcInstallBundledHeaders(const char* installPath, unsigned int flags, const char** errorLog) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcGetBundledHeadersInfo(nvrtcBundledHeadersInfo* info, const char** errorLog) except ?NVRTC_ERROR_INVALID_INPUT nogil
cdef nvrtcResult nvrtcRemoveBundledHeaders(const char* installPath, const char** errorLog) except ?NVRTC_ERROR_INVALID_INPUT nogil
