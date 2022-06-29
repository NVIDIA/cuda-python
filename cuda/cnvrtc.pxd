# Copyright 2021-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

cdef extern from "nvrtc.h":

    ctypedef enum nvrtcResult:
        NVRTC_SUCCESS = 0
        NVRTC_ERROR_OUT_OF_MEMORY = 1
        NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2
        NVRTC_ERROR_INVALID_INPUT = 3
        NVRTC_ERROR_INVALID_PROGRAM = 4
        NVRTC_ERROR_INVALID_OPTION = 5
        NVRTC_ERROR_COMPILATION = 6
        NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7
        NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8
        NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9
        NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10
        NVRTC_ERROR_INTERNAL_ERROR = 11

    cdef struct _nvrtcProgram:
        pass
    ctypedef _nvrtcProgram* nvrtcProgram

cdef const char* nvrtcGetErrorString(nvrtcResult result) nogil except ?NULL

cdef nvrtcResult nvrtcVersion(int* major, int* minor) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult nvrtcGetNumSupportedArchs(int* numArchs) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult nvrtcGetSupportedArchs(int* supportedArchs) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult nvrtcDestroyProgram(nvrtcProgram* prog) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char** options) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* ptx) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, size_t* cubinSizeRet) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char* cubin) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult nvrtcGetNVVMSize(nvrtcProgram prog, size_t* nvvmSizeRet) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult nvrtcGetNVVM(nvrtcProgram prog, char* nvvm) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char* log) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult nvrtcAddNameExpression(nvrtcProgram prog, const char* name_expression) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult nvrtcGetLoweredName(nvrtcProgram prog, const char* name_expression, const char** lowered_name) nogil except ?NVRTC_ERROR_INVALID_INPUT
