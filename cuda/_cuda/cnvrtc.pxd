# Copyright 2021-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
from cuda.cnvrtc cimport *

cdef const char* _nvrtcGetErrorString(nvrtcResult result) nogil except ?NULL

cdef nvrtcResult _nvrtcVersion(int* major, int* minor) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult _nvrtcGetNumSupportedArchs(int* numArchs) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult _nvrtcGetSupportedArchs(int* supportedArchs) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult _nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult _nvrtcDestroyProgram(nvrtcProgram* prog) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult _nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char** options) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult _nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult _nvrtcGetPTX(nvrtcProgram prog, char* ptx) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult _nvrtcGetCUBINSize(nvrtcProgram prog, size_t* cubinSizeRet) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult _nvrtcGetCUBIN(nvrtcProgram prog, char* cubin) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult _nvrtcGetNVVMSize(nvrtcProgram prog, size_t* nvvmSizeRet) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult _nvrtcGetNVVM(nvrtcProgram prog, char* nvvm) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult _nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult _nvrtcGetProgramLog(nvrtcProgram prog, char* log) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult _nvrtcAddNameExpression(nvrtcProgram prog, const char* name_expression) nogil except ?NVRTC_ERROR_INVALID_INPUT

cdef nvrtcResult _nvrtcGetLoweredName(nvrtcProgram prog, const char* name_expression, const char** lowered_name) nogil except ?NVRTC_ERROR_INVALID_INPUT
