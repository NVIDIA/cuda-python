# Copyright 2021-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
cimport cuda._cuda.cnvrtc as cnvrtc

cdef const char* nvrtcGetErrorString(nvrtcResult result) nogil except ?NULL:
    return cnvrtc._nvrtcGetErrorString(result)

cdef nvrtcResult nvrtcVersion(int* major, int* minor) nogil except ?NVRTC_ERROR_INVALID_INPUT:
    return cnvrtc._nvrtcVersion(major, minor)

cdef nvrtcResult nvrtcGetNumSupportedArchs(int* numArchs) nogil except ?NVRTC_ERROR_INVALID_INPUT:
    return cnvrtc._nvrtcGetNumSupportedArchs(numArchs)

cdef nvrtcResult nvrtcGetSupportedArchs(int* supportedArchs) nogil except ?NVRTC_ERROR_INVALID_INPUT:
    return cnvrtc._nvrtcGetSupportedArchs(supportedArchs)

cdef nvrtcResult nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames) nogil except ?NVRTC_ERROR_INVALID_INPUT:
    return cnvrtc._nvrtcCreateProgram(prog, src, name, numHeaders, headers, includeNames)

cdef nvrtcResult nvrtcDestroyProgram(nvrtcProgram* prog) nogil except ?NVRTC_ERROR_INVALID_INPUT:
    return cnvrtc._nvrtcDestroyProgram(prog)

cdef nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char** options) nogil except ?NVRTC_ERROR_INVALID_INPUT:
    return cnvrtc._nvrtcCompileProgram(prog, numOptions, options)

cdef nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet) nogil except ?NVRTC_ERROR_INVALID_INPUT:
    return cnvrtc._nvrtcGetPTXSize(prog, ptxSizeRet)

cdef nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* ptx) nogil except ?NVRTC_ERROR_INVALID_INPUT:
    return cnvrtc._nvrtcGetPTX(prog, ptx)

cdef nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, size_t* cubinSizeRet) nogil except ?NVRTC_ERROR_INVALID_INPUT:
    return cnvrtc._nvrtcGetCUBINSize(prog, cubinSizeRet)

cdef nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char* cubin) nogil except ?NVRTC_ERROR_INVALID_INPUT:
    return cnvrtc._nvrtcGetCUBIN(prog, cubin)

cdef nvrtcResult nvrtcGetNVVMSize(nvrtcProgram prog, size_t* nvvmSizeRet) nogil except ?NVRTC_ERROR_INVALID_INPUT:
    return cnvrtc._nvrtcGetNVVMSize(prog, nvvmSizeRet)

cdef nvrtcResult nvrtcGetNVVM(nvrtcProgram prog, char* nvvm) nogil except ?NVRTC_ERROR_INVALID_INPUT:
    return cnvrtc._nvrtcGetNVVM(prog, nvvm)

cdef nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet) nogil except ?NVRTC_ERROR_INVALID_INPUT:
    return cnvrtc._nvrtcGetProgramLogSize(prog, logSizeRet)

cdef nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char* log) nogil except ?NVRTC_ERROR_INVALID_INPUT:
    return cnvrtc._nvrtcGetProgramLog(prog, log)

cdef nvrtcResult nvrtcAddNameExpression(nvrtcProgram prog, const char* name_expression) nogil except ?NVRTC_ERROR_INVALID_INPUT:
    return cnvrtc._nvrtcAddNameExpression(prog, name_expression)

cdef nvrtcResult nvrtcGetLoweredName(nvrtcProgram prog, const char* name_expression, const char** lowered_name) nogil except ?NVRTC_ERROR_INVALID_INPUT:
    return cnvrtc._nvrtcGetLoweredName(prog, name_expression, lowered_name)
