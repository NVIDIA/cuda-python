# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.9.0 to 13.2.0, generator version 0.3.1.dev1630+gadce055ea.d20260422. Do not modify it directly.

from ._internal cimport nvrtc as _nvrtc

cdef const char* nvrtcGetErrorString(nvrtcResult result) except?NULL nogil:
    return _nvrtc._nvrtcGetErrorString(result)


cdef nvrtcResult nvrtcVersion(int* major, int* minor) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcVersion(major, minor)


cdef nvrtcResult nvrtcGetNumSupportedArchs(int* numArchs) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcGetNumSupportedArchs(numArchs)


cdef nvrtcResult nvrtcGetSupportedArchs(int* supportedArchs) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcGetSupportedArchs(supportedArchs)


cdef nvrtcResult nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name, int numHeaders, const char* const* headers, const char* const* includeNames) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcCreateProgram(prog, src, name, numHeaders, headers, includeNames)


cdef nvrtcResult nvrtcDestroyProgram(nvrtcProgram* prog) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcDestroyProgram(prog)


cdef nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char* const* options) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcCompileProgram(prog, numOptions, options)


cdef nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcGetPTXSize(prog, ptxSizeRet)


cdef nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* ptx) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcGetPTX(prog, ptx)


cdef nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, size_t* cubinSizeRet) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcGetCUBINSize(prog, cubinSizeRet)


cdef nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char* cubin) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcGetCUBIN(prog, cubin)


cdef nvrtcResult nvrtcGetLTOIRSize(nvrtcProgram prog, size_t* LTOIRSizeRet) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcGetLTOIRSize(prog, LTOIRSizeRet)


cdef nvrtcResult nvrtcGetLTOIR(nvrtcProgram prog, char* LTOIR) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcGetLTOIR(prog, LTOIR)


cdef nvrtcResult nvrtcGetOptiXIRSize(nvrtcProgram prog, size_t* optixirSizeRet) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcGetOptiXIRSize(prog, optixirSizeRet)


cdef nvrtcResult nvrtcGetOptiXIR(nvrtcProgram prog, char* optixir) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcGetOptiXIR(prog, optixir)


cdef nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcGetProgramLogSize(prog, logSizeRet)


cdef nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char* log) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcGetProgramLog(prog, log)


cdef nvrtcResult nvrtcAddNameExpression(nvrtcProgram prog, const char* const name_expression) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcAddNameExpression(prog, name_expression)


cdef nvrtcResult nvrtcGetLoweredName(nvrtcProgram prog, const char* const name_expression, const char** lowered_name) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcGetLoweredName(prog, name_expression, lowered_name)


cdef nvrtcResult nvrtcGetPCHHeapSize(size_t* ret) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcGetPCHHeapSize(ret)


cdef nvrtcResult nvrtcSetPCHHeapSize(size_t size) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcSetPCHHeapSize(size)


cdef nvrtcResult nvrtcGetPCHCreateStatus(nvrtcProgram prog) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcGetPCHCreateStatus(prog)


cdef nvrtcResult nvrtcGetPCHHeapSizeRequired(nvrtcProgram prog, size_t* size) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcGetPCHHeapSizeRequired(prog, size)


cdef nvrtcResult nvrtcSetFlowCallback(nvrtcProgram prog, void * callback, void* payload) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcSetFlowCallback(prog, callback, payload)


cdef nvrtcResult nvrtcGetTileIRSize(nvrtcProgram prog, size_t* TileIRSizeRet) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcGetTileIRSize(prog, TileIRSizeRet)


cdef nvrtcResult nvrtcGetTileIR(nvrtcProgram prog, char* TileIR) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvrtc._nvrtcGetTileIR(prog, TileIR)
