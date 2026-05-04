# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.9.0 to 13.2.0, generator version 0.3.1.dev1630+gadce055ea.d20260422. Do not modify it directly.

from ..cynvrtc cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef const char* _nvrtcGetErrorString(nvrtcResult result) except?NULL nogil
cdef nvrtcResult _nvrtcVersion(int* major, int* minor) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcGetNumSupportedArchs(int* numArchs) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcGetSupportedArchs(int* supportedArchs) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcDestroyProgram(nvrtcProgram* prog) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char** options) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcGetPTX(nvrtcProgram prog, char* ptx) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcGetCUBINSize(nvrtcProgram prog, size_t* cubinSizeRet) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcGetCUBIN(nvrtcProgram prog, char* cubin) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcGetLTOIRSize(nvrtcProgram prog, size_t* LTOIRSizeRet) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcGetLTOIR(nvrtcProgram prog, char* LTOIR) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcGetOptiXIRSize(nvrtcProgram prog, size_t* optixirSizeRet) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcGetOptiXIR(nvrtcProgram prog, char* optixir) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcGetProgramLog(nvrtcProgram prog, char* log) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcAddNameExpression(nvrtcProgram prog, const char* name_expression) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcGetLoweredName(nvrtcProgram prog, const char* name_expression, const char** lowered_name) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcGetPCHHeapSize(size_t* ret) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcSetPCHHeapSize(size_t size) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcGetPCHCreateStatus(nvrtcProgram prog) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcGetPCHHeapSizeRequired(nvrtcProgram prog, size_t* size) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcSetFlowCallback(nvrtcProgram prog, void * callback, void* payload) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcGetTileIRSize(nvrtcProgram prog, size_t* TileIRSizeRet) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvrtcResult _nvrtcGetTileIR(nvrtcProgram prog, char* TileIR) except?<nvrtcResult>_NVRTCRESULT_INTERNAL_LOADING_ERROR nogil
