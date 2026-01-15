# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.4.1 to 13.1.0. Do not modify it directly.

from ._internal cimport nvfatbin as _nvfatbin


###############################################################################
# Wrapper functions
###############################################################################

cdef const char* nvFatbinGetErrorString(nvFatbinResult result) except?NULL nogil:
    return _nvfatbin._nvFatbinGetErrorString(result)


cdef nvFatbinResult nvFatbinCreate(nvFatbinHandle* handle_indirect, const char** options, size_t optionsCount) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvfatbin._nvFatbinCreate(handle_indirect, options, optionsCount)


cdef nvFatbinResult nvFatbinDestroy(nvFatbinHandle* handle_indirect) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvfatbin._nvFatbinDestroy(handle_indirect)


cdef nvFatbinResult nvFatbinAddPTX(nvFatbinHandle handle, const char* code, size_t size, const char* arch, const char* identifier, const char* optionsCmdLine) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvfatbin._nvFatbinAddPTX(handle, code, size, arch, identifier, optionsCmdLine)


cdef nvFatbinResult nvFatbinAddCubin(nvFatbinHandle handle, const void* code, size_t size, const char* arch, const char* identifier) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvfatbin._nvFatbinAddCubin(handle, code, size, arch, identifier)


cdef nvFatbinResult nvFatbinAddLTOIR(nvFatbinHandle handle, const void* code, size_t size, const char* arch, const char* identifier, const char* optionsCmdLine) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvfatbin._nvFatbinAddLTOIR(handle, code, size, arch, identifier, optionsCmdLine)


cdef nvFatbinResult nvFatbinSize(nvFatbinHandle handle, size_t* size) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvfatbin._nvFatbinSize(handle, size)


cdef nvFatbinResult nvFatbinGet(nvFatbinHandle handle, void* buffer) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvfatbin._nvFatbinGet(handle, buffer)


cdef nvFatbinResult nvFatbinVersion(unsigned int* major, unsigned int* minor) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvfatbin._nvFatbinVersion(major, minor)


cdef nvFatbinResult nvFatbinAddReloc(nvFatbinHandle handle, const void* code, size_t size) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvfatbin._nvFatbinAddReloc(handle, code, size)


cdef nvFatbinResult nvFatbinAddTileIR(nvFatbinHandle handle, const void* code, size_t size, const char* identifier, const char* optionsCmdLine) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvfatbin._nvFatbinAddTileIR(handle, code, size, identifier, optionsCmdLine)
