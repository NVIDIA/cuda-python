# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated with version 13.0.0. Do not modify it directly.

from ._internal cimport nvfatbin as _nvfatbin


###############################################################################
# Wrapper functions
###############################################################################

cdef nvFatbinResult nvFatbinCreate(nvFatbinHandle* handle_indirect, const char** options, size_t optionsCount) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvfatbin._nvFatbinCreate(handle_indirect, options, optionsCount)


cdef nvFatbinResult nvFatbinDestroy(nvFatbinHandle* handle_indirect) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvfatbin._nvFatbinDestroy(handle_indirect)


cdef nvFatbinResult nvFatbinAddPTX(nvFatbinHandle handle, const char* code, size_t size, const char* arch, const char* identifier, const char* optionsCmdLine) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvfatbin._nvFatbinAddPTX(handle, code, size, arch, identifier, optionsCmdLine)


cdef nvFatbinResult nvFatbinSize(nvFatbinHandle handle, size_t* size) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvfatbin._nvFatbinSize(handle, size)


cdef nvFatbinResult nvFatbinGet(nvFatbinHandle handle, void* buffer) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvfatbin._nvFatbinGet(handle, buffer)


cdef nvFatbinResult nvFatbinVersion(unsigned int* major, unsigned int* minor) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvfatbin._nvFatbinVersion(major, minor)



