# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.4.1 to 13.3.0, generator version 0.3.1.dev1779+ga8cc71818.d20260626. Do not modify it directly.

from libc.stdint cimport intptr_t, uint32_t


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
cdef extern from 'nvFatbin.h':
    ctypedef enum nvFatbinResult:
        NVFATBIN_SUCCESS
        NVFATBIN_ERROR_INTERNAL
        NVFATBIN_ERROR_ELF_ARCH_MISMATCH
        NVFATBIN_ERROR_ELF_SIZE_MISMATCH
        NVFATBIN_ERROR_MISSING_PTX_VERSION
        NVFATBIN_ERROR_NULL_POINTER
        NVFATBIN_ERROR_COMPRESSION_FAILED
        NVFATBIN_ERROR_COMPRESSED_SIZE_EXCEEDED
        NVFATBIN_ERROR_UNRECOGNIZED_OPTION
        NVFATBIN_ERROR_INVALID_ARCH
        NVFATBIN_ERROR_INVALID_NVVM
        NVFATBIN_ERROR_EMPTY_INPUT
        NVFATBIN_ERROR_MISSING_PTX_ARCH
        NVFATBIN_ERROR_PTX_ARCH_MISMATCH
        NVFATBIN_ERROR_MISSING_FATBIN
        NVFATBIN_ERROR_INVALID_INDEX
        NVFATBIN_ERROR_IDENTIFIER_REUSE
        NVFATBIN_ERROR_INTERNAL_PTX_OPTION
cdef enum: _NVFATBINRESULT_INTERNAL_LOADING_ERROR = -42


# types
cdef extern from 'nvFatbin.h':
    ctypedef void* nvFatbinHandle 'nvFatbinHandle'



###############################################################################
# Functions
###############################################################################

cdef const char* nvFatbinGetErrorString(nvFatbinResult result) except?NULL nogil
cdef nvFatbinResult nvFatbinCreate(nvFatbinHandle* handle_indirect, const char** options, size_t optionsCount) except?<nvFatbinResult>_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvFatbinResult nvFatbinDestroy(nvFatbinHandle* handle_indirect) except?<nvFatbinResult>_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvFatbinResult nvFatbinAddPTX(nvFatbinHandle handle, const char* code, size_t size, const char* arch, const char* identifier, const char* optionsCmdLine) except?<nvFatbinResult>_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvFatbinResult nvFatbinAddCubin(nvFatbinHandle handle, const void* code, size_t size, const char* arch, const char* identifier) except?<nvFatbinResult>_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvFatbinResult nvFatbinAddLTOIR(nvFatbinHandle handle, const void* code, size_t size, const char* arch, const char* identifier, const char* optionsCmdLine) except?<nvFatbinResult>_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvFatbinResult nvFatbinSize(nvFatbinHandle handle, size_t* size) except?<nvFatbinResult>_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvFatbinResult nvFatbinGet(nvFatbinHandle handle, void* buffer) except?<nvFatbinResult>_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvFatbinResult nvFatbinVersion(unsigned int* major, unsigned int* minor) except?<nvFatbinResult>_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvFatbinResult nvFatbinAddIndex(nvFatbinHandle handle, const void* code, size_t size, const char* identifier) except?<nvFatbinResult>_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvFatbinResult nvFatbinAddReloc(nvFatbinHandle handle, const void* code, size_t size) except?<nvFatbinResult>_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvFatbinResult nvFatbinAddTileIR(nvFatbinHandle handle, const void* code, size_t size, const char* identifier, const char* optionsCmdLine) except?<nvFatbinResult>_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
