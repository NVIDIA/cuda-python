# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.0.1 to 13.0.1. Do not modify it directly.

from libc.stdint cimport intptr_t, uint32_t


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum nvJitLinkResult "nvJitLinkResult":
    NVJITLINK_SUCCESS "NVJITLINK_SUCCESS" = 0
    NVJITLINK_ERROR_UNRECOGNIZED_OPTION "NVJITLINK_ERROR_UNRECOGNIZED_OPTION"
    NVJITLINK_ERROR_MISSING_ARCH "NVJITLINK_ERROR_MISSING_ARCH"
    NVJITLINK_ERROR_INVALID_INPUT "NVJITLINK_ERROR_INVALID_INPUT"
    NVJITLINK_ERROR_PTX_COMPILE "NVJITLINK_ERROR_PTX_COMPILE"
    NVJITLINK_ERROR_NVVM_COMPILE "NVJITLINK_ERROR_NVVM_COMPILE"
    NVJITLINK_ERROR_INTERNAL "NVJITLINK_ERROR_INTERNAL"
    NVJITLINK_ERROR_THREADPOOL "NVJITLINK_ERROR_THREADPOOL"
    NVJITLINK_ERROR_UNRECOGNIZED_INPUT "NVJITLINK_ERROR_UNRECOGNIZED_INPUT"
    NVJITLINK_ERROR_FINALIZE "NVJITLINK_ERROR_FINALIZE"
    NVJITLINK_ERROR_NULL_INPUT "NVJITLINK_ERROR_NULL_INPUT"
    NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS "NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS"
    NVJITLINK_ERROR_INCORRECT_INPUT_TYPE "NVJITLINK_ERROR_INCORRECT_INPUT_TYPE"
    NVJITLINK_ERROR_ARCH_MISMATCH "NVJITLINK_ERROR_ARCH_MISMATCH"
    NVJITLINK_ERROR_OUTDATED_LIBRARY "NVJITLINK_ERROR_OUTDATED_LIBRARY"
    NVJITLINK_ERROR_MISSING_FATBIN "NVJITLINK_ERROR_MISSING_FATBIN"
    NVJITLINK_ERROR_UNRECOGNIZED_ARCH "NVJITLINK_ERROR_UNRECOGNIZED_ARCH"
    NVJITLINK_ERROR_UNSUPPORTED_ARCH "NVJITLINK_ERROR_UNSUPPORTED_ARCH"
    NVJITLINK_ERROR_LTO_NOT_ENABLED "NVJITLINK_ERROR_LTO_NOT_ENABLED"
    _NVJITLINKRESULT_INTERNAL_LOADING_ERROR "_NVJITLINKRESULT_INTERNAL_LOADING_ERROR" = -42

ctypedef enum nvJitLinkInputType "nvJitLinkInputType":
    NVJITLINK_INPUT_NONE "NVJITLINK_INPUT_NONE" = 0
    NVJITLINK_INPUT_CUBIN "NVJITLINK_INPUT_CUBIN" = 1
    NVJITLINK_INPUT_PTX "NVJITLINK_INPUT_PTX"
    NVJITLINK_INPUT_LTOIR "NVJITLINK_INPUT_LTOIR"
    NVJITLINK_INPUT_FATBIN "NVJITLINK_INPUT_FATBIN"
    NVJITLINK_INPUT_OBJECT "NVJITLINK_INPUT_OBJECT"
    NVJITLINK_INPUT_LIBRARY "NVJITLINK_INPUT_LIBRARY"
    NVJITLINK_INPUT_INDEX "NVJITLINK_INPUT_INDEX"
    NVJITLINK_INPUT_ANY "NVJITLINK_INPUT_ANY" = 10


# types
ctypedef void* nvJitLinkHandle 'nvJitLinkHandle'


###############################################################################
# Functions
###############################################################################

cdef nvJitLinkResult nvJitLinkCreate(nvJitLinkHandle* handle, uint32_t numOptions, const char** options) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult nvJitLinkDestroy(nvJitLinkHandle* handle) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult nvJitLinkAddData(nvJitLinkHandle handle, nvJitLinkInputType inputType, const void* data, size_t size, const char* name) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult nvJitLinkAddFile(nvJitLinkHandle handle, nvJitLinkInputType inputType, const char* fileName) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult nvJitLinkComplete(nvJitLinkHandle handle) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult nvJitLinkGetLinkedCubinSize(nvJitLinkHandle handle, size_t* size) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult nvJitLinkGetLinkedCubin(nvJitLinkHandle handle, void* cubin) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult nvJitLinkGetLinkedPtxSize(nvJitLinkHandle handle, size_t* size) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult nvJitLinkGetLinkedPtx(nvJitLinkHandle handle, char* ptx) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult nvJitLinkGetErrorLogSize(nvJitLinkHandle handle, size_t* size) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult nvJitLinkGetErrorLog(nvJitLinkHandle handle, char* log) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult nvJitLinkGetInfoLogSize(nvJitLinkHandle handle, size_t* size) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult nvJitLinkGetInfoLog(nvJitLinkHandle handle, char* log) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult nvJitLinkVersion(unsigned int* major, unsigned int* minor) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
