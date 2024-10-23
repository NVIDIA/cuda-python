# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.0.1 to 12.6.2. Do not modify it directly.

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

cdef nvJitLinkResult nvJitLinkCreate(nvJitLinkHandle* handle, uint32_t numOptions, const char** options) except* nogil
cdef nvJitLinkResult nvJitLinkDestroy(nvJitLinkHandle* handle) except* nogil
cdef nvJitLinkResult nvJitLinkAddData(nvJitLinkHandle handle, nvJitLinkInputType inputType, const void* data, size_t size, const char* name) except* nogil
cdef nvJitLinkResult nvJitLinkAddFile(nvJitLinkHandle handle, nvJitLinkInputType inputType, const char* fileName) except* nogil
cdef nvJitLinkResult nvJitLinkComplete(nvJitLinkHandle handle) except* nogil
cdef nvJitLinkResult nvJitLinkGetLinkedCubinSize(nvJitLinkHandle handle, size_t* size) except* nogil
cdef nvJitLinkResult nvJitLinkGetLinkedCubin(nvJitLinkHandle handle, void* cubin) except* nogil
cdef nvJitLinkResult nvJitLinkGetLinkedPtxSize(nvJitLinkHandle handle, size_t* size) except* nogil
cdef nvJitLinkResult nvJitLinkGetLinkedPtx(nvJitLinkHandle handle, char* ptx) except* nogil
cdef nvJitLinkResult nvJitLinkGetErrorLogSize(nvJitLinkHandle handle, size_t* size) except* nogil
cdef nvJitLinkResult nvJitLinkGetErrorLog(nvJitLinkHandle handle, char* log) except* nogil
cdef nvJitLinkResult nvJitLinkGetInfoLogSize(nvJitLinkHandle handle, size_t* size) except* nogil
cdef nvJitLinkResult nvJitLinkGetInfoLog(nvJitLinkHandle handle, char* log) except* nogil
cdef nvJitLinkResult nvJitLinkVersion(unsigned int* major, unsigned int* minor) except* nogil
