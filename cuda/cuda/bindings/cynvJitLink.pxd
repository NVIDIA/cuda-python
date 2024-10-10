# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.0.1 to 12.4.1. Do not modify it directly.


from libc.stdint cimport int64_t


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums



# types
cdef extern from *:
    """
    #include <driver_types.h>
    #include <library_types.h>
    #include <cuComplex.h>
    """
    ctypedef void* cudaStream_t 'cudaStream_t'





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
