# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.0.76 to 12.6.77. Do not modify it directly.

from ._internal cimport nvjitlink as _nvjitlink
<<<<<<< HEAD
<<<<<<< HEAD
from libc.stdint cimport uint32_t
=======
>>>>>>> 5d60eb1 (more changes)
=======
from libc.stdint cimport uint32_t
>>>>>>> 8c4029f (working)


###############################################################################
# Wrapper functions
###############################################################################

cdef nvJitLinkResult nvJitLinkCreate(nvJitLinkHandle* handle, uint32_t numOptions, const char** options) except* nogil:
    return _nvjitlink._nvJitLinkCreate(handle, numOptions, options)


cdef nvJitLinkResult nvJitLinkDestroy(nvJitLinkHandle* handle) except* nogil:
    return _nvjitlink._nvJitLinkDestroy(handle)


cdef nvJitLinkResult nvJitLinkAddData(nvJitLinkHandle handle, nvJitLinkInputType inputType, const void* data, size_t size, const char* name) except* nogil:
    return _nvjitlink._nvJitLinkAddData(handle, inputType, data, size, name)


cdef nvJitLinkResult nvJitLinkAddFile(nvJitLinkHandle handle, nvJitLinkInputType inputType, const char* fileName) except* nogil:
    return _nvjitlink._nvJitLinkAddFile(handle, inputType, fileName)


cdef nvJitLinkResult nvJitLinkComplete(nvJitLinkHandle handle) except* nogil:
    return _nvjitlink._nvJitLinkComplete(handle)


cdef nvJitLinkResult nvJitLinkGetLinkedCubinSize(nvJitLinkHandle handle, size_t* size) except* nogil:
    return _nvjitlink._nvJitLinkGetLinkedCubinSize(handle, size)


cdef nvJitLinkResult nvJitLinkGetLinkedCubin(nvJitLinkHandle handle, void* cubin) except* nogil:
    return _nvjitlink._nvJitLinkGetLinkedCubin(handle, cubin)


cdef nvJitLinkResult nvJitLinkGetLinkedPtxSize(nvJitLinkHandle handle, size_t* size) except* nogil:
    return _nvjitlink._nvJitLinkGetLinkedPtxSize(handle, size)


cdef nvJitLinkResult nvJitLinkGetLinkedPtx(nvJitLinkHandle handle, char* ptx) except* nogil:
    return _nvjitlink._nvJitLinkGetLinkedPtx(handle, ptx)


cdef nvJitLinkResult nvJitLinkGetErrorLogSize(nvJitLinkHandle handle, size_t* size) except* nogil:
    return _nvjitlink._nvJitLinkGetErrorLogSize(handle, size)


cdef nvJitLinkResult nvJitLinkGetErrorLog(nvJitLinkHandle handle, char* log) except* nogil:
    return _nvjitlink._nvJitLinkGetErrorLog(handle, log)


cdef nvJitLinkResult nvJitLinkGetInfoLogSize(nvJitLinkHandle handle, size_t* size) except* nogil:
    return _nvjitlink._nvJitLinkGetInfoLogSize(handle, size)


cdef nvJitLinkResult nvJitLinkGetInfoLog(nvJitLinkHandle handle, char* log) except* nogil:
    return _nvjitlink._nvJitLinkGetInfoLog(handle, log)