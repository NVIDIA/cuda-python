# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.0.1 to 13.3.0. Do not modify it directly.
# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=7618d44448c6e1142afb5ad6cb3b7e15e1d775705ff4aaadbb8fe8744cccb1a4


# <<<< PREAMBLE CONTENT >>>>

from libc.stdint cimport uint32_t


# <<<< END OF PREAMBLE CONTENT >>>>

from ._internal cimport nvjitlink as _nvjitlink


###############################################################################
# Wrapper functions
###############################################################################

cdef nvJitLinkResult nvJitLinkCreate(nvJitLinkHandle* handle, uint32_t numOptions, const char** options) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvjitlink._nvJitLinkCreate(handle, numOptions, options)


cdef nvJitLinkResult nvJitLinkDestroy(nvJitLinkHandle* handle) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvjitlink._nvJitLinkDestroy(handle)


cdef nvJitLinkResult nvJitLinkAddData(nvJitLinkHandle handle, nvJitLinkInputType inputType, const void* data, size_t size, const char* name) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvjitlink._nvJitLinkAddData(handle, inputType, data, size, name)


cdef nvJitLinkResult nvJitLinkAddFile(nvJitLinkHandle handle, nvJitLinkInputType inputType, const char* fileName) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvjitlink._nvJitLinkAddFile(handle, inputType, fileName)


cdef nvJitLinkResult nvJitLinkComplete(nvJitLinkHandle handle) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvjitlink._nvJitLinkComplete(handle)


cdef nvJitLinkResult nvJitLinkGetLinkedCubinSize(nvJitLinkHandle handle, size_t* size) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvjitlink._nvJitLinkGetLinkedCubinSize(handle, size)


cdef nvJitLinkResult nvJitLinkGetLinkedCubin(nvJitLinkHandle handle, void* cubin) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvjitlink._nvJitLinkGetLinkedCubin(handle, cubin)


cdef nvJitLinkResult nvJitLinkGetLinkedPtxSize(nvJitLinkHandle handle, size_t* size) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvjitlink._nvJitLinkGetLinkedPtxSize(handle, size)


cdef nvJitLinkResult nvJitLinkGetLinkedPtx(nvJitLinkHandle handle, char* ptx) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvjitlink._nvJitLinkGetLinkedPtx(handle, ptx)


cdef nvJitLinkResult nvJitLinkGetErrorLogSize(nvJitLinkHandle handle, size_t* size) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvjitlink._nvJitLinkGetErrorLogSize(handle, size)


cdef nvJitLinkResult nvJitLinkGetErrorLog(nvJitLinkHandle handle, char* log) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvjitlink._nvJitLinkGetErrorLog(handle, log)


cdef nvJitLinkResult nvJitLinkGetInfoLogSize(nvJitLinkHandle handle, size_t* size) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvjitlink._nvJitLinkGetInfoLogSize(handle, size)


cdef nvJitLinkResult nvJitLinkGetInfoLog(nvJitLinkHandle handle, char* log) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvjitlink._nvJitLinkGetInfoLog(handle, log)


cdef nvJitLinkResult nvJitLinkVersion(unsigned int* major, unsigned int* minor) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvjitlink._nvJitLinkVersion(major, minor)


cdef nvJitLinkResult nvJitLinkGetLinkedLTOIRSize(nvJitLinkHandle handle, size_t* size) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvjitlink._nvJitLinkGetLinkedLTOIRSize(handle, size)


cdef nvJitLinkResult nvJitLinkGetLinkedLTOIR(nvJitLinkHandle handle, void* ltoir) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvjitlink._nvJitLinkGetLinkedLTOIR(handle, ltoir)
