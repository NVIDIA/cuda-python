# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.0.1 to 13.3.0. Do not modify it directly.

# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=fd32577c0d6b922ff30c56dc4f3dcbed2251c393098c27d972ccc8688564fa50
from ..cynvjitlink cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef nvJitLinkResult _nvJitLinkCreate(nvJitLinkHandle* handle, uint32_t numOptions, const char** options) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult _nvJitLinkDestroy(nvJitLinkHandle* handle) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult _nvJitLinkAddData(nvJitLinkHandle handle, nvJitLinkInputType inputType, const void* data, size_t size, const char* name) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult _nvJitLinkAddFile(nvJitLinkHandle handle, nvJitLinkInputType inputType, const char* fileName) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult _nvJitLinkComplete(nvJitLinkHandle handle) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult _nvJitLinkGetLinkedCubinSize(nvJitLinkHandle handle, size_t* size) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult _nvJitLinkGetLinkedCubin(nvJitLinkHandle handle, void* cubin) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult _nvJitLinkGetLinkedPtxSize(nvJitLinkHandle handle, size_t* size) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult _nvJitLinkGetLinkedPtx(nvJitLinkHandle handle, char* ptx) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult _nvJitLinkGetErrorLogSize(nvJitLinkHandle handle, size_t* size) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult _nvJitLinkGetErrorLog(nvJitLinkHandle handle, char* log) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult _nvJitLinkGetInfoLogSize(nvJitLinkHandle handle, size_t* size) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult _nvJitLinkGetInfoLog(nvJitLinkHandle handle, char* log) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult _nvJitLinkVersion(unsigned int* major, unsigned int* minor) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult _nvJitLinkGetLinkedLTOIRSize(nvJitLinkHandle handle, size_t* size) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvJitLinkResult _nvJitLinkGetLinkedLTOIR(nvJitLinkHandle handle, void* ltoir) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil
