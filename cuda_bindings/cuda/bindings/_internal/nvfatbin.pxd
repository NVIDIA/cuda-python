# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.4.1 to 13.4.0. Do not modify it directly.

# !!! WARNING: THIS FILE CONTAINS PRERELEASE APIs !!!
# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=4176ab61d2c088c30ef453fdc5aee9f5f66e0cf76732826ab900fd211c143e14
from ..cynvfatbin cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef const char* _nvFatbinGetErrorString(nvFatbinResult result) except?NULL nogil
cdef nvFatbinResult _nvFatbinCreate(nvFatbinHandle* handle_indirect, const char** options, size_t optionsCount) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvFatbinResult _nvFatbinDestroy(nvFatbinHandle* handle_indirect) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvFatbinResult _nvFatbinAddPTX(nvFatbinHandle handle, const char* code, size_t size, const char* arch, const char* identifier, const char* optionsCmdLine) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvFatbinResult _nvFatbinAddCubin(nvFatbinHandle handle, const void* code, size_t size, const char* arch, const char* identifier) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvFatbinResult _nvFatbinAddLTOIR(nvFatbinHandle handle, const void* code, size_t size, const char* arch, const char* identifier, const char* optionsCmdLine) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvFatbinResult _nvFatbinSize(nvFatbinHandle handle, size_t* size) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvFatbinResult _nvFatbinGet(nvFatbinHandle handle, void* buffer) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvFatbinResult _nvFatbinVersion(unsigned int* major, unsigned int* minor) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvFatbinResult _nvFatbinAddIndex(nvFatbinHandle handle, const void* code, size_t size, const char* identifier) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvFatbinResult _nvFatbinAddReloc(nvFatbinHandle handle, const void* code, size_t size) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
cdef nvFatbinResult _nvFatbinAddTileIR(nvFatbinHandle handle, const void* code, size_t size, const char* identifier, const char* optionsCmdLine) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil
