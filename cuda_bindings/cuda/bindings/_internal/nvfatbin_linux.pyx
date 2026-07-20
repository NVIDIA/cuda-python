# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.4.1 to 13.3.0. Do not modify it directly.
# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=f86a7f7527aad594d7b2e67742165454729ecca3810c00e6786f772099b17850


# <<<< PREAMBLE CONTENT >>>>

cdef extern from * nogil:
    """
    #if defined(_MSC_VER) && !defined(__clang__)
        #include <intrin.h>
        static __forceinline int atomic_int_load(int *p) {
            int v = *(int volatile *)p; _ReadBarrier(); return v;
        }
        static __forceinline void atomic_int_store(int *p, int v) {
            _WriteBarrier(); *(int volatile *)p = v;
        }
    #elif defined(__cplusplus)
        /* GCC/Clang __atomic builtins work in any C++ standard without headers */
        static inline int atomic_int_load(int *p) {
            return __atomic_load_n(p, __ATOMIC_ACQUIRE);
        }
        static inline void atomic_int_store(int *p, int v) {
            __atomic_store_n(p, v, __ATOMIC_RELEASE);
        }
    #else
        #include <stdatomic.h>
        static inline int atomic_int_load(int *p) {
            return (int)atomic_load_explicit((atomic_int *)p, memory_order_acquire);
        }
        static inline void atomic_int_store(int *p, int v) {
            atomic_store_explicit((atomic_int *)p, v, memory_order_release);
        }
    #endif

    """
    cdef int _cyb_atomic_int_load "atomic_int_load"(int *p) nogil
    cdef void _cyb_atomic_int_store "atomic_int_store"(int *p, int v) nogil

cdef extern from "<dlfcn.h>":
    void* _cyb_dlsym "dlsym"(void*, const char*) nogil
    const void * _cyb_RTLD_DEFAULT "RTLD_DEFAULT"

from libc.stdint cimport intptr_t as _cyb_intptr_t

import threading as _cyb_threading

cdef int _cyb___py_nvfatbin_init = 0
cdef dict _cyb_func_ptrs = None
cdef object _cyb_symbol_lock = _cyb_threading.Lock()

# <<<< END OF PREAMBLE CONTENT >>>>

from libc.stdint cimport uintptr_t

from .utils import FunctionNotFoundError, NotSupportedError
from cuda.pathfinder import load_nvidia_dynamic_lib


###############################################################################
# Wrapper init
###############################################################################

cdef void* __nvFatbinGetErrorString = NULL
cdef void* __nvFatbinCreate = NULL
cdef void* __nvFatbinDestroy = NULL
cdef void* __nvFatbinAddPTX = NULL
cdef void* __nvFatbinAddCubin = NULL
cdef void* __nvFatbinAddLTOIR = NULL
cdef void* __nvFatbinSize = NULL
cdef void* __nvFatbinGet = NULL
cdef void* __nvFatbinVersion = NULL
cdef void* __nvFatbinAddIndex = NULL
cdef void* __nvFatbinAddReloc = NULL
cdef void* __nvFatbinAddTileIR = NULL

cdef int _init_nvfatbin() except -1 nogil:
    global _cyb___py_nvfatbin_init
    cdef void* handle = NULL
    with gil, _cyb_symbol_lock:
        if _cyb___py_nvfatbin_init: return 0

        global __nvFatbinGetErrorString
        __nvFatbinGetErrorString = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvFatbinGetErrorString')
        if __nvFatbinGetErrorString == NULL:
            if handle == NULL:
                handle = load_library()
            __nvFatbinGetErrorString = _cyb_dlsym(handle, 'nvFatbinGetErrorString')

        global __nvFatbinCreate
        __nvFatbinCreate = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvFatbinCreate')
        if __nvFatbinCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __nvFatbinCreate = _cyb_dlsym(handle, 'nvFatbinCreate')

        global __nvFatbinDestroy
        __nvFatbinDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvFatbinDestroy')
        if __nvFatbinDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __nvFatbinDestroy = _cyb_dlsym(handle, 'nvFatbinDestroy')

        global __nvFatbinAddPTX
        __nvFatbinAddPTX = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvFatbinAddPTX')
        if __nvFatbinAddPTX == NULL:
            if handle == NULL:
                handle = load_library()
            __nvFatbinAddPTX = _cyb_dlsym(handle, 'nvFatbinAddPTX')

        global __nvFatbinAddCubin
        __nvFatbinAddCubin = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvFatbinAddCubin')
        if __nvFatbinAddCubin == NULL:
            if handle == NULL:
                handle = load_library()
            __nvFatbinAddCubin = _cyb_dlsym(handle, 'nvFatbinAddCubin')

        global __nvFatbinAddLTOIR
        __nvFatbinAddLTOIR = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvFatbinAddLTOIR')
        if __nvFatbinAddLTOIR == NULL:
            if handle == NULL:
                handle = load_library()
            __nvFatbinAddLTOIR = _cyb_dlsym(handle, 'nvFatbinAddLTOIR')

        global __nvFatbinSize
        __nvFatbinSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvFatbinSize')
        if __nvFatbinSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvFatbinSize = _cyb_dlsym(handle, 'nvFatbinSize')

        global __nvFatbinGet
        __nvFatbinGet = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvFatbinGet')
        if __nvFatbinGet == NULL:
            if handle == NULL:
                handle = load_library()
            __nvFatbinGet = _cyb_dlsym(handle, 'nvFatbinGet')

        global __nvFatbinVersion
        __nvFatbinVersion = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvFatbinVersion')
        if __nvFatbinVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __nvFatbinVersion = _cyb_dlsym(handle, 'nvFatbinVersion')

        global __nvFatbinAddIndex
        __nvFatbinAddIndex = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvFatbinAddIndex')
        if __nvFatbinAddIndex == NULL:
            if handle == NULL:
                handle = load_library()
            __nvFatbinAddIndex = _cyb_dlsym(handle, 'nvFatbinAddIndex')

        global __nvFatbinAddReloc
        __nvFatbinAddReloc = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvFatbinAddReloc')
        if __nvFatbinAddReloc == NULL:
            if handle == NULL:
                handle = load_library()
            __nvFatbinAddReloc = _cyb_dlsym(handle, 'nvFatbinAddReloc')

        global __nvFatbinAddTileIR
        __nvFatbinAddTileIR = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvFatbinAddTileIR')
        if __nvFatbinAddTileIR == NULL:
            if handle == NULL:
                handle = load_library()
            __nvFatbinAddTileIR = _cyb_dlsym(handle, 'nvFatbinAddTileIR')

        _cyb_atomic_int_store(<int *>&_cyb___py_nvfatbin_init, 1)
        return 0

cdef inline int _check_or_init_nvfatbin() except -1 nogil:
    if _cyb_atomic_int_load(<int *>&_cyb___py_nvfatbin_init):
        return 0

    return _init_nvfatbin()


cpdef dict _inspect_function_pointers():
    global _cyb_func_ptrs
    if _cyb_func_ptrs is not None:
        return _cyb_func_ptrs

    _check_or_init_nvfatbin()
    cdef dict data = {}
    global __nvFatbinGetErrorString
    data["__nvFatbinGetErrorString"] = <_cyb_intptr_t>__nvFatbinGetErrorString

    global __nvFatbinCreate
    data["__nvFatbinCreate"] = <_cyb_intptr_t>__nvFatbinCreate

    global __nvFatbinDestroy
    data["__nvFatbinDestroy"] = <_cyb_intptr_t>__nvFatbinDestroy

    global __nvFatbinAddPTX
    data["__nvFatbinAddPTX"] = <_cyb_intptr_t>__nvFatbinAddPTX

    global __nvFatbinAddCubin
    data["__nvFatbinAddCubin"] = <_cyb_intptr_t>__nvFatbinAddCubin

    global __nvFatbinAddLTOIR
    data["__nvFatbinAddLTOIR"] = <_cyb_intptr_t>__nvFatbinAddLTOIR

    global __nvFatbinSize
    data["__nvFatbinSize"] = <_cyb_intptr_t>__nvFatbinSize

    global __nvFatbinGet
    data["__nvFatbinGet"] = <_cyb_intptr_t>__nvFatbinGet

    global __nvFatbinVersion
    data["__nvFatbinVersion"] = <_cyb_intptr_t>__nvFatbinVersion

    global __nvFatbinAddIndex
    data["__nvFatbinAddIndex"] = <_cyb_intptr_t>__nvFatbinAddIndex

    global __nvFatbinAddReloc
    data["__nvFatbinAddReloc"] = <_cyb_intptr_t>__nvFatbinAddReloc

    global __nvFatbinAddTileIR
    data["__nvFatbinAddTileIR"] = <_cyb_intptr_t>__nvFatbinAddTileIR
    _cyb_func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global _cyb_func_ptrs
    if _cyb_func_ptrs is None:
        _cyb_func_ptrs = _inspect_function_pointers()
    return _cyb_func_ptrs[name]




cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("nvfatbin")._handle_uint
    return <void*>handle


###############################################################################
# Wrapper functions
###############################################################################

cdef const char* _nvFatbinGetErrorString(nvFatbinResult result) except?NULL nogil:
    global __nvFatbinGetErrorString
    _check_or_init_nvfatbin()
    if __nvFatbinGetErrorString == NULL:
        with gil:
            raise FunctionNotFoundError("function nvFatbinGetErrorString is not found")
    return (<const char* (*)(nvFatbinResult) noexcept nogil>__nvFatbinGetErrorString)(
        result)


cdef nvFatbinResult _nvFatbinCreate(nvFatbinHandle* handle_indirect, const char** options, size_t optionsCount) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvFatbinCreate
    _check_or_init_nvfatbin()
    if __nvFatbinCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function nvFatbinCreate is not found")
    return (<nvFatbinResult (*)(nvFatbinHandle*, const char**, size_t) noexcept nogil>__nvFatbinCreate)(
        handle_indirect, options, optionsCount)


cdef nvFatbinResult _nvFatbinDestroy(nvFatbinHandle* handle_indirect) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvFatbinDestroy
    _check_or_init_nvfatbin()
    if __nvFatbinDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function nvFatbinDestroy is not found")
    return (<nvFatbinResult (*)(nvFatbinHandle*) noexcept nogil>__nvFatbinDestroy)(
        handle_indirect)


cdef nvFatbinResult _nvFatbinAddPTX(nvFatbinHandle handle, const char* code, size_t size, const char* arch, const char* identifier, const char* optionsCmdLine) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvFatbinAddPTX
    _check_or_init_nvfatbin()
    if __nvFatbinAddPTX == NULL:
        with gil:
            raise FunctionNotFoundError("function nvFatbinAddPTX is not found")
    return (<nvFatbinResult (*)(nvFatbinHandle, const char*, size_t, const char*, const char*, const char*) noexcept nogil>__nvFatbinAddPTX)(
        handle, code, size, arch, identifier, optionsCmdLine)


cdef nvFatbinResult _nvFatbinAddCubin(nvFatbinHandle handle, const void* code, size_t size, const char* arch, const char* identifier) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvFatbinAddCubin
    _check_or_init_nvfatbin()
    if __nvFatbinAddCubin == NULL:
        with gil:
            raise FunctionNotFoundError("function nvFatbinAddCubin is not found")
    return (<nvFatbinResult (*)(nvFatbinHandle, const void*, size_t, const char*, const char*) noexcept nogil>__nvFatbinAddCubin)(
        handle, code, size, arch, identifier)


cdef nvFatbinResult _nvFatbinAddLTOIR(nvFatbinHandle handle, const void* code, size_t size, const char* arch, const char* identifier, const char* optionsCmdLine) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvFatbinAddLTOIR
    _check_or_init_nvfatbin()
    if __nvFatbinAddLTOIR == NULL:
        with gil:
            raise FunctionNotFoundError("function nvFatbinAddLTOIR is not found")
    return (<nvFatbinResult (*)(nvFatbinHandle, const void*, size_t, const char*, const char*, const char*) noexcept nogil>__nvFatbinAddLTOIR)(
        handle, code, size, arch, identifier, optionsCmdLine)


cdef nvFatbinResult _nvFatbinSize(nvFatbinHandle handle, size_t* size) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvFatbinSize
    _check_or_init_nvfatbin()
    if __nvFatbinSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvFatbinSize is not found")
    return (<nvFatbinResult (*)(nvFatbinHandle, size_t*) noexcept nogil>__nvFatbinSize)(
        handle, size)


cdef nvFatbinResult _nvFatbinGet(nvFatbinHandle handle, void* buffer) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvFatbinGet
    _check_or_init_nvfatbin()
    if __nvFatbinGet == NULL:
        with gil:
            raise FunctionNotFoundError("function nvFatbinGet is not found")
    return (<nvFatbinResult (*)(nvFatbinHandle, void*) noexcept nogil>__nvFatbinGet)(
        handle, buffer)


cdef nvFatbinResult _nvFatbinVersion(unsigned int* major, unsigned int* minor) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvFatbinVersion
    _check_or_init_nvfatbin()
    if __nvFatbinVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvFatbinVersion is not found")
    return (<nvFatbinResult (*)(unsigned int*, unsigned int*) noexcept nogil>__nvFatbinVersion)(
        major, minor)


cdef nvFatbinResult _nvFatbinAddIndex(nvFatbinHandle handle, const void* code, size_t size, const char* identifier) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvFatbinAddIndex
    _check_or_init_nvfatbin()
    if __nvFatbinAddIndex == NULL:
        with gil:
            raise FunctionNotFoundError("function nvFatbinAddIndex is not found")
    return (<nvFatbinResult (*)(nvFatbinHandle, const void*, size_t, const char*) noexcept nogil>__nvFatbinAddIndex)(
        handle, code, size, identifier)


cdef nvFatbinResult _nvFatbinAddReloc(nvFatbinHandle handle, const void* code, size_t size) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvFatbinAddReloc
    _check_or_init_nvfatbin()
    if __nvFatbinAddReloc == NULL:
        with gil:
            raise FunctionNotFoundError("function nvFatbinAddReloc is not found")
    return (<nvFatbinResult (*)(nvFatbinHandle, const void*, size_t) noexcept nogil>__nvFatbinAddReloc)(
        handle, code, size)


cdef nvFatbinResult _nvFatbinAddTileIR(nvFatbinHandle handle, const void* code, size_t size, const char* identifier, const char* optionsCmdLine) except?_NVFATBINRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvFatbinAddTileIR
    _check_or_init_nvfatbin()
    if __nvFatbinAddTileIR == NULL:
        with gil:
            raise FunctionNotFoundError("function nvFatbinAddTileIR is not found")
    return (<nvFatbinResult (*)(nvFatbinHandle, const void*, size_t, const char*, const char*) noexcept nogil>__nvFatbinAddTileIR)(
        handle, code, size, identifier, optionsCmdLine)
