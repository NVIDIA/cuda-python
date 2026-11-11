# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.4.1 to 13.5.0. Do not modify it directly.
# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=8934b7f337c877db73540d82524de10847b1beb8d17b4bd86a2cd07329fc1882


# <<<< PREAMBLE CONTENT >>>>

from libc.stdint cimport intptr_t

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

cdef enum: _cyb_nvfatbin_n_functions = 12
cdef void** _cyb_nvfatbin_slots[_cyb_nvfatbin_n_functions]
cdef const char* _cyb_nvfatbin_names[_cyb_nvfatbin_n_functions]
_cyb_nvfatbin_slots[0] = &__nvFatbinGetErrorString; _cyb_nvfatbin_names[0] = b'nvFatbinGetErrorString'
_cyb_nvfatbin_slots[1] = &__nvFatbinCreate; _cyb_nvfatbin_names[1] = b'nvFatbinCreate'
_cyb_nvfatbin_slots[2] = &__nvFatbinDestroy; _cyb_nvfatbin_names[2] = b'nvFatbinDestroy'
_cyb_nvfatbin_slots[3] = &__nvFatbinAddPTX; _cyb_nvfatbin_names[3] = b'nvFatbinAddPTX'
_cyb_nvfatbin_slots[4] = &__nvFatbinAddCubin; _cyb_nvfatbin_names[4] = b'nvFatbinAddCubin'
_cyb_nvfatbin_slots[5] = &__nvFatbinAddLTOIR; _cyb_nvfatbin_names[5] = b'nvFatbinAddLTOIR'
_cyb_nvfatbin_slots[6] = &__nvFatbinSize; _cyb_nvfatbin_names[6] = b'nvFatbinSize'
_cyb_nvfatbin_slots[7] = &__nvFatbinGet; _cyb_nvfatbin_names[7] = b'nvFatbinGet'
_cyb_nvfatbin_slots[8] = &__nvFatbinVersion; _cyb_nvfatbin_names[8] = b'nvFatbinVersion'
_cyb_nvfatbin_slots[9] = &__nvFatbinAddIndex; _cyb_nvfatbin_names[9] = b'nvFatbinAddIndex'
_cyb_nvfatbin_slots[10] = &__nvFatbinAddReloc; _cyb_nvfatbin_names[10] = b'nvFatbinAddReloc'
_cyb_nvfatbin_slots[11] = &__nvFatbinAddTileIR; _cyb_nvfatbin_names[11] = b'nvFatbinAddTileIR'

cdef int _init_nvfatbin() except -1 nogil:
    global _cyb___py_nvfatbin_init
    cdef void* handle = NULL
    cdef Py_ssize_t i
    with gil, _cyb_symbol_lock:
        if _cyb___py_nvfatbin_init: return 0

        for i in range(_cyb_nvfatbin_n_functions):
            _cyb_nvfatbin_slots[i][0] = _cyb_dlsym(_cyb_RTLD_DEFAULT, _cyb_nvfatbin_names[i])
            if _cyb_nvfatbin_slots[i][0] == NULL:
                if handle == NULL:
                    handle = load_library()
                _cyb_nvfatbin_slots[i][0] = _cyb_dlsym(handle, _cyb_nvfatbin_names[i])

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
    cdef Py_ssize_t i
    for i in range(_cyb_nvfatbin_n_functions):
        data['__' + _cyb_nvfatbin_names[i].decode()] = <intptr_t>_cyb_nvfatbin_slots[i][0]
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
