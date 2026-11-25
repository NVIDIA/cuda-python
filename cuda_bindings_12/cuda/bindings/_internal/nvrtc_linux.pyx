# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 12.9.0. Do not modify it directly.
# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=a86fd01987bd6934281500ad99dc4da0ae555b3e1deec8218333f2b98d4ffec0


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

cdef int _cyb___py_nvrtc_init = 0
cdef dict _cyb_func_ptrs = None
cdef object _cyb_symbol_lock = _cyb_threading.Lock()

# <<<< END OF PREAMBLE CONTENT >>>>

from libc.stdint cimport uintptr_t

from .utils import FunctionNotFoundError, NotSupportedError
from cuda.pathfinder import load_nvidia_dynamic_lib


###############################################################################
# Wrapper init
###############################################################################

cdef void* __nvrtcGetErrorString = NULL
cdef void* __nvrtcVersion = NULL
cdef void* __nvrtcGetNumSupportedArchs = NULL
cdef void* __nvrtcGetSupportedArchs = NULL
cdef void* __nvrtcCreateProgram = NULL
cdef void* __nvrtcDestroyProgram = NULL
cdef void* __nvrtcCompileProgram = NULL
cdef void* __nvrtcGetPTXSize = NULL
cdef void* __nvrtcGetPTX = NULL
cdef void* __nvrtcGetCUBINSize = NULL
cdef void* __nvrtcGetCUBIN = NULL
cdef void* __nvrtcGetLTOIRSize = NULL
cdef void* __nvrtcGetLTOIR = NULL
cdef void* __nvrtcGetOptiXIRSize = NULL
cdef void* __nvrtcGetOptiXIR = NULL
cdef void* __nvrtcGetProgramLogSize = NULL
cdef void* __nvrtcGetProgramLog = NULL
cdef void* __nvrtcAddNameExpression = NULL
cdef void* __nvrtcGetLoweredName = NULL
cdef void* __nvrtcGetPCHHeapSize = NULL
cdef void* __nvrtcSetPCHHeapSize = NULL
cdef void* __nvrtcGetPCHCreateStatus = NULL
cdef void* __nvrtcGetPCHHeapSizeRequired = NULL
cdef void* __nvrtcSetFlowCallback = NULL

cdef enum: _cyb_nvrtc_n_functions = 24
cdef void** _cyb_nvrtc_slots[_cyb_nvrtc_n_functions]
cdef const char* _cyb_nvrtc_names[_cyb_nvrtc_n_functions]
_cyb_nvrtc_slots[0] = &__nvrtcGetErrorString; _cyb_nvrtc_names[0] = b'nvrtcGetErrorString'
_cyb_nvrtc_slots[1] = &__nvrtcVersion; _cyb_nvrtc_names[1] = b'nvrtcVersion'
_cyb_nvrtc_slots[2] = &__nvrtcGetNumSupportedArchs; _cyb_nvrtc_names[2] = b'nvrtcGetNumSupportedArchs'
_cyb_nvrtc_slots[3] = &__nvrtcGetSupportedArchs; _cyb_nvrtc_names[3] = b'nvrtcGetSupportedArchs'
_cyb_nvrtc_slots[4] = &__nvrtcCreateProgram; _cyb_nvrtc_names[4] = b'nvrtcCreateProgram'
_cyb_nvrtc_slots[5] = &__nvrtcDestroyProgram; _cyb_nvrtc_names[5] = b'nvrtcDestroyProgram'
_cyb_nvrtc_slots[6] = &__nvrtcCompileProgram; _cyb_nvrtc_names[6] = b'nvrtcCompileProgram'
_cyb_nvrtc_slots[7] = &__nvrtcGetPTXSize; _cyb_nvrtc_names[7] = b'nvrtcGetPTXSize'
_cyb_nvrtc_slots[8] = &__nvrtcGetPTX; _cyb_nvrtc_names[8] = b'nvrtcGetPTX'
_cyb_nvrtc_slots[9] = &__nvrtcGetCUBINSize; _cyb_nvrtc_names[9] = b'nvrtcGetCUBINSize'
_cyb_nvrtc_slots[10] = &__nvrtcGetCUBIN; _cyb_nvrtc_names[10] = b'nvrtcGetCUBIN'
_cyb_nvrtc_slots[11] = &__nvrtcGetLTOIRSize; _cyb_nvrtc_names[11] = b'nvrtcGetLTOIRSize'
_cyb_nvrtc_slots[12] = &__nvrtcGetLTOIR; _cyb_nvrtc_names[12] = b'nvrtcGetLTOIR'
_cyb_nvrtc_slots[13] = &__nvrtcGetOptiXIRSize; _cyb_nvrtc_names[13] = b'nvrtcGetOptiXIRSize'
_cyb_nvrtc_slots[14] = &__nvrtcGetOptiXIR; _cyb_nvrtc_names[14] = b'nvrtcGetOptiXIR'
_cyb_nvrtc_slots[15] = &__nvrtcGetProgramLogSize; _cyb_nvrtc_names[15] = b'nvrtcGetProgramLogSize'
_cyb_nvrtc_slots[16] = &__nvrtcGetProgramLog; _cyb_nvrtc_names[16] = b'nvrtcGetProgramLog'
_cyb_nvrtc_slots[17] = &__nvrtcAddNameExpression; _cyb_nvrtc_names[17] = b'nvrtcAddNameExpression'
_cyb_nvrtc_slots[18] = &__nvrtcGetLoweredName; _cyb_nvrtc_names[18] = b'nvrtcGetLoweredName'
_cyb_nvrtc_slots[19] = &__nvrtcGetPCHHeapSize; _cyb_nvrtc_names[19] = b'nvrtcGetPCHHeapSize'
_cyb_nvrtc_slots[20] = &__nvrtcSetPCHHeapSize; _cyb_nvrtc_names[20] = b'nvrtcSetPCHHeapSize'
_cyb_nvrtc_slots[21] = &__nvrtcGetPCHCreateStatus; _cyb_nvrtc_names[21] = b'nvrtcGetPCHCreateStatus'
_cyb_nvrtc_slots[22] = &__nvrtcGetPCHHeapSizeRequired; _cyb_nvrtc_names[22] = b'nvrtcGetPCHHeapSizeRequired'
_cyb_nvrtc_slots[23] = &__nvrtcSetFlowCallback; _cyb_nvrtc_names[23] = b'nvrtcSetFlowCallback'

cdef int _init_nvrtc() except -1 nogil:
    global _cyb___py_nvrtc_init
    cdef void* handle = NULL
    cdef Py_ssize_t i
    with gil, _cyb_symbol_lock:
        if _cyb___py_nvrtc_init: return 0

        for i in range(_cyb_nvrtc_n_functions):
            _cyb_nvrtc_slots[i][0] = _cyb_dlsym(_cyb_RTLD_DEFAULT, _cyb_nvrtc_names[i])
            if _cyb_nvrtc_slots[i][0] == NULL:
                if handle == NULL:
                    handle = load_library()
                _cyb_nvrtc_slots[i][0] = _cyb_dlsym(handle, _cyb_nvrtc_names[i])

        _cyb_atomic_int_store(<int *>&_cyb___py_nvrtc_init, 1)
        return 0

cdef inline int _check_or_init_nvrtc() except -1 nogil:
    if _cyb_atomic_int_load(<int *>&_cyb___py_nvrtc_init):
        return 0

    return _init_nvrtc()


cpdef dict _inspect_function_pointers():
    global _cyb_func_ptrs
    if _cyb_func_ptrs is not None:
        return _cyb_func_ptrs

    _check_or_init_nvrtc()
    cdef dict data = {}
    cdef Py_ssize_t i
    for i in range(_cyb_nvrtc_n_functions):
        data['__' + _cyb_nvrtc_names[i].decode()] = <intptr_t>_cyb_nvrtc_slots[i][0]
    _cyb_func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global _cyb_func_ptrs
    if _cyb_func_ptrs is None:
        _cyb_func_ptrs = _inspect_function_pointers()
    return _cyb_func_ptrs[name]




cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("nvrtc")._handle_uint
    return <void*>handle


###############################################################################
# Wrapper functions
###############################################################################

cdef const char* _nvrtcGetErrorString(nvrtcResult result) except?NULL nogil:
    global __nvrtcGetErrorString
    _check_or_init_nvrtc()
    if __nvrtcGetErrorString == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetErrorString is not found")
    return (<const char* (*)(nvrtcResult) noexcept nogil>__nvrtcGetErrorString)(
        result)


cdef nvrtcResult _nvrtcVersion(int* major, int* minor) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcVersion
    _check_or_init_nvrtc()
    if __nvrtcVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcVersion is not found")
    return (<nvrtcResult (*)(int*, int*) noexcept nogil>__nvrtcVersion)(
        major, minor)


cdef nvrtcResult _nvrtcGetNumSupportedArchs(int* numArchs) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetNumSupportedArchs
    _check_or_init_nvrtc()
    if __nvrtcGetNumSupportedArchs == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetNumSupportedArchs is not found")
    return (<nvrtcResult (*)(int*) noexcept nogil>__nvrtcGetNumSupportedArchs)(
        numArchs)


cdef nvrtcResult _nvrtcGetSupportedArchs(int* supportedArchs) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetSupportedArchs
    _check_or_init_nvrtc()
    if __nvrtcGetSupportedArchs == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetSupportedArchs is not found")
    return (<nvrtcResult (*)(int*) noexcept nogil>__nvrtcGetSupportedArchs)(
        supportedArchs)


cdef nvrtcResult _nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcCreateProgram
    _check_or_init_nvrtc()
    if __nvrtcCreateProgram == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcCreateProgram is not found")
    return (<nvrtcResult (*)(nvrtcProgram*, const char*, const char*, int, const char**, const char**) noexcept nogil>__nvrtcCreateProgram)(
        prog, src, name, numHeaders, headers, includeNames)


cdef nvrtcResult _nvrtcDestroyProgram(nvrtcProgram* prog) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcDestroyProgram
    _check_or_init_nvrtc()
    if __nvrtcDestroyProgram == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcDestroyProgram is not found")
    return (<nvrtcResult (*)(nvrtcProgram*) noexcept nogil>__nvrtcDestroyProgram)(
        prog)


cdef nvrtcResult _nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char** options) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcCompileProgram
    _check_or_init_nvrtc()
    if __nvrtcCompileProgram == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcCompileProgram is not found")
    return (<nvrtcResult (*)(nvrtcProgram, int, const char**) noexcept nogil>__nvrtcCompileProgram)(
        prog, numOptions, options)


cdef nvrtcResult _nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetPTXSize
    _check_or_init_nvrtc()
    if __nvrtcGetPTXSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetPTXSize is not found")
    return (<nvrtcResult (*)(nvrtcProgram, size_t*) noexcept nogil>__nvrtcGetPTXSize)(
        prog, ptxSizeRet)


cdef nvrtcResult _nvrtcGetPTX(nvrtcProgram prog, char* ptx) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetPTX
    _check_or_init_nvrtc()
    if __nvrtcGetPTX == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetPTX is not found")
    return (<nvrtcResult (*)(nvrtcProgram, char*) noexcept nogil>__nvrtcGetPTX)(
        prog, ptx)


cdef nvrtcResult _nvrtcGetCUBINSize(nvrtcProgram prog, size_t* cubinSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetCUBINSize
    _check_or_init_nvrtc()
    if __nvrtcGetCUBINSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetCUBINSize is not found")
    return (<nvrtcResult (*)(nvrtcProgram, size_t*) noexcept nogil>__nvrtcGetCUBINSize)(
        prog, cubinSizeRet)


cdef nvrtcResult _nvrtcGetCUBIN(nvrtcProgram prog, char* cubin) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetCUBIN
    _check_or_init_nvrtc()
    if __nvrtcGetCUBIN == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetCUBIN is not found")
    return (<nvrtcResult (*)(nvrtcProgram, char*) noexcept nogil>__nvrtcGetCUBIN)(
        prog, cubin)


cdef nvrtcResult _nvrtcGetLTOIRSize(nvrtcProgram prog, size_t* LTOIRSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetLTOIRSize
    _check_or_init_nvrtc()
    if __nvrtcGetLTOIRSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetLTOIRSize is not found")
    return (<nvrtcResult (*)(nvrtcProgram, size_t*) noexcept nogil>__nvrtcGetLTOIRSize)(
        prog, LTOIRSizeRet)


cdef nvrtcResult _nvrtcGetLTOIR(nvrtcProgram prog, char* LTOIR) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetLTOIR
    _check_or_init_nvrtc()
    if __nvrtcGetLTOIR == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetLTOIR is not found")
    return (<nvrtcResult (*)(nvrtcProgram, char*) noexcept nogil>__nvrtcGetLTOIR)(
        prog, LTOIR)


cdef nvrtcResult _nvrtcGetOptiXIRSize(nvrtcProgram prog, size_t* optixirSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetOptiXIRSize
    _check_or_init_nvrtc()
    if __nvrtcGetOptiXIRSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetOptiXIRSize is not found")
    return (<nvrtcResult (*)(nvrtcProgram, size_t*) noexcept nogil>__nvrtcGetOptiXIRSize)(
        prog, optixirSizeRet)


cdef nvrtcResult _nvrtcGetOptiXIR(nvrtcProgram prog, char* optixir) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetOptiXIR
    _check_or_init_nvrtc()
    if __nvrtcGetOptiXIR == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetOptiXIR is not found")
    return (<nvrtcResult (*)(nvrtcProgram, char*) noexcept nogil>__nvrtcGetOptiXIR)(
        prog, optixir)


cdef nvrtcResult _nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetProgramLogSize
    _check_or_init_nvrtc()
    if __nvrtcGetProgramLogSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetProgramLogSize is not found")
    return (<nvrtcResult (*)(nvrtcProgram, size_t*) noexcept nogil>__nvrtcGetProgramLogSize)(
        prog, logSizeRet)


cdef nvrtcResult _nvrtcGetProgramLog(nvrtcProgram prog, char* log) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetProgramLog
    _check_or_init_nvrtc()
    if __nvrtcGetProgramLog == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetProgramLog is not found")
    return (<nvrtcResult (*)(nvrtcProgram, char*) noexcept nogil>__nvrtcGetProgramLog)(
        prog, log)


cdef nvrtcResult _nvrtcAddNameExpression(nvrtcProgram prog, const char* name_expression) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcAddNameExpression
    _check_or_init_nvrtc()
    if __nvrtcAddNameExpression == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcAddNameExpression is not found")
    return (<nvrtcResult (*)(nvrtcProgram, const char*) noexcept nogil>__nvrtcAddNameExpression)(
        prog, name_expression)


cdef nvrtcResult _nvrtcGetLoweredName(nvrtcProgram prog, const char* name_expression, const char** lowered_name) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetLoweredName
    _check_or_init_nvrtc()
    if __nvrtcGetLoweredName == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetLoweredName is not found")
    return (<nvrtcResult (*)(nvrtcProgram, const char*, const char**) noexcept nogil>__nvrtcGetLoweredName)(
        prog, name_expression, lowered_name)


cdef nvrtcResult _nvrtcGetPCHHeapSize(size_t* ret) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetPCHHeapSize
    _check_or_init_nvrtc()
    if __nvrtcGetPCHHeapSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetPCHHeapSize is not found")
    return (<nvrtcResult (*)(size_t*) noexcept nogil>__nvrtcGetPCHHeapSize)(
        ret)


cdef nvrtcResult _nvrtcSetPCHHeapSize(size_t size) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcSetPCHHeapSize
    _check_or_init_nvrtc()
    if __nvrtcSetPCHHeapSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcSetPCHHeapSize is not found")
    return (<nvrtcResult (*)(size_t) noexcept nogil>__nvrtcSetPCHHeapSize)(
        size)


cdef nvrtcResult _nvrtcGetPCHCreateStatus(nvrtcProgram prog) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetPCHCreateStatus
    _check_or_init_nvrtc()
    if __nvrtcGetPCHCreateStatus == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetPCHCreateStatus is not found")
    return (<nvrtcResult (*)(nvrtcProgram) noexcept nogil>__nvrtcGetPCHCreateStatus)(
        prog)


cdef nvrtcResult _nvrtcGetPCHHeapSizeRequired(nvrtcProgram prog, size_t* size) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetPCHHeapSizeRequired
    _check_or_init_nvrtc()
    if __nvrtcGetPCHHeapSizeRequired == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetPCHHeapSizeRequired is not found")
    return (<nvrtcResult (*)(nvrtcProgram, size_t*) noexcept nogil>__nvrtcGetPCHHeapSizeRequired)(
        prog, size)


cdef nvrtcResult _nvrtcSetFlowCallback(nvrtcProgram prog, void * callback, void* payload) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcSetFlowCallback
    _check_or_init_nvrtc()
    if __nvrtcSetFlowCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcSetFlowCallback is not found")
    return (<nvrtcResult (*)(nvrtcProgram, void *, void*) noexcept nogil>__nvrtcSetFlowCallback)(
        prog, callback, payload)
