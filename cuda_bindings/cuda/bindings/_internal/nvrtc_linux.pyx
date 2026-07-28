# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.9.0 to 13.3.0. Do not modify it directly.
# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=ff5d5beb8dc8a8fb1dac0241d4b72d29bf8c55cfd2cba627567cfab2d4d10767


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
cdef void* __nvrtcGetTileIRSize = NULL
cdef void* __nvrtcGetTileIR = NULL
cdef void* __nvrtcInstallBundledHeaders = NULL
cdef void* __nvrtcGetBundledHeadersInfo = NULL
cdef void* __nvrtcRemoveBundledHeaders = NULL

cdef int _init_nvrtc() except -1 nogil:
    global _cyb___py_nvrtc_init
    cdef void* handle = NULL
    with gil, _cyb_symbol_lock:
        if _cyb___py_nvrtc_init: return 0

        global __nvrtcGetErrorString
        __nvrtcGetErrorString = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcGetErrorString')
        if __nvrtcGetErrorString == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetErrorString = _cyb_dlsym(handle, 'nvrtcGetErrorString')

        global __nvrtcVersion
        __nvrtcVersion = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcVersion')
        if __nvrtcVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcVersion = _cyb_dlsym(handle, 'nvrtcVersion')

        global __nvrtcGetNumSupportedArchs
        __nvrtcGetNumSupportedArchs = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcGetNumSupportedArchs')
        if __nvrtcGetNumSupportedArchs == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetNumSupportedArchs = _cyb_dlsym(handle, 'nvrtcGetNumSupportedArchs')

        global __nvrtcGetSupportedArchs
        __nvrtcGetSupportedArchs = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcGetSupportedArchs')
        if __nvrtcGetSupportedArchs == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetSupportedArchs = _cyb_dlsym(handle, 'nvrtcGetSupportedArchs')

        global __nvrtcCreateProgram
        __nvrtcCreateProgram = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcCreateProgram')
        if __nvrtcCreateProgram == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcCreateProgram = _cyb_dlsym(handle, 'nvrtcCreateProgram')

        global __nvrtcDestroyProgram
        __nvrtcDestroyProgram = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcDestroyProgram')
        if __nvrtcDestroyProgram == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcDestroyProgram = _cyb_dlsym(handle, 'nvrtcDestroyProgram')

        global __nvrtcCompileProgram
        __nvrtcCompileProgram = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcCompileProgram')
        if __nvrtcCompileProgram == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcCompileProgram = _cyb_dlsym(handle, 'nvrtcCompileProgram')

        global __nvrtcGetPTXSize
        __nvrtcGetPTXSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcGetPTXSize')
        if __nvrtcGetPTXSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetPTXSize = _cyb_dlsym(handle, 'nvrtcGetPTXSize')

        global __nvrtcGetPTX
        __nvrtcGetPTX = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcGetPTX')
        if __nvrtcGetPTX == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetPTX = _cyb_dlsym(handle, 'nvrtcGetPTX')

        global __nvrtcGetCUBINSize
        __nvrtcGetCUBINSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcGetCUBINSize')
        if __nvrtcGetCUBINSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetCUBINSize = _cyb_dlsym(handle, 'nvrtcGetCUBINSize')

        global __nvrtcGetCUBIN
        __nvrtcGetCUBIN = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcGetCUBIN')
        if __nvrtcGetCUBIN == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetCUBIN = _cyb_dlsym(handle, 'nvrtcGetCUBIN')

        global __nvrtcGetLTOIRSize
        __nvrtcGetLTOIRSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcGetLTOIRSize')
        if __nvrtcGetLTOIRSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetLTOIRSize = _cyb_dlsym(handle, 'nvrtcGetLTOIRSize')

        global __nvrtcGetLTOIR
        __nvrtcGetLTOIR = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcGetLTOIR')
        if __nvrtcGetLTOIR == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetLTOIR = _cyb_dlsym(handle, 'nvrtcGetLTOIR')

        global __nvrtcGetOptiXIRSize
        __nvrtcGetOptiXIRSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcGetOptiXIRSize')
        if __nvrtcGetOptiXIRSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetOptiXIRSize = _cyb_dlsym(handle, 'nvrtcGetOptiXIRSize')

        global __nvrtcGetOptiXIR
        __nvrtcGetOptiXIR = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcGetOptiXIR')
        if __nvrtcGetOptiXIR == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetOptiXIR = _cyb_dlsym(handle, 'nvrtcGetOptiXIR')

        global __nvrtcGetProgramLogSize
        __nvrtcGetProgramLogSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcGetProgramLogSize')
        if __nvrtcGetProgramLogSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetProgramLogSize = _cyb_dlsym(handle, 'nvrtcGetProgramLogSize')

        global __nvrtcGetProgramLog
        __nvrtcGetProgramLog = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcGetProgramLog')
        if __nvrtcGetProgramLog == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetProgramLog = _cyb_dlsym(handle, 'nvrtcGetProgramLog')

        global __nvrtcAddNameExpression
        __nvrtcAddNameExpression = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcAddNameExpression')
        if __nvrtcAddNameExpression == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcAddNameExpression = _cyb_dlsym(handle, 'nvrtcAddNameExpression')

        global __nvrtcGetLoweredName
        __nvrtcGetLoweredName = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcGetLoweredName')
        if __nvrtcGetLoweredName == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetLoweredName = _cyb_dlsym(handle, 'nvrtcGetLoweredName')

        global __nvrtcGetPCHHeapSize
        __nvrtcGetPCHHeapSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcGetPCHHeapSize')
        if __nvrtcGetPCHHeapSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetPCHHeapSize = _cyb_dlsym(handle, 'nvrtcGetPCHHeapSize')

        global __nvrtcSetPCHHeapSize
        __nvrtcSetPCHHeapSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcSetPCHHeapSize')
        if __nvrtcSetPCHHeapSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcSetPCHHeapSize = _cyb_dlsym(handle, 'nvrtcSetPCHHeapSize')

        global __nvrtcGetPCHCreateStatus
        __nvrtcGetPCHCreateStatus = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcGetPCHCreateStatus')
        if __nvrtcGetPCHCreateStatus == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetPCHCreateStatus = _cyb_dlsym(handle, 'nvrtcGetPCHCreateStatus')

        global __nvrtcGetPCHHeapSizeRequired
        __nvrtcGetPCHHeapSizeRequired = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcGetPCHHeapSizeRequired')
        if __nvrtcGetPCHHeapSizeRequired == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetPCHHeapSizeRequired = _cyb_dlsym(handle, 'nvrtcGetPCHHeapSizeRequired')

        global __nvrtcSetFlowCallback
        __nvrtcSetFlowCallback = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcSetFlowCallback')
        if __nvrtcSetFlowCallback == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcSetFlowCallback = _cyb_dlsym(handle, 'nvrtcSetFlowCallback')

        global __nvrtcGetTileIRSize
        __nvrtcGetTileIRSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcGetTileIRSize')
        if __nvrtcGetTileIRSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetTileIRSize = _cyb_dlsym(handle, 'nvrtcGetTileIRSize')

        global __nvrtcGetTileIR
        __nvrtcGetTileIR = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcGetTileIR')
        if __nvrtcGetTileIR == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetTileIR = _cyb_dlsym(handle, 'nvrtcGetTileIR')

        global __nvrtcInstallBundledHeaders
        __nvrtcInstallBundledHeaders = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcInstallBundledHeaders')
        if __nvrtcInstallBundledHeaders == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcInstallBundledHeaders = _cyb_dlsym(handle, 'nvrtcInstallBundledHeaders')

        global __nvrtcGetBundledHeadersInfo
        __nvrtcGetBundledHeadersInfo = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcGetBundledHeadersInfo')
        if __nvrtcGetBundledHeadersInfo == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetBundledHeadersInfo = _cyb_dlsym(handle, 'nvrtcGetBundledHeadersInfo')

        global __nvrtcRemoveBundledHeaders
        __nvrtcRemoveBundledHeaders = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvrtcRemoveBundledHeaders')
        if __nvrtcRemoveBundledHeaders == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcRemoveBundledHeaders = _cyb_dlsym(handle, 'nvrtcRemoveBundledHeaders')

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
    global __nvrtcGetErrorString
    data["__nvrtcGetErrorString"] = <_cyb_intptr_t>__nvrtcGetErrorString

    global __nvrtcVersion
    data["__nvrtcVersion"] = <_cyb_intptr_t>__nvrtcVersion

    global __nvrtcGetNumSupportedArchs
    data["__nvrtcGetNumSupportedArchs"] = <_cyb_intptr_t>__nvrtcGetNumSupportedArchs

    global __nvrtcGetSupportedArchs
    data["__nvrtcGetSupportedArchs"] = <_cyb_intptr_t>__nvrtcGetSupportedArchs

    global __nvrtcCreateProgram
    data["__nvrtcCreateProgram"] = <_cyb_intptr_t>__nvrtcCreateProgram

    global __nvrtcDestroyProgram
    data["__nvrtcDestroyProgram"] = <_cyb_intptr_t>__nvrtcDestroyProgram

    global __nvrtcCompileProgram
    data["__nvrtcCompileProgram"] = <_cyb_intptr_t>__nvrtcCompileProgram

    global __nvrtcGetPTXSize
    data["__nvrtcGetPTXSize"] = <_cyb_intptr_t>__nvrtcGetPTXSize

    global __nvrtcGetPTX
    data["__nvrtcGetPTX"] = <_cyb_intptr_t>__nvrtcGetPTX

    global __nvrtcGetCUBINSize
    data["__nvrtcGetCUBINSize"] = <_cyb_intptr_t>__nvrtcGetCUBINSize

    global __nvrtcGetCUBIN
    data["__nvrtcGetCUBIN"] = <_cyb_intptr_t>__nvrtcGetCUBIN

    global __nvrtcGetLTOIRSize
    data["__nvrtcGetLTOIRSize"] = <_cyb_intptr_t>__nvrtcGetLTOIRSize

    global __nvrtcGetLTOIR
    data["__nvrtcGetLTOIR"] = <_cyb_intptr_t>__nvrtcGetLTOIR

    global __nvrtcGetOptiXIRSize
    data["__nvrtcGetOptiXIRSize"] = <_cyb_intptr_t>__nvrtcGetOptiXIRSize

    global __nvrtcGetOptiXIR
    data["__nvrtcGetOptiXIR"] = <_cyb_intptr_t>__nvrtcGetOptiXIR

    global __nvrtcGetProgramLogSize
    data["__nvrtcGetProgramLogSize"] = <_cyb_intptr_t>__nvrtcGetProgramLogSize

    global __nvrtcGetProgramLog
    data["__nvrtcGetProgramLog"] = <_cyb_intptr_t>__nvrtcGetProgramLog

    global __nvrtcAddNameExpression
    data["__nvrtcAddNameExpression"] = <_cyb_intptr_t>__nvrtcAddNameExpression

    global __nvrtcGetLoweredName
    data["__nvrtcGetLoweredName"] = <_cyb_intptr_t>__nvrtcGetLoweredName

    global __nvrtcGetPCHHeapSize
    data["__nvrtcGetPCHHeapSize"] = <_cyb_intptr_t>__nvrtcGetPCHHeapSize

    global __nvrtcSetPCHHeapSize
    data["__nvrtcSetPCHHeapSize"] = <_cyb_intptr_t>__nvrtcSetPCHHeapSize

    global __nvrtcGetPCHCreateStatus
    data["__nvrtcGetPCHCreateStatus"] = <_cyb_intptr_t>__nvrtcGetPCHCreateStatus

    global __nvrtcGetPCHHeapSizeRequired
    data["__nvrtcGetPCHHeapSizeRequired"] = <_cyb_intptr_t>__nvrtcGetPCHHeapSizeRequired

    global __nvrtcSetFlowCallback
    data["__nvrtcSetFlowCallback"] = <_cyb_intptr_t>__nvrtcSetFlowCallback

    global __nvrtcGetTileIRSize
    data["__nvrtcGetTileIRSize"] = <_cyb_intptr_t>__nvrtcGetTileIRSize

    global __nvrtcGetTileIR
    data["__nvrtcGetTileIR"] = <_cyb_intptr_t>__nvrtcGetTileIR

    global __nvrtcInstallBundledHeaders
    data["__nvrtcInstallBundledHeaders"] = <_cyb_intptr_t>__nvrtcInstallBundledHeaders

    global __nvrtcGetBundledHeadersInfo
    data["__nvrtcGetBundledHeadersInfo"] = <_cyb_intptr_t>__nvrtcGetBundledHeadersInfo

    global __nvrtcRemoveBundledHeaders
    data["__nvrtcRemoveBundledHeaders"] = <_cyb_intptr_t>__nvrtcRemoveBundledHeaders
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


cdef nvrtcResult _nvrtcGetTileIRSize(nvrtcProgram prog, size_t* TileIRSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetTileIRSize
    _check_or_init_nvrtc()
    if __nvrtcGetTileIRSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetTileIRSize is not found")
    return (<nvrtcResult (*)(nvrtcProgram, size_t*) noexcept nogil>__nvrtcGetTileIRSize)(
        prog, TileIRSizeRet)


cdef nvrtcResult _nvrtcGetTileIR(nvrtcProgram prog, char* TileIR) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetTileIR
    _check_or_init_nvrtc()
    if __nvrtcGetTileIR == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetTileIR is not found")
    return (<nvrtcResult (*)(nvrtcProgram, char*) noexcept nogil>__nvrtcGetTileIR)(
        prog, TileIR)


cdef nvrtcResult _nvrtcInstallBundledHeaders(const char* installPath, unsigned int flags, const char** errorLog) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcInstallBundledHeaders
    _check_or_init_nvrtc()
    if __nvrtcInstallBundledHeaders == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcInstallBundledHeaders is not found")
    return (<nvrtcResult (*)(const char*, unsigned int, const char**) noexcept nogil>__nvrtcInstallBundledHeaders)(
        installPath, flags, errorLog)


cdef nvrtcResult _nvrtcGetBundledHeadersInfo(nvrtcBundledHeadersInfo* info, const char** errorLog) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetBundledHeadersInfo
    _check_or_init_nvrtc()
    if __nvrtcGetBundledHeadersInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetBundledHeadersInfo is not found")
    return (<nvrtcResult (*)(nvrtcBundledHeadersInfo*, const char**) noexcept nogil>__nvrtcGetBundledHeadersInfo)(
        info, errorLog)


cdef nvrtcResult _nvrtcRemoveBundledHeaders(const char* installPath, const char** errorLog) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcRemoveBundledHeaders
    _check_or_init_nvrtc()
    if __nvrtcRemoveBundledHeaders == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcRemoveBundledHeaders is not found")
    return (<nvrtcResult (*)(const char*, const char**) noexcept nogil>__nvrtcRemoveBundledHeaders)(
        installPath, errorLog)
