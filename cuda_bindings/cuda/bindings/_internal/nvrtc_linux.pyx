# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated with version 12.9.0, generator version 0.3.1.dev1603+gcfe73f973. Do not modify it directly.

from libc.stdint cimport intptr_t, uintptr_t

import threading
from .utils import FunctionNotFoundError, NotSupportedError

from cuda.pathfinder import load_nvidia_dynamic_lib

###############################################################################
# Extern
###############################################################################

# You must 'from .utils import NotSupportedError' before using this template

cdef extern from "<dlfcn.h>" nogil:
    void* dlopen(const char*, int)
    char* dlerror()
    void* dlsym(void*, const char*)
    int dlclose(void*)

    enum:
        RTLD_LAZY
        RTLD_NOW
        RTLD_GLOBAL
        RTLD_LOCAL

    const void* RTLD_DEFAULT 'RTLD_DEFAULT'

cdef int get_cuda_version():
    cdef void* handle = NULL
    cdef int err, driver_ver = 0

    # Load driver to check version
    handle = dlopen('libcuda.so.1', RTLD_NOW | RTLD_GLOBAL)
    if handle == NULL:
        err_msg = dlerror()
        raise NotSupportedError(f'CUDA driver is not found ({err_msg.decode()})')
    cuDriverGetVersion = dlsym(handle, "cuDriverGetVersion")
    if cuDriverGetVersion == NULL:
        raise RuntimeError('Did not find cuDriverGetVersion symbol in libcuda.so.1')
    err = (<int (*)(int*) noexcept nogil>cuDriverGetVersion)(&driver_ver)
    if err != 0:
        raise RuntimeError(f'cuDriverGetVersion returned error code {err}')

    return driver_ver

###############################################################################
# Wrapper init
###############################################################################

cdef object __symbol_lock = threading.Lock()
cdef bint __py_nvrtc_init = False

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

cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("nvrtc")._handle_uint
    return <void*>handle

cdef int _init_nvrtc() except -1 nogil:
    global __py_nvrtc_init

    cdef void* handle = NULL

    with gil, __symbol_lock:
        # Recheck the flag after obtaining the locks
        if __py_nvrtc_init:
            return 0

        # Load function
        global __nvrtcGetErrorString
        __nvrtcGetErrorString = dlsym(RTLD_DEFAULT, 'nvrtcGetErrorString')
        if __nvrtcGetErrorString == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetErrorString = dlsym(handle, 'nvrtcGetErrorString')

        global __nvrtcVersion
        __nvrtcVersion = dlsym(RTLD_DEFAULT, 'nvrtcVersion')
        if __nvrtcVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcVersion = dlsym(handle, 'nvrtcVersion')

        global __nvrtcGetNumSupportedArchs
        __nvrtcGetNumSupportedArchs = dlsym(RTLD_DEFAULT, 'nvrtcGetNumSupportedArchs')
        if __nvrtcGetNumSupportedArchs == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetNumSupportedArchs = dlsym(handle, 'nvrtcGetNumSupportedArchs')

        global __nvrtcGetSupportedArchs
        __nvrtcGetSupportedArchs = dlsym(RTLD_DEFAULT, 'nvrtcGetSupportedArchs')
        if __nvrtcGetSupportedArchs == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetSupportedArchs = dlsym(handle, 'nvrtcGetSupportedArchs')

        global __nvrtcCreateProgram
        __nvrtcCreateProgram = dlsym(RTLD_DEFAULT, 'nvrtcCreateProgram')
        if __nvrtcCreateProgram == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcCreateProgram = dlsym(handle, 'nvrtcCreateProgram')

        global __nvrtcDestroyProgram
        __nvrtcDestroyProgram = dlsym(RTLD_DEFAULT, 'nvrtcDestroyProgram')
        if __nvrtcDestroyProgram == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcDestroyProgram = dlsym(handle, 'nvrtcDestroyProgram')

        global __nvrtcCompileProgram
        __nvrtcCompileProgram = dlsym(RTLD_DEFAULT, 'nvrtcCompileProgram')
        if __nvrtcCompileProgram == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcCompileProgram = dlsym(handle, 'nvrtcCompileProgram')

        global __nvrtcGetPTXSize
        __nvrtcGetPTXSize = dlsym(RTLD_DEFAULT, 'nvrtcGetPTXSize')
        if __nvrtcGetPTXSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetPTXSize = dlsym(handle, 'nvrtcGetPTXSize')

        global __nvrtcGetPTX
        __nvrtcGetPTX = dlsym(RTLD_DEFAULT, 'nvrtcGetPTX')
        if __nvrtcGetPTX == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetPTX = dlsym(handle, 'nvrtcGetPTX')

        global __nvrtcGetCUBINSize
        __nvrtcGetCUBINSize = dlsym(RTLD_DEFAULT, 'nvrtcGetCUBINSize')
        if __nvrtcGetCUBINSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetCUBINSize = dlsym(handle, 'nvrtcGetCUBINSize')

        global __nvrtcGetCUBIN
        __nvrtcGetCUBIN = dlsym(RTLD_DEFAULT, 'nvrtcGetCUBIN')
        if __nvrtcGetCUBIN == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetCUBIN = dlsym(handle, 'nvrtcGetCUBIN')

        global __nvrtcGetLTOIRSize
        __nvrtcGetLTOIRSize = dlsym(RTLD_DEFAULT, 'nvrtcGetLTOIRSize')
        if __nvrtcGetLTOIRSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetLTOIRSize = dlsym(handle, 'nvrtcGetLTOIRSize')

        global __nvrtcGetLTOIR
        __nvrtcGetLTOIR = dlsym(RTLD_DEFAULT, 'nvrtcGetLTOIR')
        if __nvrtcGetLTOIR == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetLTOIR = dlsym(handle, 'nvrtcGetLTOIR')

        global __nvrtcGetOptiXIRSize
        __nvrtcGetOptiXIRSize = dlsym(RTLD_DEFAULT, 'nvrtcGetOptiXIRSize')
        if __nvrtcGetOptiXIRSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetOptiXIRSize = dlsym(handle, 'nvrtcGetOptiXIRSize')

        global __nvrtcGetOptiXIR
        __nvrtcGetOptiXIR = dlsym(RTLD_DEFAULT, 'nvrtcGetOptiXIR')
        if __nvrtcGetOptiXIR == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetOptiXIR = dlsym(handle, 'nvrtcGetOptiXIR')

        global __nvrtcGetProgramLogSize
        __nvrtcGetProgramLogSize = dlsym(RTLD_DEFAULT, 'nvrtcGetProgramLogSize')
        if __nvrtcGetProgramLogSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetProgramLogSize = dlsym(handle, 'nvrtcGetProgramLogSize')

        global __nvrtcGetProgramLog
        __nvrtcGetProgramLog = dlsym(RTLD_DEFAULT, 'nvrtcGetProgramLog')
        if __nvrtcGetProgramLog == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetProgramLog = dlsym(handle, 'nvrtcGetProgramLog')

        global __nvrtcAddNameExpression
        __nvrtcAddNameExpression = dlsym(RTLD_DEFAULT, 'nvrtcAddNameExpression')
        if __nvrtcAddNameExpression == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcAddNameExpression = dlsym(handle, 'nvrtcAddNameExpression')

        global __nvrtcGetLoweredName
        __nvrtcGetLoweredName = dlsym(RTLD_DEFAULT, 'nvrtcGetLoweredName')
        if __nvrtcGetLoweredName == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetLoweredName = dlsym(handle, 'nvrtcGetLoweredName')

        global __nvrtcGetPCHHeapSize
        __nvrtcGetPCHHeapSize = dlsym(RTLD_DEFAULT, 'nvrtcGetPCHHeapSize')
        if __nvrtcGetPCHHeapSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetPCHHeapSize = dlsym(handle, 'nvrtcGetPCHHeapSize')

        global __nvrtcSetPCHHeapSize
        __nvrtcSetPCHHeapSize = dlsym(RTLD_DEFAULT, 'nvrtcSetPCHHeapSize')
        if __nvrtcSetPCHHeapSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcSetPCHHeapSize = dlsym(handle, 'nvrtcSetPCHHeapSize')

        global __nvrtcGetPCHCreateStatus
        __nvrtcGetPCHCreateStatus = dlsym(RTLD_DEFAULT, 'nvrtcGetPCHCreateStatus')
        if __nvrtcGetPCHCreateStatus == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetPCHCreateStatus = dlsym(handle, 'nvrtcGetPCHCreateStatus')

        global __nvrtcGetPCHHeapSizeRequired
        __nvrtcGetPCHHeapSizeRequired = dlsym(RTLD_DEFAULT, 'nvrtcGetPCHHeapSizeRequired')
        if __nvrtcGetPCHHeapSizeRequired == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcGetPCHHeapSizeRequired = dlsym(handle, 'nvrtcGetPCHHeapSizeRequired')

        global __nvrtcSetFlowCallback
        __nvrtcSetFlowCallback = dlsym(RTLD_DEFAULT, 'nvrtcSetFlowCallback')
        if __nvrtcSetFlowCallback == NULL:
            if handle == NULL:
                handle = load_library()
            __nvrtcSetFlowCallback = dlsym(handle, 'nvrtcSetFlowCallback')

        __py_nvrtc_init = True
        return 0

cdef inline int _check_or_init_nvrtc() except -1 nogil:
    if __py_nvrtc_init:
        return 0

    return _init_nvrtc()

cdef dict func_ptrs = None

cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_nvrtc()
    cdef dict data = {}

    global __nvrtcGetErrorString
    data["__nvrtcGetErrorString"] = <intptr_t>__nvrtcGetErrorString

    global __nvrtcVersion
    data["__nvrtcVersion"] = <intptr_t>__nvrtcVersion

    global __nvrtcGetNumSupportedArchs
    data["__nvrtcGetNumSupportedArchs"] = <intptr_t>__nvrtcGetNumSupportedArchs

    global __nvrtcGetSupportedArchs
    data["__nvrtcGetSupportedArchs"] = <intptr_t>__nvrtcGetSupportedArchs

    global __nvrtcCreateProgram
    data["__nvrtcCreateProgram"] = <intptr_t>__nvrtcCreateProgram

    global __nvrtcDestroyProgram
    data["__nvrtcDestroyProgram"] = <intptr_t>__nvrtcDestroyProgram

    global __nvrtcCompileProgram
    data["__nvrtcCompileProgram"] = <intptr_t>__nvrtcCompileProgram

    global __nvrtcGetPTXSize
    data["__nvrtcGetPTXSize"] = <intptr_t>__nvrtcGetPTXSize

    global __nvrtcGetPTX
    data["__nvrtcGetPTX"] = <intptr_t>__nvrtcGetPTX

    global __nvrtcGetCUBINSize
    data["__nvrtcGetCUBINSize"] = <intptr_t>__nvrtcGetCUBINSize

    global __nvrtcGetCUBIN
    data["__nvrtcGetCUBIN"] = <intptr_t>__nvrtcGetCUBIN

    global __nvrtcGetLTOIRSize
    data["__nvrtcGetLTOIRSize"] = <intptr_t>__nvrtcGetLTOIRSize

    global __nvrtcGetLTOIR
    data["__nvrtcGetLTOIR"] = <intptr_t>__nvrtcGetLTOIR

    global __nvrtcGetOptiXIRSize
    data["__nvrtcGetOptiXIRSize"] = <intptr_t>__nvrtcGetOptiXIRSize

    global __nvrtcGetOptiXIR
    data["__nvrtcGetOptiXIR"] = <intptr_t>__nvrtcGetOptiXIR

    global __nvrtcGetProgramLogSize
    data["__nvrtcGetProgramLogSize"] = <intptr_t>__nvrtcGetProgramLogSize

    global __nvrtcGetProgramLog
    data["__nvrtcGetProgramLog"] = <intptr_t>__nvrtcGetProgramLog

    global __nvrtcAddNameExpression
    data["__nvrtcAddNameExpression"] = <intptr_t>__nvrtcAddNameExpression

    global __nvrtcGetLoweredName
    data["__nvrtcGetLoweredName"] = <intptr_t>__nvrtcGetLoweredName

    global __nvrtcGetPCHHeapSize
    data["__nvrtcGetPCHHeapSize"] = <intptr_t>__nvrtcGetPCHHeapSize

    global __nvrtcSetPCHHeapSize
    data["__nvrtcSetPCHHeapSize"] = <intptr_t>__nvrtcSetPCHHeapSize

    global __nvrtcGetPCHCreateStatus
    data["__nvrtcGetPCHCreateStatus"] = <intptr_t>__nvrtcGetPCHCreateStatus

    global __nvrtcGetPCHHeapSizeRequired
    data["__nvrtcGetPCHHeapSizeRequired"] = <intptr_t>__nvrtcGetPCHHeapSizeRequired

    global __nvrtcSetFlowCallback
    data["__nvrtcSetFlowCallback"] = <intptr_t>__nvrtcSetFlowCallback

    func_ptrs = data
    return data

cpdef _inspect_function_pointer(str name):
    global func_ptrs
    if func_ptrs is None:
        func_ptrs = _inspect_function_pointers()
    return func_ptrs[name]

###############################################################################
# Wrapper functions
###############################################################################

cdef const char* _nvrtcGetErrorString(nvrtcResult result) except ?NULL nogil:
    global __nvrtcGetErrorString
    _check_or_init_nvrtc()
    if __nvrtcGetErrorString == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetErrorString is not found")
    return (<const char* (*)(nvrtcResult) noexcept nogil>__nvrtcGetErrorString)(result)

cdef nvrtcResult _nvrtcVersion(int* major, int* minor) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcVersion
    _check_or_init_nvrtc()
    if __nvrtcVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcVersion is not found")
    return (<nvrtcResult (*)(int*, int*) noexcept nogil>__nvrtcVersion)(major, minor)

cdef nvrtcResult _nvrtcGetNumSupportedArchs(int* numArchs) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetNumSupportedArchs
    _check_or_init_nvrtc()
    if __nvrtcGetNumSupportedArchs == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetNumSupportedArchs is not found")
    return (<nvrtcResult (*)(int*) noexcept nogil>__nvrtcGetNumSupportedArchs)(numArchs)

cdef nvrtcResult _nvrtcGetSupportedArchs(int* supportedArchs) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetSupportedArchs
    _check_or_init_nvrtc()
    if __nvrtcGetSupportedArchs == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetSupportedArchs is not found")
    return (<nvrtcResult (*)(int*) noexcept nogil>__nvrtcGetSupportedArchs)(supportedArchs)

cdef nvrtcResult _nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcCreateProgram
    _check_or_init_nvrtc()
    if __nvrtcCreateProgram == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcCreateProgram is not found")
    return (<nvrtcResult (*)(nvrtcProgram*, const char*, const char*, int, const char**, const char**) noexcept nogil>__nvrtcCreateProgram)(prog, src, name, numHeaders, headers, includeNames)

cdef nvrtcResult _nvrtcDestroyProgram(nvrtcProgram* prog) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcDestroyProgram
    _check_or_init_nvrtc()
    if __nvrtcDestroyProgram == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcDestroyProgram is not found")
    return (<nvrtcResult (*)(nvrtcProgram*) noexcept nogil>__nvrtcDestroyProgram)(prog)

cdef nvrtcResult _nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char** options) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcCompileProgram
    _check_or_init_nvrtc()
    if __nvrtcCompileProgram == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcCompileProgram is not found")
    return (<nvrtcResult (*)(nvrtcProgram, int, const char**) noexcept nogil>__nvrtcCompileProgram)(prog, numOptions, options)

cdef nvrtcResult _nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetPTXSize
    _check_or_init_nvrtc()
    if __nvrtcGetPTXSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetPTXSize is not found")
    return (<nvrtcResult (*)(nvrtcProgram, size_t*) noexcept nogil>__nvrtcGetPTXSize)(prog, ptxSizeRet)

cdef nvrtcResult _nvrtcGetPTX(nvrtcProgram prog, char* ptx) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetPTX
    _check_or_init_nvrtc()
    if __nvrtcGetPTX == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetPTX is not found")
    return (<nvrtcResult (*)(nvrtcProgram, char*) noexcept nogil>__nvrtcGetPTX)(prog, ptx)

cdef nvrtcResult _nvrtcGetCUBINSize(nvrtcProgram prog, size_t* cubinSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetCUBINSize
    _check_or_init_nvrtc()
    if __nvrtcGetCUBINSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetCUBINSize is not found")
    return (<nvrtcResult (*)(nvrtcProgram, size_t*) noexcept nogil>__nvrtcGetCUBINSize)(prog, cubinSizeRet)

cdef nvrtcResult _nvrtcGetCUBIN(nvrtcProgram prog, char* cubin) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetCUBIN
    _check_or_init_nvrtc()
    if __nvrtcGetCUBIN == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetCUBIN is not found")
    return (<nvrtcResult (*)(nvrtcProgram, char*) noexcept nogil>__nvrtcGetCUBIN)(prog, cubin)

cdef nvrtcResult _nvrtcGetLTOIRSize(nvrtcProgram prog, size_t* LTOIRSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetLTOIRSize
    _check_or_init_nvrtc()
    if __nvrtcGetLTOIRSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetLTOIRSize is not found")
    return (<nvrtcResult (*)(nvrtcProgram, size_t*) noexcept nogil>__nvrtcGetLTOIRSize)(prog, LTOIRSizeRet)

cdef nvrtcResult _nvrtcGetLTOIR(nvrtcProgram prog, char* LTOIR) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetLTOIR
    _check_or_init_nvrtc()
    if __nvrtcGetLTOIR == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetLTOIR is not found")
    return (<nvrtcResult (*)(nvrtcProgram, char*) noexcept nogil>__nvrtcGetLTOIR)(prog, LTOIR)

cdef nvrtcResult _nvrtcGetOptiXIRSize(nvrtcProgram prog, size_t* optixirSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetOptiXIRSize
    _check_or_init_nvrtc()
    if __nvrtcGetOptiXIRSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetOptiXIRSize is not found")
    return (<nvrtcResult (*)(nvrtcProgram, size_t*) noexcept nogil>__nvrtcGetOptiXIRSize)(prog, optixirSizeRet)

cdef nvrtcResult _nvrtcGetOptiXIR(nvrtcProgram prog, char* optixir) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetOptiXIR
    _check_or_init_nvrtc()
    if __nvrtcGetOptiXIR == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetOptiXIR is not found")
    return (<nvrtcResult (*)(nvrtcProgram, char*) noexcept nogil>__nvrtcGetOptiXIR)(prog, optixir)

cdef nvrtcResult _nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetProgramLogSize
    _check_or_init_nvrtc()
    if __nvrtcGetProgramLogSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetProgramLogSize is not found")
    return (<nvrtcResult (*)(nvrtcProgram, size_t*) noexcept nogil>__nvrtcGetProgramLogSize)(prog, logSizeRet)

cdef nvrtcResult _nvrtcGetProgramLog(nvrtcProgram prog, char* log) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetProgramLog
    _check_or_init_nvrtc()
    if __nvrtcGetProgramLog == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetProgramLog is not found")
    return (<nvrtcResult (*)(nvrtcProgram, char*) noexcept nogil>__nvrtcGetProgramLog)(prog, log)

cdef nvrtcResult _nvrtcAddNameExpression(nvrtcProgram prog, const char* name_expression) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcAddNameExpression
    _check_or_init_nvrtc()
    if __nvrtcAddNameExpression == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcAddNameExpression is not found")
    return (<nvrtcResult (*)(nvrtcProgram, const char*) noexcept nogil>__nvrtcAddNameExpression)(prog, name_expression)

cdef nvrtcResult _nvrtcGetLoweredName(nvrtcProgram prog, const char* name_expression, const char** lowered_name) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetLoweredName
    _check_or_init_nvrtc()
    if __nvrtcGetLoweredName == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetLoweredName is not found")
    return (<nvrtcResult (*)(nvrtcProgram, const char*, const char**) noexcept nogil>__nvrtcGetLoweredName)(prog, name_expression, lowered_name)

cdef nvrtcResult _nvrtcGetPCHHeapSize(size_t* ret) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetPCHHeapSize
    _check_or_init_nvrtc()
    if __nvrtcGetPCHHeapSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetPCHHeapSize is not found")
    return (<nvrtcResult (*)(size_t*) noexcept nogil>__nvrtcGetPCHHeapSize)(ret)

cdef nvrtcResult _nvrtcSetPCHHeapSize(size_t size) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcSetPCHHeapSize
    _check_or_init_nvrtc()
    if __nvrtcSetPCHHeapSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcSetPCHHeapSize is not found")
    return (<nvrtcResult (*)(size_t) noexcept nogil>__nvrtcSetPCHHeapSize)(size)

cdef nvrtcResult _nvrtcGetPCHCreateStatus(nvrtcProgram prog) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetPCHCreateStatus
    _check_or_init_nvrtc()
    if __nvrtcGetPCHCreateStatus == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetPCHCreateStatus is not found")
    return (<nvrtcResult (*)(nvrtcProgram) noexcept nogil>__nvrtcGetPCHCreateStatus)(prog)

cdef nvrtcResult _nvrtcGetPCHHeapSizeRequired(nvrtcProgram prog, size_t* size) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetPCHHeapSizeRequired
    _check_or_init_nvrtc()
    if __nvrtcGetPCHHeapSizeRequired == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcGetPCHHeapSizeRequired is not found")
    return (<nvrtcResult (*)(nvrtcProgram, size_t*) noexcept nogil>__nvrtcGetPCHHeapSizeRequired)(prog, size)

cdef nvrtcResult _nvrtcSetFlowCallback(nvrtcProgram prog, void* callback, void* payload) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcSetFlowCallback
    _check_or_init_nvrtc()
    if __nvrtcSetFlowCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function nvrtcSetFlowCallback is not found")
    return (<nvrtcResult (*)(nvrtcProgram, void*, void*) noexcept nogil>__nvrtcSetFlowCallback)(prog, callback, payload)
