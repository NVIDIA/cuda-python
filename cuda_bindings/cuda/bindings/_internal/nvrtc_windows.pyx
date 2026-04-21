# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated with version 12.9.0, generator version 0.3.1.dev1603+gcfe73f973. Do not modify it directly.

from libc.stdint cimport intptr_t

import threading
from .utils import FunctionNotFoundError, NotSupportedError

from cuda.pathfinder import load_nvidia_dynamic_lib

from libc.stddef cimport wchar_t
from libc.stdint cimport uintptr_t
from cpython cimport PyUnicode_AsWideCharString, PyMem_Free

# You must 'from .utils import NotSupportedError' before using this template

cdef extern from "windows.h" nogil:
    ctypedef void* HMODULE
    ctypedef void* HANDLE
    ctypedef void* FARPROC
    ctypedef unsigned long DWORD
    ctypedef const wchar_t *LPCWSTR
    ctypedef const char *LPCSTR

    cdef DWORD LOAD_LIBRARY_SEARCH_SYSTEM32 = 0x00000800
    cdef DWORD LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000
    cdef DWORD LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100

    HMODULE _LoadLibraryExW "LoadLibraryExW"(
        LPCWSTR lpLibFileName,
        HANDLE hFile,
        DWORD dwFlags
    )

    FARPROC _GetProcAddress "GetProcAddress"(HMODULE hModule, LPCSTR lpProcName)

cdef inline uintptr_t LoadLibraryExW(str path, HANDLE hFile, DWORD dwFlags):
    cdef uintptr_t result
    cdef wchar_t* wpath = PyUnicode_AsWideCharString(path, NULL)
    with nogil:
        result = <uintptr_t>_LoadLibraryExW(
            wpath,
            hFile,
            dwFlags
        )
    PyMem_Free(wpath)
    return result

cdef inline void *GetProcAddress(uintptr_t hModule, const char* lpProcName) nogil:
    return _GetProcAddress(<HMODULE>hModule, lpProcName)

cdef int get_cuda_version():
    cdef int err, driver_ver = 0

    # Load driver to check version
    handle = LoadLibraryExW("nvcuda.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32)
    if handle == 0:
        raise NotSupportedError('CUDA driver is not found')
    cuDriverGetVersion = GetProcAddress(handle, 'cuDriverGetVersion')
    if cuDriverGetVersion == NULL:
        raise RuntimeError('Did not find cuDriverGetVersion symbol in nvcuda.dll')
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

cdef int _init_nvrtc() except -1 nogil:
    global __py_nvrtc_init

    with gil, __symbol_lock:
        # Recheck the flag after obtaining the locks
        if __py_nvrtc_init:
            return 0

        # Load library
        handle = load_nvidia_dynamic_lib("nvrtc")._handle_uint

        # Load function
        global __nvrtcGetErrorString
        __nvrtcGetErrorString = GetProcAddress(handle, 'nvrtcGetErrorString')

        global __nvrtcVersion
        __nvrtcVersion = GetProcAddress(handle, 'nvrtcVersion')

        global __nvrtcGetNumSupportedArchs
        __nvrtcGetNumSupportedArchs = GetProcAddress(handle, 'nvrtcGetNumSupportedArchs')

        global __nvrtcGetSupportedArchs
        __nvrtcGetSupportedArchs = GetProcAddress(handle, 'nvrtcGetSupportedArchs')

        global __nvrtcCreateProgram
        __nvrtcCreateProgram = GetProcAddress(handle, 'nvrtcCreateProgram')

        global __nvrtcDestroyProgram
        __nvrtcDestroyProgram = GetProcAddress(handle, 'nvrtcDestroyProgram')

        global __nvrtcCompileProgram
        __nvrtcCompileProgram = GetProcAddress(handle, 'nvrtcCompileProgram')

        global __nvrtcGetPTXSize
        __nvrtcGetPTXSize = GetProcAddress(handle, 'nvrtcGetPTXSize')

        global __nvrtcGetPTX
        __nvrtcGetPTX = GetProcAddress(handle, 'nvrtcGetPTX')

        global __nvrtcGetCUBINSize
        __nvrtcGetCUBINSize = GetProcAddress(handle, 'nvrtcGetCUBINSize')

        global __nvrtcGetCUBIN
        __nvrtcGetCUBIN = GetProcAddress(handle, 'nvrtcGetCUBIN')

        global __nvrtcGetLTOIRSize
        __nvrtcGetLTOIRSize = GetProcAddress(handle, 'nvrtcGetLTOIRSize')

        global __nvrtcGetLTOIR
        __nvrtcGetLTOIR = GetProcAddress(handle, 'nvrtcGetLTOIR')

        global __nvrtcGetOptiXIRSize
        __nvrtcGetOptiXIRSize = GetProcAddress(handle, 'nvrtcGetOptiXIRSize')

        global __nvrtcGetOptiXIR
        __nvrtcGetOptiXIR = GetProcAddress(handle, 'nvrtcGetOptiXIR')

        global __nvrtcGetProgramLogSize
        __nvrtcGetProgramLogSize = GetProcAddress(handle, 'nvrtcGetProgramLogSize')

        global __nvrtcGetProgramLog
        __nvrtcGetProgramLog = GetProcAddress(handle, 'nvrtcGetProgramLog')

        global __nvrtcAddNameExpression
        __nvrtcAddNameExpression = GetProcAddress(handle, 'nvrtcAddNameExpression')

        global __nvrtcGetLoweredName
        __nvrtcGetLoweredName = GetProcAddress(handle, 'nvrtcGetLoweredName')

        global __nvrtcGetPCHHeapSize
        __nvrtcGetPCHHeapSize = GetProcAddress(handle, 'nvrtcGetPCHHeapSize')

        global __nvrtcSetPCHHeapSize
        __nvrtcSetPCHHeapSize = GetProcAddress(handle, 'nvrtcSetPCHHeapSize')

        global __nvrtcGetPCHCreateStatus
        __nvrtcGetPCHCreateStatus = GetProcAddress(handle, 'nvrtcGetPCHCreateStatus')

        global __nvrtcGetPCHHeapSizeRequired
        __nvrtcGetPCHHeapSizeRequired = GetProcAddress(handle, 'nvrtcGetPCHHeapSizeRequired')

        global __nvrtcSetFlowCallback
        __nvrtcSetFlowCallback = GetProcAddress(handle, 'nvrtcSetFlowCallback')

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
