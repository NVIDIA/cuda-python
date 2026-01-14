# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.4.1 to 13.1.0. Do not modify it directly.

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
cdef bint __py_nvfatbin_init = False

cdef void* __nvFatbinGetErrorString = NULL
cdef void* __nvFatbinCreate = NULL
cdef void* __nvFatbinDestroy = NULL
cdef void* __nvFatbinAddPTX = NULL
cdef void* __nvFatbinAddCubin = NULL
cdef void* __nvFatbinAddLTOIR = NULL
cdef void* __nvFatbinSize = NULL
cdef void* __nvFatbinGet = NULL
cdef void* __nvFatbinVersion = NULL
cdef void* __nvFatbinAddReloc = NULL
cdef void* __nvFatbinAddTileIR = NULL


cdef int _init_nvfatbin() except -1 nogil:
    global __py_nvfatbin_init

    with gil, __symbol_lock:
        # Recheck the flag after obtaining the locks
        if __py_nvfatbin_init:
            return 0

        # Load library
        handle = load_nvidia_dynamic_lib("nvfatbin")._handle_uint

        # Load function
        global __nvFatbinGetErrorString
        __nvFatbinGetErrorString = GetProcAddress(handle, 'nvFatbinGetErrorString')

        global __nvFatbinCreate
        __nvFatbinCreate = GetProcAddress(handle, 'nvFatbinCreate')

        global __nvFatbinDestroy
        __nvFatbinDestroy = GetProcAddress(handle, 'nvFatbinDestroy')

        global __nvFatbinAddPTX
        __nvFatbinAddPTX = GetProcAddress(handle, 'nvFatbinAddPTX')

        global __nvFatbinAddCubin
        __nvFatbinAddCubin = GetProcAddress(handle, 'nvFatbinAddCubin')

        global __nvFatbinAddLTOIR
        __nvFatbinAddLTOIR = GetProcAddress(handle, 'nvFatbinAddLTOIR')

        global __nvFatbinSize
        __nvFatbinSize = GetProcAddress(handle, 'nvFatbinSize')

        global __nvFatbinGet
        __nvFatbinGet = GetProcAddress(handle, 'nvFatbinGet')

        global __nvFatbinVersion
        __nvFatbinVersion = GetProcAddress(handle, 'nvFatbinVersion')

        global __nvFatbinAddReloc
        __nvFatbinAddReloc = GetProcAddress(handle, 'nvFatbinAddReloc')

        global __nvFatbinAddTileIR
        __nvFatbinAddTileIR = GetProcAddress(handle, 'nvFatbinAddTileIR')

        __py_nvfatbin_init = True
        return 0


cdef inline int _check_or_init_nvfatbin() except -1 nogil:
    if __py_nvfatbin_init:
        return 0

    return _init_nvfatbin()


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_nvfatbin()
    cdef dict data = {}

    global __nvFatbinGetErrorString
    data["__nvFatbinGetErrorString"] = <intptr_t>__nvFatbinGetErrorString

    global __nvFatbinCreate
    data["__nvFatbinCreate"] = <intptr_t>__nvFatbinCreate

    global __nvFatbinDestroy
    data["__nvFatbinDestroy"] = <intptr_t>__nvFatbinDestroy

    global __nvFatbinAddPTX
    data["__nvFatbinAddPTX"] = <intptr_t>__nvFatbinAddPTX

    global __nvFatbinAddCubin
    data["__nvFatbinAddCubin"] = <intptr_t>__nvFatbinAddCubin

    global __nvFatbinAddLTOIR
    data["__nvFatbinAddLTOIR"] = <intptr_t>__nvFatbinAddLTOIR

    global __nvFatbinSize
    data["__nvFatbinSize"] = <intptr_t>__nvFatbinSize

    global __nvFatbinGet
    data["__nvFatbinGet"] = <intptr_t>__nvFatbinGet

    global __nvFatbinVersion
    data["__nvFatbinVersion"] = <intptr_t>__nvFatbinVersion

    global __nvFatbinAddReloc
    data["__nvFatbinAddReloc"] = <intptr_t>__nvFatbinAddReloc

    global __nvFatbinAddTileIR
    data["__nvFatbinAddTileIR"] = <intptr_t>__nvFatbinAddTileIR

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
