# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# This code was automatically generated across versions from 1.5.0 to 13.3.0, generator version 0.3.1.dev1465+gc5c5c8652. Do not modify it directly.

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
cdef bint __py_cudla_init = False

cdef void* __cudlaGetVersion = NULL
cdef void* __cudlaDeviceGetCount = NULL
cdef void* __cudlaCreateDevice = NULL
cdef void* __cudlaMemRegister = NULL
cdef void* __cudlaModuleLoadFromMemory = NULL
cdef void* __cudlaModuleGetAttributes = NULL
cdef void* __cudlaModuleUnload = NULL
cdef void* __cudlaSubmitTask = NULL
cdef void* __cudlaDeviceGetAttribute = NULL
cdef void* __cudlaMemUnregister = NULL
cdef void* __cudlaGetLastError = NULL
cdef void* __cudlaDestroyDevice = NULL
cdef void* __cudlaSetTaskTimeoutInMs = NULL


cdef int _init_cudla() except -1 nogil:
    global __py_cudla_init

    with gil, __symbol_lock:
        # Recheck the flag after obtaining the locks
        if __py_cudla_init:
            return 0

        # Load library
        handle = load_nvidia_dynamic_lib("cudla")._handle_uint

        # Load function
        global __cudlaGetVersion
        __cudlaGetVersion = GetProcAddress(handle, 'cudlaGetVersion')

        global __cudlaDeviceGetCount
        __cudlaDeviceGetCount = GetProcAddress(handle, 'cudlaDeviceGetCount')

        global __cudlaCreateDevice
        __cudlaCreateDevice = GetProcAddress(handle, 'cudlaCreateDevice')

        global __cudlaMemRegister
        __cudlaMemRegister = GetProcAddress(handle, 'cudlaMemRegister')

        global __cudlaModuleLoadFromMemory
        __cudlaModuleLoadFromMemory = GetProcAddress(handle, 'cudlaModuleLoadFromMemory')

        global __cudlaModuleGetAttributes
        __cudlaModuleGetAttributes = GetProcAddress(handle, 'cudlaModuleGetAttributes')

        global __cudlaModuleUnload
        __cudlaModuleUnload = GetProcAddress(handle, 'cudlaModuleUnload')

        global __cudlaSubmitTask
        __cudlaSubmitTask = GetProcAddress(handle, 'cudlaSubmitTask')

        global __cudlaDeviceGetAttribute
        __cudlaDeviceGetAttribute = GetProcAddress(handle, 'cudlaDeviceGetAttribute')

        global __cudlaMemUnregister
        __cudlaMemUnregister = GetProcAddress(handle, 'cudlaMemUnregister')

        global __cudlaGetLastError
        __cudlaGetLastError = GetProcAddress(handle, 'cudlaGetLastError')

        global __cudlaDestroyDevice
        __cudlaDestroyDevice = GetProcAddress(handle, 'cudlaDestroyDevice')

        global __cudlaSetTaskTimeoutInMs
        __cudlaSetTaskTimeoutInMs = GetProcAddress(handle, 'cudlaSetTaskTimeoutInMs')

        __py_cudla_init = True
        return 0


cdef inline int _check_or_init_cudla() except -1 nogil:
    if __py_cudla_init:
        return 0

    return _init_cudla()


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_cudla()
    cdef dict data = {}

    global __cudlaGetVersion
    data["__cudlaGetVersion"] = <intptr_t>__cudlaGetVersion

    global __cudlaDeviceGetCount
    data["__cudlaDeviceGetCount"] = <intptr_t>__cudlaDeviceGetCount

    global __cudlaCreateDevice
    data["__cudlaCreateDevice"] = <intptr_t>__cudlaCreateDevice

    global __cudlaMemRegister
    data["__cudlaMemRegister"] = <intptr_t>__cudlaMemRegister

    global __cudlaModuleLoadFromMemory
    data["__cudlaModuleLoadFromMemory"] = <intptr_t>__cudlaModuleLoadFromMemory

    global __cudlaModuleGetAttributes
    data["__cudlaModuleGetAttributes"] = <intptr_t>__cudlaModuleGetAttributes

    global __cudlaModuleUnload
    data["__cudlaModuleUnload"] = <intptr_t>__cudlaModuleUnload

    global __cudlaSubmitTask
    data["__cudlaSubmitTask"] = <intptr_t>__cudlaSubmitTask

    global __cudlaDeviceGetAttribute
    data["__cudlaDeviceGetAttribute"] = <intptr_t>__cudlaDeviceGetAttribute

    global __cudlaMemUnregister
    data["__cudlaMemUnregister"] = <intptr_t>__cudlaMemUnregister

    global __cudlaGetLastError
    data["__cudlaGetLastError"] = <intptr_t>__cudlaGetLastError

    global __cudlaDestroyDevice
    data["__cudlaDestroyDevice"] = <intptr_t>__cudlaDestroyDevice

    global __cudlaSetTaskTimeoutInMs
    data["__cudlaSetTaskTimeoutInMs"] = <intptr_t>__cudlaSetTaskTimeoutInMs

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

cdef cudlaStatus _cudlaGetVersion(uint64_t* const version) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    global __cudlaGetVersion
    _check_or_init_cudla()
    if __cudlaGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cudlaGetVersion is not found")
    return (<cudlaStatus (*)(uint64_t* const) noexcept nogil>__cudlaGetVersion)(
        version)


cdef cudlaStatus _cudlaDeviceGetCount(uint64_t* const pNumDevices) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    global __cudlaDeviceGetCount
    _check_or_init_cudla()
    if __cudlaDeviceGetCount == NULL:
        with gil:
            raise FunctionNotFoundError("function cudlaDeviceGetCount is not found")
    return (<cudlaStatus (*)(uint64_t* const) noexcept nogil>__cudlaDeviceGetCount)(
        pNumDevices)


cdef cudlaStatus _cudlaCreateDevice(const uint64_t device, cudlaDevHandle* const devHandle, const uint32_t flags) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    global __cudlaCreateDevice
    _check_or_init_cudla()
    if __cudlaCreateDevice == NULL:
        with gil:
            raise FunctionNotFoundError("function cudlaCreateDevice is not found")
    return (<cudlaStatus (*)(const uint64_t, cudlaDevHandle* const, const uint32_t) noexcept nogil>__cudlaCreateDevice)(
        device, devHandle, flags)


cdef cudlaStatus _cudlaMemRegister(const cudlaDevHandle devHandle, const uint64_t* const ptr, const size_t size, uint64_t** const devPtr, const uint32_t flags) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    global __cudlaMemRegister
    _check_or_init_cudla()
    if __cudlaMemRegister == NULL:
        with gil:
            raise FunctionNotFoundError("function cudlaMemRegister is not found")
    return (<cudlaStatus (*)(const cudlaDevHandle, const uint64_t* const, const size_t, uint64_t** const, const uint32_t) noexcept nogil>__cudlaMemRegister)(
        devHandle, ptr, size, devPtr, flags)


cdef cudlaStatus _cudlaModuleLoadFromMemory(const cudlaDevHandle devHandle, const uint8_t* const pModule, const size_t moduleSize, cudlaModule* const hModule, const uint32_t flags) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    global __cudlaModuleLoadFromMemory
    _check_or_init_cudla()
    if __cudlaModuleLoadFromMemory == NULL:
        with gil:
            raise FunctionNotFoundError("function cudlaModuleLoadFromMemory is not found")
    return (<cudlaStatus (*)(const cudlaDevHandle, const uint8_t* const, const size_t, cudlaModule* const, const uint32_t) noexcept nogil>__cudlaModuleLoadFromMemory)(
        devHandle, pModule, moduleSize, hModule, flags)


cdef cudlaStatus _cudlaModuleGetAttributes(const cudlaModule hModule, const cudlaModuleAttributeType attrType, cudlaModuleAttribute* const attribute) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    global __cudlaModuleGetAttributes
    _check_or_init_cudla()
    if __cudlaModuleGetAttributes == NULL:
        with gil:
            raise FunctionNotFoundError("function cudlaModuleGetAttributes is not found")
    return (<cudlaStatus (*)(const cudlaModule, const cudlaModuleAttributeType, cudlaModuleAttribute* const) noexcept nogil>__cudlaModuleGetAttributes)(
        hModule, attrType, attribute)


cdef cudlaStatus _cudlaModuleUnload(const cudlaModule hModule, const uint32_t flags) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    global __cudlaModuleUnload
    _check_or_init_cudla()
    if __cudlaModuleUnload == NULL:
        with gil:
            raise FunctionNotFoundError("function cudlaModuleUnload is not found")
    return (<cudlaStatus (*)(const cudlaModule, const uint32_t) noexcept nogil>__cudlaModuleUnload)(
        hModule, flags)


cdef cudlaStatus _cudlaSubmitTask(const cudlaDevHandle devHandle, const cudlaTask* const ptrToTasks, const uint32_t numTasks, void* const stream, const uint32_t flags) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    global __cudlaSubmitTask
    _check_or_init_cudla()
    if __cudlaSubmitTask == NULL:
        with gil:
            raise FunctionNotFoundError("function cudlaSubmitTask is not found")
    return (<cudlaStatus (*)(const cudlaDevHandle, const cudlaTask* const, const uint32_t, void* const, const uint32_t) noexcept nogil>__cudlaSubmitTask)(
        devHandle, ptrToTasks, numTasks, stream, flags)


cdef cudlaStatus _cudlaDeviceGetAttribute(const cudlaDevHandle devHandle, const cudlaDevAttributeType attrib, cudlaDevAttribute* const pAttribute) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    global __cudlaDeviceGetAttribute
    _check_or_init_cudla()
    if __cudlaDeviceGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cudlaDeviceGetAttribute is not found")
    return (<cudlaStatus (*)(const cudlaDevHandle, const cudlaDevAttributeType, cudlaDevAttribute* const) noexcept nogil>__cudlaDeviceGetAttribute)(
        devHandle, attrib, pAttribute)


cdef cudlaStatus _cudlaMemUnregister(const cudlaDevHandle devHandle, const uint64_t* const devPtr) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    global __cudlaMemUnregister
    _check_or_init_cudla()
    if __cudlaMemUnregister == NULL:
        with gil:
            raise FunctionNotFoundError("function cudlaMemUnregister is not found")
    return (<cudlaStatus (*)(const cudlaDevHandle, const uint64_t* const) noexcept nogil>__cudlaMemUnregister)(
        devHandle, devPtr)


cdef cudlaStatus _cudlaGetLastError(const cudlaDevHandle devHandle) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    global __cudlaGetLastError
    _check_or_init_cudla()
    if __cudlaGetLastError == NULL:
        with gil:
            raise FunctionNotFoundError("function cudlaGetLastError is not found")
    return (<cudlaStatus (*)(const cudlaDevHandle) noexcept nogil>__cudlaGetLastError)(
        devHandle)


cdef cudlaStatus _cudlaDestroyDevice(const cudlaDevHandle devHandle) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    global __cudlaDestroyDevice
    _check_or_init_cudla()
    if __cudlaDestroyDevice == NULL:
        with gil:
            raise FunctionNotFoundError("function cudlaDestroyDevice is not found")
    return (<cudlaStatus (*)(const cudlaDevHandle) noexcept nogil>__cudlaDestroyDevice)(
        devHandle)


cdef cudlaStatus _cudlaSetTaskTimeoutInMs(const cudlaDevHandle devHandle, const uint32_t timeout) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    global __cudlaSetTaskTimeoutInMs
    _check_or_init_cudla()
    if __cudlaSetTaskTimeoutInMs == NULL:
        with gil:
            raise FunctionNotFoundError("function cudlaSetTaskTimeoutInMs is not found")
    return (<cudlaStatus (*)(const cudlaDevHandle, const uint32_t) noexcept nogil>__cudlaSetTaskTimeoutInMs)(
        devHandle, timeout)
