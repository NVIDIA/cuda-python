# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This code was automatically generated across versions from 1.5.0 to 13.3.0. Do not modify it directly.
# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=32765bada1ce206d4a0f4790d6c14563b07e79567fc4569c6a843ccb086ab620


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

cdef extern from "<windows.h>":
    ctypedef void* HMODULE
    void* _cyb_GetProcAddress "GetProcAddress"(HMODULE, const char*) nogil

from libc.stdint cimport intptr_t as _cyb_intptr_t

import threading as _cyb_threading

cdef int _cyb___py_cudla_init = 0
cdef dict _cyb_func_ptrs = None
cdef object _cyb_symbol_lock = _cyb_threading.Lock()

# <<<< END OF PREAMBLE CONTENT >>>>

from libc.stdint cimport uintptr_t

from .utils import FunctionNotFoundError, NotSupportedError

from cuda.pathfinder import load_nvidia_dynamic_lib


###############################################################################
# Wrapper init
###############################################################################


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
    global _cyb___py_cudla_init

    cdef int err
    cdef uintptr_t handle
    with gil, _cyb_symbol_lock:
        if _cyb___py_cudla_init: return 0

        handle = load_library()
        global __cudlaGetVersion
        __cudlaGetVersion = _cyb_GetProcAddress(<HMODULE>handle, 'cudlaGetVersion')

        global __cudlaDeviceGetCount
        __cudlaDeviceGetCount = _cyb_GetProcAddress(<HMODULE>handle, 'cudlaDeviceGetCount')

        global __cudlaCreateDevice
        __cudlaCreateDevice = _cyb_GetProcAddress(<HMODULE>handle, 'cudlaCreateDevice')

        global __cudlaMemRegister
        __cudlaMemRegister = _cyb_GetProcAddress(<HMODULE>handle, 'cudlaMemRegister')

        global __cudlaModuleLoadFromMemory
        __cudlaModuleLoadFromMemory = _cyb_GetProcAddress(<HMODULE>handle, 'cudlaModuleLoadFromMemory')

        global __cudlaModuleGetAttributes
        __cudlaModuleGetAttributes = _cyb_GetProcAddress(<HMODULE>handle, 'cudlaModuleGetAttributes')

        global __cudlaModuleUnload
        __cudlaModuleUnload = _cyb_GetProcAddress(<HMODULE>handle, 'cudlaModuleUnload')

        global __cudlaSubmitTask
        __cudlaSubmitTask = _cyb_GetProcAddress(<HMODULE>handle, 'cudlaSubmitTask')

        global __cudlaDeviceGetAttribute
        __cudlaDeviceGetAttribute = _cyb_GetProcAddress(<HMODULE>handle, 'cudlaDeviceGetAttribute')

        global __cudlaMemUnregister
        __cudlaMemUnregister = _cyb_GetProcAddress(<HMODULE>handle, 'cudlaMemUnregister')

        global __cudlaGetLastError
        __cudlaGetLastError = _cyb_GetProcAddress(<HMODULE>handle, 'cudlaGetLastError')

        global __cudlaDestroyDevice
        __cudlaDestroyDevice = _cyb_GetProcAddress(<HMODULE>handle, 'cudlaDestroyDevice')

        global __cudlaSetTaskTimeoutInMs
        __cudlaSetTaskTimeoutInMs = _cyb_GetProcAddress(<HMODULE>handle, 'cudlaSetTaskTimeoutInMs')

        _cyb_atomic_int_store(<int *>&_cyb___py_cudla_init, 1)
        return 0

cdef inline int _check_or_init_cudla() except -1 nogil:
    if _cyb_atomic_int_load(<int *>&_cyb___py_cudla_init):
        return 0

    return _init_cudla()


cpdef dict _inspect_function_pointers():
    global _cyb_func_ptrs
    if _cyb_func_ptrs is not None:
        return _cyb_func_ptrs

    _check_or_init_cudla()
    cdef dict data = {}
    global __cudlaGetVersion
    data["__cudlaGetVersion"] = <_cyb_intptr_t>__cudlaGetVersion

    global __cudlaDeviceGetCount
    data["__cudlaDeviceGetCount"] = <_cyb_intptr_t>__cudlaDeviceGetCount

    global __cudlaCreateDevice
    data["__cudlaCreateDevice"] = <_cyb_intptr_t>__cudlaCreateDevice

    global __cudlaMemRegister
    data["__cudlaMemRegister"] = <_cyb_intptr_t>__cudlaMemRegister

    global __cudlaModuleLoadFromMemory
    data["__cudlaModuleLoadFromMemory"] = <_cyb_intptr_t>__cudlaModuleLoadFromMemory

    global __cudlaModuleGetAttributes
    data["__cudlaModuleGetAttributes"] = <_cyb_intptr_t>__cudlaModuleGetAttributes

    global __cudlaModuleUnload
    data["__cudlaModuleUnload"] = <_cyb_intptr_t>__cudlaModuleUnload

    global __cudlaSubmitTask
    data["__cudlaSubmitTask"] = <_cyb_intptr_t>__cudlaSubmitTask

    global __cudlaDeviceGetAttribute
    data["__cudlaDeviceGetAttribute"] = <_cyb_intptr_t>__cudlaDeviceGetAttribute

    global __cudlaMemUnregister
    data["__cudlaMemUnregister"] = <_cyb_intptr_t>__cudlaMemUnregister

    global __cudlaGetLastError
    data["__cudlaGetLastError"] = <_cyb_intptr_t>__cudlaGetLastError

    global __cudlaDestroyDevice
    data["__cudlaDestroyDevice"] = <_cyb_intptr_t>__cudlaDestroyDevice

    global __cudlaSetTaskTimeoutInMs
    data["__cudlaSetTaskTimeoutInMs"] = <_cyb_intptr_t>__cudlaSetTaskTimeoutInMs
    _cyb_func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global _cyb_func_ptrs
    if _cyb_func_ptrs is None:
        _cyb_func_ptrs = _inspect_function_pointers()
    return _cyb_func_ptrs[name]




cdef uintptr_t load_library() except* with gil:
    return load_nvidia_dynamic_lib("cudla")._handle_uint


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
