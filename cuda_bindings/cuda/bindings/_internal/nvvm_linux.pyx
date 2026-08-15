# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.0.1 to 13.3.0. Do not modify it directly.
# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=e0b66c45b6f66a7ab7bae49939a26007efef95651084f1ef09fd32848260f2c8


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

cdef int _cyb___py_nvvm_init = 0
cdef dict _cyb_func_ptrs = None
cdef object _cyb_symbol_lock = _cyb_threading.Lock()

# <<<< END OF PREAMBLE CONTENT >>>>

from libc.stdint cimport uintptr_t

from .utils import FunctionNotFoundError, NotSupportedError
from cuda.pathfinder import load_nvidia_dynamic_lib


###############################################################################
# Wrapper init
###############################################################################

cdef void* __nvvmGetErrorString = NULL
cdef void* __nvvmVersion = NULL
cdef void* __nvvmIRVersion = NULL
cdef void* __nvvmCreateProgram = NULL
cdef void* __nvvmDestroyProgram = NULL
cdef void* __nvvmAddModuleToProgram = NULL
cdef void* __nvvmLazyAddModuleToProgram = NULL
cdef void* __nvvmCompileProgram = NULL
cdef void* __nvvmVerifyProgram = NULL
cdef void* __nvvmGetCompiledResultSize = NULL
cdef void* __nvvmGetCompiledResult = NULL
cdef void* __nvvmGetProgramLogSize = NULL
cdef void* __nvvmGetProgramLog = NULL
cdef void* __nvvmLLVMVersion = NULL

cdef int _init_nvvm() except -1 nogil:
    global _cyb___py_nvvm_init
    cdef void* handle = NULL
    with gil, _cyb_symbol_lock:
        if _cyb___py_nvvm_init: return 0

        global __nvvmGetErrorString
        __nvvmGetErrorString = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvvmGetErrorString')
        if __nvvmGetErrorString == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmGetErrorString = _cyb_dlsym(handle, 'nvvmGetErrorString')

        global __nvvmVersion
        __nvvmVersion = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvvmVersion')
        if __nvvmVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmVersion = _cyb_dlsym(handle, 'nvvmVersion')

        global __nvvmIRVersion
        __nvvmIRVersion = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvvmIRVersion')
        if __nvvmIRVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmIRVersion = _cyb_dlsym(handle, 'nvvmIRVersion')

        global __nvvmCreateProgram
        __nvvmCreateProgram = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvvmCreateProgram')
        if __nvvmCreateProgram == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmCreateProgram = _cyb_dlsym(handle, 'nvvmCreateProgram')

        global __nvvmDestroyProgram
        __nvvmDestroyProgram = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvvmDestroyProgram')
        if __nvvmDestroyProgram == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmDestroyProgram = _cyb_dlsym(handle, 'nvvmDestroyProgram')

        global __nvvmAddModuleToProgram
        __nvvmAddModuleToProgram = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvvmAddModuleToProgram')
        if __nvvmAddModuleToProgram == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmAddModuleToProgram = _cyb_dlsym(handle, 'nvvmAddModuleToProgram')

        global __nvvmLazyAddModuleToProgram
        __nvvmLazyAddModuleToProgram = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvvmLazyAddModuleToProgram')
        if __nvvmLazyAddModuleToProgram == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmLazyAddModuleToProgram = _cyb_dlsym(handle, 'nvvmLazyAddModuleToProgram')

        global __nvvmCompileProgram
        __nvvmCompileProgram = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvvmCompileProgram')
        if __nvvmCompileProgram == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmCompileProgram = _cyb_dlsym(handle, 'nvvmCompileProgram')

        global __nvvmVerifyProgram
        __nvvmVerifyProgram = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvvmVerifyProgram')
        if __nvvmVerifyProgram == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmVerifyProgram = _cyb_dlsym(handle, 'nvvmVerifyProgram')

        global __nvvmGetCompiledResultSize
        __nvvmGetCompiledResultSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvvmGetCompiledResultSize')
        if __nvvmGetCompiledResultSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmGetCompiledResultSize = _cyb_dlsym(handle, 'nvvmGetCompiledResultSize')

        global __nvvmGetCompiledResult
        __nvvmGetCompiledResult = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvvmGetCompiledResult')
        if __nvvmGetCompiledResult == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmGetCompiledResult = _cyb_dlsym(handle, 'nvvmGetCompiledResult')

        global __nvvmGetProgramLogSize
        __nvvmGetProgramLogSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvvmGetProgramLogSize')
        if __nvvmGetProgramLogSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmGetProgramLogSize = _cyb_dlsym(handle, 'nvvmGetProgramLogSize')

        global __nvvmGetProgramLog
        __nvvmGetProgramLog = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvvmGetProgramLog')
        if __nvvmGetProgramLog == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmGetProgramLog = _cyb_dlsym(handle, 'nvvmGetProgramLog')

        global __nvvmLLVMVersion
        __nvvmLLVMVersion = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvvmLLVMVersion')
        if __nvvmLLVMVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmLLVMVersion = _cyb_dlsym(handle, 'nvvmLLVMVersion')

        _cyb_atomic_int_store(<int *>&_cyb___py_nvvm_init, 1)
        return 0

cdef inline int _check_or_init_nvvm() except -1 nogil:
    if _cyb_atomic_int_load(<int *>&_cyb___py_nvvm_init):
        return 0

    return _init_nvvm()


cpdef dict _inspect_function_pointers():
    global _cyb_func_ptrs
    if _cyb_func_ptrs is not None:
        return _cyb_func_ptrs

    _check_or_init_nvvm()
    cdef dict data = {}
    global __nvvmGetErrorString
    data["__nvvmGetErrorString"] = <_cyb_intptr_t>__nvvmGetErrorString

    global __nvvmVersion
    data["__nvvmVersion"] = <_cyb_intptr_t>__nvvmVersion

    global __nvvmIRVersion
    data["__nvvmIRVersion"] = <_cyb_intptr_t>__nvvmIRVersion

    global __nvvmCreateProgram
    data["__nvvmCreateProgram"] = <_cyb_intptr_t>__nvvmCreateProgram

    global __nvvmDestroyProgram
    data["__nvvmDestroyProgram"] = <_cyb_intptr_t>__nvvmDestroyProgram

    global __nvvmAddModuleToProgram
    data["__nvvmAddModuleToProgram"] = <_cyb_intptr_t>__nvvmAddModuleToProgram

    global __nvvmLazyAddModuleToProgram
    data["__nvvmLazyAddModuleToProgram"] = <_cyb_intptr_t>__nvvmLazyAddModuleToProgram

    global __nvvmCompileProgram
    data["__nvvmCompileProgram"] = <_cyb_intptr_t>__nvvmCompileProgram

    global __nvvmVerifyProgram
    data["__nvvmVerifyProgram"] = <_cyb_intptr_t>__nvvmVerifyProgram

    global __nvvmGetCompiledResultSize
    data["__nvvmGetCompiledResultSize"] = <_cyb_intptr_t>__nvvmGetCompiledResultSize

    global __nvvmGetCompiledResult
    data["__nvvmGetCompiledResult"] = <_cyb_intptr_t>__nvvmGetCompiledResult

    global __nvvmGetProgramLogSize
    data["__nvvmGetProgramLogSize"] = <_cyb_intptr_t>__nvvmGetProgramLogSize

    global __nvvmGetProgramLog
    data["__nvvmGetProgramLog"] = <_cyb_intptr_t>__nvvmGetProgramLog

    global __nvvmLLVMVersion
    data["__nvvmLLVMVersion"] = <_cyb_intptr_t>__nvvmLLVMVersion
    _cyb_func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global _cyb_func_ptrs
    if _cyb_func_ptrs is None:
        _cyb_func_ptrs = _inspect_function_pointers()
    return _cyb_func_ptrs[name]




cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("nvvm")._handle_uint
    return <void*>handle


###############################################################################
# Wrapper functions
###############################################################################

cdef const char* _nvvmGetErrorString(nvvmResult result) except?NULL nogil:
    global __nvvmGetErrorString
    _check_or_init_nvvm()
    if __nvvmGetErrorString == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmGetErrorString is not found")
    return (<const char* (*)(nvvmResult) noexcept nogil>__nvvmGetErrorString)(
        result)


cdef nvvmResult _nvvmVersion(int* major, int* minor) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvvmVersion
    _check_or_init_nvvm()
    if __nvvmVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmVersion is not found")
    return (<nvvmResult (*)(int*, int*) noexcept nogil>__nvvmVersion)(
        major, minor)


cdef nvvmResult _nvvmIRVersion(int* majorIR, int* minorIR, int* majorDbg, int* minorDbg) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvvmIRVersion
    _check_or_init_nvvm()
    if __nvvmIRVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmIRVersion is not found")
    return (<nvvmResult (*)(int*, int*, int*, int*) noexcept nogil>__nvvmIRVersion)(
        majorIR, minorIR, majorDbg, minorDbg)


cdef nvvmResult _nvvmCreateProgram(nvvmProgram* prog) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvvmCreateProgram
    _check_or_init_nvvm()
    if __nvvmCreateProgram == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmCreateProgram is not found")
    return (<nvvmResult (*)(nvvmProgram*) noexcept nogil>__nvvmCreateProgram)(
        prog)


cdef nvvmResult _nvvmDestroyProgram(nvvmProgram* prog) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvvmDestroyProgram
    _check_or_init_nvvm()
    if __nvvmDestroyProgram == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmDestroyProgram is not found")
    return (<nvvmResult (*)(nvvmProgram*) noexcept nogil>__nvvmDestroyProgram)(
        prog)


cdef nvvmResult _nvvmAddModuleToProgram(nvvmProgram prog, const char* buffer, size_t size, const char* name) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvvmAddModuleToProgram
    _check_or_init_nvvm()
    if __nvvmAddModuleToProgram == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmAddModuleToProgram is not found")
    return (<nvvmResult (*)(nvvmProgram, const char*, size_t, const char*) noexcept nogil>__nvvmAddModuleToProgram)(
        prog, buffer, size, name)


cdef nvvmResult _nvvmLazyAddModuleToProgram(nvvmProgram prog, const char* buffer, size_t size, const char* name) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvvmLazyAddModuleToProgram
    _check_or_init_nvvm()
    if __nvvmLazyAddModuleToProgram == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmLazyAddModuleToProgram is not found")
    return (<nvvmResult (*)(nvvmProgram, const char*, size_t, const char*) noexcept nogil>__nvvmLazyAddModuleToProgram)(
        prog, buffer, size, name)


cdef nvvmResult _nvvmCompileProgram(nvvmProgram prog, int numOptions, const char** options) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvvmCompileProgram
    _check_or_init_nvvm()
    if __nvvmCompileProgram == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmCompileProgram is not found")
    return (<nvvmResult (*)(nvvmProgram, int, const char**) noexcept nogil>__nvvmCompileProgram)(
        prog, numOptions, options)


cdef nvvmResult _nvvmVerifyProgram(nvvmProgram prog, int numOptions, const char** options) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvvmVerifyProgram
    _check_or_init_nvvm()
    if __nvvmVerifyProgram == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmVerifyProgram is not found")
    return (<nvvmResult (*)(nvvmProgram, int, const char**) noexcept nogil>__nvvmVerifyProgram)(
        prog, numOptions, options)


cdef nvvmResult _nvvmGetCompiledResultSize(nvvmProgram prog, size_t* bufferSizeRet) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvvmGetCompiledResultSize
    _check_or_init_nvvm()
    if __nvvmGetCompiledResultSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmGetCompiledResultSize is not found")
    return (<nvvmResult (*)(nvvmProgram, size_t*) noexcept nogil>__nvvmGetCompiledResultSize)(
        prog, bufferSizeRet)


cdef nvvmResult _nvvmGetCompiledResult(nvvmProgram prog, char* buffer) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvvmGetCompiledResult
    _check_or_init_nvvm()
    if __nvvmGetCompiledResult == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmGetCompiledResult is not found")
    return (<nvvmResult (*)(nvvmProgram, char*) noexcept nogil>__nvvmGetCompiledResult)(
        prog, buffer)


cdef nvvmResult _nvvmGetProgramLogSize(nvvmProgram prog, size_t* bufferSizeRet) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvvmGetProgramLogSize
    _check_or_init_nvvm()
    if __nvvmGetProgramLogSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmGetProgramLogSize is not found")
    return (<nvvmResult (*)(nvvmProgram, size_t*) noexcept nogil>__nvvmGetProgramLogSize)(
        prog, bufferSizeRet)


cdef nvvmResult _nvvmGetProgramLog(nvvmProgram prog, char* buffer) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvvmGetProgramLog
    _check_or_init_nvvm()
    if __nvvmGetProgramLog == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmGetProgramLog is not found")
    return (<nvvmResult (*)(nvvmProgram, char*) noexcept nogil>__nvvmGetProgramLog)(
        prog, buffer)


cdef nvvmResult _nvvmLLVMVersion(const char* arch, int* major) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvvmLLVMVersion
    _check_or_init_nvvm()
    if __nvvmLLVMVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmLLVMVersion is not found")
    return (<nvvmResult (*)(const char*, int*) noexcept nogil>__nvvmLLVMVersion)(
        arch, major)
