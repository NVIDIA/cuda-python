# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.0.1 to 13.0.1. Do not modify it directly.

from libc.stdint cimport intptr_t, uintptr_t

import threading
from .utils import FunctionNotFoundError, NotSupportedError

from cuda.pathfinder import load_nvidia_dynamic_lib


###############################################################################
# Extern
###############################################################################

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
cdef bint __py_nvvm_init = False

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


cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("nvvm")._handle_uint
    return <void*>handle


cdef int _check_or_init_nvvm() except -1 nogil:
    global __py_nvvm_init
    if __py_nvvm_init:
        return 0

    cdef void* handle = NULL

    with gil, __symbol_lock:
        # Load function
        global __nvvmGetErrorString
        __nvvmGetErrorString = dlsym(RTLD_DEFAULT, 'nvvmGetErrorString')
        if __nvvmGetErrorString == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmGetErrorString = dlsym(handle, 'nvvmGetErrorString')

        global __nvvmVersion
        __nvvmVersion = dlsym(RTLD_DEFAULT, 'nvvmVersion')
        if __nvvmVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmVersion = dlsym(handle, 'nvvmVersion')

        global __nvvmIRVersion
        __nvvmIRVersion = dlsym(RTLD_DEFAULT, 'nvvmIRVersion')
        if __nvvmIRVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmIRVersion = dlsym(handle, 'nvvmIRVersion')

        global __nvvmCreateProgram
        __nvvmCreateProgram = dlsym(RTLD_DEFAULT, 'nvvmCreateProgram')
        if __nvvmCreateProgram == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmCreateProgram = dlsym(handle, 'nvvmCreateProgram')

        global __nvvmDestroyProgram
        __nvvmDestroyProgram = dlsym(RTLD_DEFAULT, 'nvvmDestroyProgram')
        if __nvvmDestroyProgram == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmDestroyProgram = dlsym(handle, 'nvvmDestroyProgram')

        global __nvvmAddModuleToProgram
        __nvvmAddModuleToProgram = dlsym(RTLD_DEFAULT, 'nvvmAddModuleToProgram')
        if __nvvmAddModuleToProgram == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmAddModuleToProgram = dlsym(handle, 'nvvmAddModuleToProgram')

        global __nvvmLazyAddModuleToProgram
        __nvvmLazyAddModuleToProgram = dlsym(RTLD_DEFAULT, 'nvvmLazyAddModuleToProgram')
        if __nvvmLazyAddModuleToProgram == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmLazyAddModuleToProgram = dlsym(handle, 'nvvmLazyAddModuleToProgram')

        global __nvvmCompileProgram
        __nvvmCompileProgram = dlsym(RTLD_DEFAULT, 'nvvmCompileProgram')
        if __nvvmCompileProgram == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmCompileProgram = dlsym(handle, 'nvvmCompileProgram')

        global __nvvmVerifyProgram
        __nvvmVerifyProgram = dlsym(RTLD_DEFAULT, 'nvvmVerifyProgram')
        if __nvvmVerifyProgram == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmVerifyProgram = dlsym(handle, 'nvvmVerifyProgram')

        global __nvvmGetCompiledResultSize
        __nvvmGetCompiledResultSize = dlsym(RTLD_DEFAULT, 'nvvmGetCompiledResultSize')
        if __nvvmGetCompiledResultSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmGetCompiledResultSize = dlsym(handle, 'nvvmGetCompiledResultSize')

        global __nvvmGetCompiledResult
        __nvvmGetCompiledResult = dlsym(RTLD_DEFAULT, 'nvvmGetCompiledResult')
        if __nvvmGetCompiledResult == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmGetCompiledResult = dlsym(handle, 'nvvmGetCompiledResult')

        global __nvvmGetProgramLogSize
        __nvvmGetProgramLogSize = dlsym(RTLD_DEFAULT, 'nvvmGetProgramLogSize')
        if __nvvmGetProgramLogSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmGetProgramLogSize = dlsym(handle, 'nvvmGetProgramLogSize')

        global __nvvmGetProgramLog
        __nvvmGetProgramLog = dlsym(RTLD_DEFAULT, 'nvvmGetProgramLog')
        if __nvvmGetProgramLog == NULL:
            if handle == NULL:
                handle = load_library()
            __nvvmGetProgramLog = dlsym(handle, 'nvvmGetProgramLog')

        __py_nvvm_init = True
        return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_nvvm()
    cdef dict data = {}

    global __nvvmGetErrorString
    data["__nvvmGetErrorString"] = <intptr_t>__nvvmGetErrorString

    global __nvvmVersion
    data["__nvvmVersion"] = <intptr_t>__nvvmVersion

    global __nvvmIRVersion
    data["__nvvmIRVersion"] = <intptr_t>__nvvmIRVersion

    global __nvvmCreateProgram
    data["__nvvmCreateProgram"] = <intptr_t>__nvvmCreateProgram

    global __nvvmDestroyProgram
    data["__nvvmDestroyProgram"] = <intptr_t>__nvvmDestroyProgram

    global __nvvmAddModuleToProgram
    data["__nvvmAddModuleToProgram"] = <intptr_t>__nvvmAddModuleToProgram

    global __nvvmLazyAddModuleToProgram
    data["__nvvmLazyAddModuleToProgram"] = <intptr_t>__nvvmLazyAddModuleToProgram

    global __nvvmCompileProgram
    data["__nvvmCompileProgram"] = <intptr_t>__nvvmCompileProgram

    global __nvvmVerifyProgram
    data["__nvvmVerifyProgram"] = <intptr_t>__nvvmVerifyProgram

    global __nvvmGetCompiledResultSize
    data["__nvvmGetCompiledResultSize"] = <intptr_t>__nvvmGetCompiledResultSize

    global __nvvmGetCompiledResult
    data["__nvvmGetCompiledResult"] = <intptr_t>__nvvmGetCompiledResult

    global __nvvmGetProgramLogSize
    data["__nvvmGetProgramLogSize"] = <intptr_t>__nvvmGetProgramLogSize

    global __nvvmGetProgramLog
    data["__nvvmGetProgramLog"] = <intptr_t>__nvvmGetProgramLog

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
