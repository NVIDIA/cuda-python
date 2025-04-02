# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 11.0.3 to 12.8.0. Do not modify it directly.

from libc.stdint cimport intptr_t, uintptr_t

from .utils import FunctionNotFoundError, NotSupportedError

from cuda.bindings import path_finder

import os
import site

import win32api


###############################################################################
# Wrapper init
###############################################################################

LOAD_LIBRARY_SEARCH_SYSTEM32     = 0x00000800
LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000
LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100
cdef bint __py_nvvm_init = False
cdef void* __cuDriverGetVersion = NULL

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


cdef void* load_library(int driver_ver) except* with gil:
    cdef uintptr_t handle = path_finder.load_nvidia_dynamic_library("nvvm")
    return <void*>handle


cdef int _check_or_init_nvvm() except -1 nogil:
    global __py_nvvm_init
    if __py_nvvm_init:
        return 0

    cdef int err, driver_ver
    cdef intptr_t handle
    with gil:
        # Load driver to check version
        try:
            nvcuda_handle = win32api.LoadLibraryEx("nvcuda.dll", 0, LOAD_LIBRARY_SEARCH_SYSTEM32)
        except Exception as e:
            raise NotSupportedError(f'CUDA driver is not found ({e})')
        global __cuDriverGetVersion
        if __cuDriverGetVersion == NULL:
            __cuDriverGetVersion = <void*><intptr_t>win32api.GetProcAddress(nvcuda_handle, 'cuDriverGetVersion')
            if __cuDriverGetVersion == NULL:
                raise RuntimeError('something went wrong')
        err = (<int (*)(int*) nogil>__cuDriverGetVersion)(&driver_ver)
        if err != 0:
            raise RuntimeError('something went wrong')

        # Load library
        handle = <intptr_t>load_library(driver_ver)

        # Load function
        global __nvvmVersion
        try:
            __nvvmVersion = <void*><intptr_t>win32api.GetProcAddress(handle, 'nvvmVersion')
        except:
            pass

        global __nvvmIRVersion
        try:
            __nvvmIRVersion = <void*><intptr_t>win32api.GetProcAddress(handle, 'nvvmIRVersion')
        except:
            pass

        global __nvvmCreateProgram
        try:
            __nvvmCreateProgram = <void*><intptr_t>win32api.GetProcAddress(handle, 'nvvmCreateProgram')
        except:
            pass

        global __nvvmDestroyProgram
        try:
            __nvvmDestroyProgram = <void*><intptr_t>win32api.GetProcAddress(handle, 'nvvmDestroyProgram')
        except:
            pass

        global __nvvmAddModuleToProgram
        try:
            __nvvmAddModuleToProgram = <void*><intptr_t>win32api.GetProcAddress(handle, 'nvvmAddModuleToProgram')
        except:
            pass

        global __nvvmLazyAddModuleToProgram
        try:
            __nvvmLazyAddModuleToProgram = <void*><intptr_t>win32api.GetProcAddress(handle, 'nvvmLazyAddModuleToProgram')
        except:
            pass

        global __nvvmCompileProgram
        try:
            __nvvmCompileProgram = <void*><intptr_t>win32api.GetProcAddress(handle, 'nvvmCompileProgram')
        except:
            pass

        global __nvvmVerifyProgram
        try:
            __nvvmVerifyProgram = <void*><intptr_t>win32api.GetProcAddress(handle, 'nvvmVerifyProgram')
        except:
            pass

        global __nvvmGetCompiledResultSize
        try:
            __nvvmGetCompiledResultSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'nvvmGetCompiledResultSize')
        except:
            pass

        global __nvvmGetCompiledResult
        try:
            __nvvmGetCompiledResult = <void*><intptr_t>win32api.GetProcAddress(handle, 'nvvmGetCompiledResult')
        except:
            pass

        global __nvvmGetProgramLogSize
        try:
            __nvvmGetProgramLogSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'nvvmGetProgramLogSize')
        except:
            pass

        global __nvvmGetProgramLog
        try:
            __nvvmGetProgramLog = <void*><intptr_t>win32api.GetProcAddress(handle, 'nvvmGetProgramLog')
        except:
            pass

    __py_nvvm_init = True
    return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_nvvm()
    cdef dict data = {}

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

cdef nvvmResult _nvvmVersion(int* major, int* minor) except* nogil:
    global __nvvmVersion
    _check_or_init_nvvm()
    if __nvvmVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmVersion is not found")
    return (<nvvmResult (*)(int*, int*) nogil>__nvvmVersion)(
        major, minor)


cdef nvvmResult _nvvmIRVersion(int* majorIR, int* minorIR, int* majorDbg, int* minorDbg) except* nogil:
    global __nvvmIRVersion
    _check_or_init_nvvm()
    if __nvvmIRVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmIRVersion is not found")
    return (<nvvmResult (*)(int*, int*, int*, int*) nogil>__nvvmIRVersion)(
        majorIR, minorIR, majorDbg, minorDbg)


cdef nvvmResult _nvvmCreateProgram(nvvmProgram* prog) except* nogil:
    global __nvvmCreateProgram
    _check_or_init_nvvm()
    if __nvvmCreateProgram == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmCreateProgram is not found")
    return (<nvvmResult (*)(nvvmProgram*) nogil>__nvvmCreateProgram)(
        prog)


cdef nvvmResult _nvvmDestroyProgram(nvvmProgram* prog) except* nogil:
    global __nvvmDestroyProgram
    _check_or_init_nvvm()
    if __nvvmDestroyProgram == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmDestroyProgram is not found")
    return (<nvvmResult (*)(nvvmProgram*) nogil>__nvvmDestroyProgram)(
        prog)


cdef nvvmResult _nvvmAddModuleToProgram(nvvmProgram prog, const char* buffer, size_t size, const char* name) except* nogil:
    global __nvvmAddModuleToProgram
    _check_or_init_nvvm()
    if __nvvmAddModuleToProgram == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmAddModuleToProgram is not found")
    return (<nvvmResult (*)(nvvmProgram, const char*, size_t, const char*) nogil>__nvvmAddModuleToProgram)(
        prog, buffer, size, name)


cdef nvvmResult _nvvmLazyAddModuleToProgram(nvvmProgram prog, const char* buffer, size_t size, const char* name) except* nogil:
    global __nvvmLazyAddModuleToProgram
    _check_or_init_nvvm()
    if __nvvmLazyAddModuleToProgram == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmLazyAddModuleToProgram is not found")
    return (<nvvmResult (*)(nvvmProgram, const char*, size_t, const char*) nogil>__nvvmLazyAddModuleToProgram)(
        prog, buffer, size, name)


cdef nvvmResult _nvvmCompileProgram(nvvmProgram prog, int numOptions, const char** options) except* nogil:
    global __nvvmCompileProgram
    _check_or_init_nvvm()
    if __nvvmCompileProgram == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmCompileProgram is not found")
    return (<nvvmResult (*)(nvvmProgram, int, const char**) nogil>__nvvmCompileProgram)(
        prog, numOptions, options)


cdef nvvmResult _nvvmVerifyProgram(nvvmProgram prog, int numOptions, const char** options) except* nogil:
    global __nvvmVerifyProgram
    _check_or_init_nvvm()
    if __nvvmVerifyProgram == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmVerifyProgram is not found")
    return (<nvvmResult (*)(nvvmProgram, int, const char**) nogil>__nvvmVerifyProgram)(
        prog, numOptions, options)


cdef nvvmResult _nvvmGetCompiledResultSize(nvvmProgram prog, size_t* bufferSizeRet) except* nogil:
    global __nvvmGetCompiledResultSize
    _check_or_init_nvvm()
    if __nvvmGetCompiledResultSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmGetCompiledResultSize is not found")
    return (<nvvmResult (*)(nvvmProgram, size_t*) nogil>__nvvmGetCompiledResultSize)(
        prog, bufferSizeRet)


cdef nvvmResult _nvvmGetCompiledResult(nvvmProgram prog, char* buffer) except* nogil:
    global __nvvmGetCompiledResult
    _check_or_init_nvvm()
    if __nvvmGetCompiledResult == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmGetCompiledResult is not found")
    return (<nvvmResult (*)(nvvmProgram, char*) nogil>__nvvmGetCompiledResult)(
        prog, buffer)


cdef nvvmResult _nvvmGetProgramLogSize(nvvmProgram prog, size_t* bufferSizeRet) except* nogil:
    global __nvvmGetProgramLogSize
    _check_or_init_nvvm()
    if __nvvmGetProgramLogSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmGetProgramLogSize is not found")
    return (<nvvmResult (*)(nvvmProgram, size_t*) nogil>__nvvmGetProgramLogSize)(
        prog, bufferSizeRet)


cdef nvvmResult _nvvmGetProgramLog(nvvmProgram prog, char* buffer) except* nogil:
    global __nvvmGetProgramLog
    _check_or_init_nvvm()
    if __nvvmGetProgramLog == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmGetProgramLog is not found")
    return (<nvvmResult (*)(nvvmProgram, char*) nogil>__nvvmGetProgramLog)(
        prog, buffer)
