# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated with version 12.6.1. Do not modify it directly.

from libc.stdint cimport intptr_t

from .utils cimport get_nvvm_dso_version_suffix

from .utils import FunctionNotFoundError, NotSupportedError

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
cdef void* __nvvmCompileProgram = NULL


cdef inline list get_site_packages():
    return [site.getusersitepackages()] + site.getsitepackages()


cdef load_library(const int driver_ver):
    handle = 0

    for suffix in get_nvvm_dso_version_suffix(driver_ver):
        if len(suffix) == 0:
            continue
        dll_name = f"nvvm_{suffix}0_0.dll"

        # First check if the DLL has been loaded by 3rd parties
        try:
            handle = win32api.GetModuleHandle(dll_name)
        except:
            pass
        else:
            break

        # Next, check if DLLs are installed via pip
        for sp in get_site_packages():
            mod_path = os.path.join(sp, "nvidia", "nvvm", "bin")
            if not os.path.isdir(mod_path):
                continue
            os.add_dll_directory(mod_path)
        try:
            handle = win32api.LoadLibraryEx(
                # Note: LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR needs an abs path...
                os.path.join(mod_path, dll_name),
                0, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR)
        except:
            pass
        else:
            break

        # Finally, try default search
        try:
            handle = win32api.LoadLibrary(dll_name)
        except:
            pass
        else:
            break
    else:
        raise RuntimeError('Failed to load nvvm')

    assert handle != 0
    return handle


cdef int _check_or_init_nvvm() except -1 nogil:
    global __py_nvvm_init
    if __py_nvvm_init:
        return 0

    cdef int err, driver_ver
    with gil:
        # Load driver to check version
        try:
            handle = win32api.LoadLibraryEx("nvcuda.dll", 0, LOAD_LIBRARY_SEARCH_SYSTEM32)
        except Exception as e:
            raise NotSupportedError(f'CUDA driver is not found ({e})')
        global __cuDriverGetVersion
        if __cuDriverGetVersion == NULL:
            __cuDriverGetVersion = <void*><intptr_t>win32api.GetProcAddress(handle, 'cuDriverGetVersion')
            if __cuDriverGetVersion == NULL:
                raise RuntimeError('something went wrong')
        err = (<int (*)(int*) nogil>__cuDriverGetVersion)(&driver_ver)
        if err != 0:
            raise RuntimeError('something went wrong')

        # Load library
        handle = load_library(driver_ver)

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

        global __nvvmCompileProgram
        try:
            __nvvmCompileProgram = <void*><intptr_t>win32api.GetProcAddress(handle, 'nvvmCompileProgram')
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

    global __nvvmCompileProgram
    data["__nvvmCompileProgram"] = <intptr_t>__nvvmCompileProgram

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


cdef nvvmResult _nvvmCompileProgram(nvvmProgram prog, int numOptions, const char** options) except* nogil:
    global __nvvmCompileProgram
    _check_or_init_nvvm()
    if __nvvmCompileProgram == NULL:
        with gil:
            raise FunctionNotFoundError("function nvvmCompileProgram is not found")
    return (<nvvmResult (*)(nvvmProgram, int, const char**) nogil>__nvvmCompileProgram)(
        prog, numOptions, options)
