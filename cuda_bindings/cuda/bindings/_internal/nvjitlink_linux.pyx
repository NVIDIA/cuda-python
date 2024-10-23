# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.0.1 to 12.6.2. Do not modify it directly.

from libc.stdint cimport intptr_t

from .utils cimport get_nvjitlink_dso_version_suffix

from .utils import FunctionNotFoundError, NotSupportedError

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


###############################################################################
# Wrapper init
###############################################################################

cdef bint __py_nvjitlink_init = False
cdef void* __cuDriverGetVersion = NULL

cdef void* __nvJitLinkCreate = NULL
cdef void* __nvJitLinkDestroy = NULL
cdef void* __nvJitLinkAddData = NULL
cdef void* __nvJitLinkAddFile = NULL
cdef void* __nvJitLinkComplete = NULL
cdef void* __nvJitLinkGetLinkedCubinSize = NULL
cdef void* __nvJitLinkGetLinkedCubin = NULL
cdef void* __nvJitLinkGetLinkedPtxSize = NULL
cdef void* __nvJitLinkGetLinkedPtx = NULL
cdef void* __nvJitLinkGetErrorLogSize = NULL
cdef void* __nvJitLinkGetErrorLog = NULL
cdef void* __nvJitLinkGetInfoLogSize = NULL
cdef void* __nvJitLinkGetInfoLog = NULL
cdef void* __nvJitLinkVersion = NULL


cdef void* load_library(const int driver_ver) except* with gil:
    cdef void* handle
    for suffix in get_nvjitlink_dso_version_suffix(driver_ver):
        so_name = "libnvJitLink.so" + (f".{suffix}" if suffix else suffix)
        handle = dlopen(so_name.encode(), RTLD_NOW | RTLD_GLOBAL)
        if handle != NULL:
            break
    else:
        err_msg = dlerror()
        raise RuntimeError(f'Failed to dlopen libnvJitLink ({err_msg.decode()})')
    return handle


cdef int _check_or_init_nvjitlink() except -1 nogil:
    global __py_nvjitlink_init
    if __py_nvjitlink_init:
        return 0

    # Load driver to check version
    cdef void* handle = NULL
    handle = dlopen('libcuda.so.1', RTLD_NOW | RTLD_GLOBAL)
    if handle == NULL:
        with gil:
            err_msg = dlerror()
            raise NotSupportedError(f'CUDA driver is not found ({err_msg.decode()})')
    global __cuDriverGetVersion
    if __cuDriverGetVersion == NULL:
        __cuDriverGetVersion = dlsym(handle, "cuDriverGetVersion")
    if __cuDriverGetVersion == NULL:
        with gil:
            raise RuntimeError('something went wrong')
    cdef int err, driver_ver
    err = (<int (*)(int*) nogil>__cuDriverGetVersion)(&driver_ver)
    if err != 0:
        with gil:
            raise RuntimeError('something went wrong')
    #dlclose(handle)
    handle = NULL

    # Load function
    global __nvJitLinkCreate
    __nvJitLinkCreate = dlsym(RTLD_DEFAULT, 'nvJitLinkCreate')
    if __nvJitLinkCreate == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __nvJitLinkCreate = dlsym(handle, 'nvJitLinkCreate')
    
    global __nvJitLinkDestroy
    __nvJitLinkDestroy = dlsym(RTLD_DEFAULT, 'nvJitLinkDestroy')
    if __nvJitLinkDestroy == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __nvJitLinkDestroy = dlsym(handle, 'nvJitLinkDestroy')
    
    global __nvJitLinkAddData
    __nvJitLinkAddData = dlsym(RTLD_DEFAULT, 'nvJitLinkAddData')
    if __nvJitLinkAddData == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __nvJitLinkAddData = dlsym(handle, 'nvJitLinkAddData')
    
    global __nvJitLinkAddFile
    __nvJitLinkAddFile = dlsym(RTLD_DEFAULT, 'nvJitLinkAddFile')
    if __nvJitLinkAddFile == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __nvJitLinkAddFile = dlsym(handle, 'nvJitLinkAddFile')
    
    global __nvJitLinkComplete
    __nvJitLinkComplete = dlsym(RTLD_DEFAULT, 'nvJitLinkComplete')
    if __nvJitLinkComplete == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __nvJitLinkComplete = dlsym(handle, 'nvJitLinkComplete')
    
    global __nvJitLinkGetLinkedCubinSize
    __nvJitLinkGetLinkedCubinSize = dlsym(RTLD_DEFAULT, 'nvJitLinkGetLinkedCubinSize')
    if __nvJitLinkGetLinkedCubinSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __nvJitLinkGetLinkedCubinSize = dlsym(handle, 'nvJitLinkGetLinkedCubinSize')
    
    global __nvJitLinkGetLinkedCubin
    __nvJitLinkGetLinkedCubin = dlsym(RTLD_DEFAULT, 'nvJitLinkGetLinkedCubin')
    if __nvJitLinkGetLinkedCubin == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __nvJitLinkGetLinkedCubin = dlsym(handle, 'nvJitLinkGetLinkedCubin')
    
    global __nvJitLinkGetLinkedPtxSize
    __nvJitLinkGetLinkedPtxSize = dlsym(RTLD_DEFAULT, 'nvJitLinkGetLinkedPtxSize')
    if __nvJitLinkGetLinkedPtxSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __nvJitLinkGetLinkedPtxSize = dlsym(handle, 'nvJitLinkGetLinkedPtxSize')
    
    global __nvJitLinkGetLinkedPtx
    __nvJitLinkGetLinkedPtx = dlsym(RTLD_DEFAULT, 'nvJitLinkGetLinkedPtx')
    if __nvJitLinkGetLinkedPtx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __nvJitLinkGetLinkedPtx = dlsym(handle, 'nvJitLinkGetLinkedPtx')
    
    global __nvJitLinkGetErrorLogSize
    __nvJitLinkGetErrorLogSize = dlsym(RTLD_DEFAULT, 'nvJitLinkGetErrorLogSize')
    if __nvJitLinkGetErrorLogSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __nvJitLinkGetErrorLogSize = dlsym(handle, 'nvJitLinkGetErrorLogSize')
    
    global __nvJitLinkGetErrorLog
    __nvJitLinkGetErrorLog = dlsym(RTLD_DEFAULT, 'nvJitLinkGetErrorLog')
    if __nvJitLinkGetErrorLog == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __nvJitLinkGetErrorLog = dlsym(handle, 'nvJitLinkGetErrorLog')
    
    global __nvJitLinkGetInfoLogSize
    __nvJitLinkGetInfoLogSize = dlsym(RTLD_DEFAULT, 'nvJitLinkGetInfoLogSize')
    if __nvJitLinkGetInfoLogSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __nvJitLinkGetInfoLogSize = dlsym(handle, 'nvJitLinkGetInfoLogSize')
    
    global __nvJitLinkGetInfoLog
    __nvJitLinkGetInfoLog = dlsym(RTLD_DEFAULT, 'nvJitLinkGetInfoLog')
    if __nvJitLinkGetInfoLog == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __nvJitLinkGetInfoLog = dlsym(handle, 'nvJitLinkGetInfoLog')
    
    global __nvJitLinkVersion
    __nvJitLinkVersion = dlsym(RTLD_DEFAULT, 'nvJitLinkVersion')
    if __nvJitLinkVersion == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __nvJitLinkVersion = dlsym(handle, 'nvJitLinkVersion')

    __py_nvjitlink_init = True
    return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_nvjitlink()
    cdef dict data = {}

    global __nvJitLinkCreate
    data["__nvJitLinkCreate"] = <intptr_t>__nvJitLinkCreate
    
    global __nvJitLinkDestroy
    data["__nvJitLinkDestroy"] = <intptr_t>__nvJitLinkDestroy
    
    global __nvJitLinkAddData
    data["__nvJitLinkAddData"] = <intptr_t>__nvJitLinkAddData
    
    global __nvJitLinkAddFile
    data["__nvJitLinkAddFile"] = <intptr_t>__nvJitLinkAddFile
    
    global __nvJitLinkComplete
    data["__nvJitLinkComplete"] = <intptr_t>__nvJitLinkComplete
    
    global __nvJitLinkGetLinkedCubinSize
    data["__nvJitLinkGetLinkedCubinSize"] = <intptr_t>__nvJitLinkGetLinkedCubinSize
    
    global __nvJitLinkGetLinkedCubin
    data["__nvJitLinkGetLinkedCubin"] = <intptr_t>__nvJitLinkGetLinkedCubin
    
    global __nvJitLinkGetLinkedPtxSize
    data["__nvJitLinkGetLinkedPtxSize"] = <intptr_t>__nvJitLinkGetLinkedPtxSize
    
    global __nvJitLinkGetLinkedPtx
    data["__nvJitLinkGetLinkedPtx"] = <intptr_t>__nvJitLinkGetLinkedPtx
    
    global __nvJitLinkGetErrorLogSize
    data["__nvJitLinkGetErrorLogSize"] = <intptr_t>__nvJitLinkGetErrorLogSize
    
    global __nvJitLinkGetErrorLog
    data["__nvJitLinkGetErrorLog"] = <intptr_t>__nvJitLinkGetErrorLog
    
    global __nvJitLinkGetInfoLogSize
    data["__nvJitLinkGetInfoLogSize"] = <intptr_t>__nvJitLinkGetInfoLogSize
    
    global __nvJitLinkGetInfoLog
    data["__nvJitLinkGetInfoLog"] = <intptr_t>__nvJitLinkGetInfoLog
    
    global __nvJitLinkVersion
    data["__nvJitLinkVersion"] = <intptr_t>__nvJitLinkVersion

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

cdef nvJitLinkResult _nvJitLinkCreate(nvJitLinkHandle* handle, uint32_t numOptions, const char** options) except* nogil:
    global __nvJitLinkCreate
    _check_or_init_nvjitlink()
    if __nvJitLinkCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkCreate is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle*, uint32_t, const char**) nogil>__nvJitLinkCreate)(
        handle, numOptions, options)


cdef nvJitLinkResult _nvJitLinkDestroy(nvJitLinkHandle* handle) except* nogil:
    global __nvJitLinkDestroy
    _check_or_init_nvjitlink()
    if __nvJitLinkDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkDestroy is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle*) nogil>__nvJitLinkDestroy)(
        handle)


cdef nvJitLinkResult _nvJitLinkAddData(nvJitLinkHandle handle, nvJitLinkInputType inputType, const void* data, size_t size, const char* name) except* nogil:
    global __nvJitLinkAddData
    _check_or_init_nvjitlink()
    if __nvJitLinkAddData == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkAddData is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, nvJitLinkInputType, const void*, size_t, const char*) nogil>__nvJitLinkAddData)(
        handle, inputType, data, size, name)


cdef nvJitLinkResult _nvJitLinkAddFile(nvJitLinkHandle handle, nvJitLinkInputType inputType, const char* fileName) except* nogil:
    global __nvJitLinkAddFile
    _check_or_init_nvjitlink()
    if __nvJitLinkAddFile == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkAddFile is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, nvJitLinkInputType, const char*) nogil>__nvJitLinkAddFile)(
        handle, inputType, fileName)


cdef nvJitLinkResult _nvJitLinkComplete(nvJitLinkHandle handle) except* nogil:
    global __nvJitLinkComplete
    _check_or_init_nvjitlink()
    if __nvJitLinkComplete == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkComplete is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle) nogil>__nvJitLinkComplete)(
        handle)


cdef nvJitLinkResult _nvJitLinkGetLinkedCubinSize(nvJitLinkHandle handle, size_t* size) except* nogil:
    global __nvJitLinkGetLinkedCubinSize
    _check_or_init_nvjitlink()
    if __nvJitLinkGetLinkedCubinSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkGetLinkedCubinSize is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, size_t*) nogil>__nvJitLinkGetLinkedCubinSize)(
        handle, size)


cdef nvJitLinkResult _nvJitLinkGetLinkedCubin(nvJitLinkHandle handle, void* cubin) except* nogil:
    global __nvJitLinkGetLinkedCubin
    _check_or_init_nvjitlink()
    if __nvJitLinkGetLinkedCubin == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkGetLinkedCubin is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, void*) nogil>__nvJitLinkGetLinkedCubin)(
        handle, cubin)


cdef nvJitLinkResult _nvJitLinkGetLinkedPtxSize(nvJitLinkHandle handle, size_t* size) except* nogil:
    global __nvJitLinkGetLinkedPtxSize
    _check_or_init_nvjitlink()
    if __nvJitLinkGetLinkedPtxSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkGetLinkedPtxSize is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, size_t*) nogil>__nvJitLinkGetLinkedPtxSize)(
        handle, size)


cdef nvJitLinkResult _nvJitLinkGetLinkedPtx(nvJitLinkHandle handle, char* ptx) except* nogil:
    global __nvJitLinkGetLinkedPtx
    _check_or_init_nvjitlink()
    if __nvJitLinkGetLinkedPtx == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkGetLinkedPtx is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, char*) nogil>__nvJitLinkGetLinkedPtx)(
        handle, ptx)


cdef nvJitLinkResult _nvJitLinkGetErrorLogSize(nvJitLinkHandle handle, size_t* size) except* nogil:
    global __nvJitLinkGetErrorLogSize
    _check_or_init_nvjitlink()
    if __nvJitLinkGetErrorLogSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkGetErrorLogSize is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, size_t*) nogil>__nvJitLinkGetErrorLogSize)(
        handle, size)


cdef nvJitLinkResult _nvJitLinkGetErrorLog(nvJitLinkHandle handle, char* log) except* nogil:
    global __nvJitLinkGetErrorLog
    _check_or_init_nvjitlink()
    if __nvJitLinkGetErrorLog == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkGetErrorLog is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, char*) nogil>__nvJitLinkGetErrorLog)(
        handle, log)


cdef nvJitLinkResult _nvJitLinkGetInfoLogSize(nvJitLinkHandle handle, size_t* size) except* nogil:
    global __nvJitLinkGetInfoLogSize
    _check_or_init_nvjitlink()
    if __nvJitLinkGetInfoLogSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkGetInfoLogSize is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, size_t*) nogil>__nvJitLinkGetInfoLogSize)(
        handle, size)


cdef nvJitLinkResult _nvJitLinkGetInfoLog(nvJitLinkHandle handle, char* log) except* nogil:
    global __nvJitLinkGetInfoLog
    _check_or_init_nvjitlink()
    if __nvJitLinkGetInfoLog == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkGetInfoLog is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, char*) nogil>__nvJitLinkGetInfoLog)(
        handle, log)


cdef nvJitLinkResult _nvJitLinkVersion(unsigned int* major, unsigned int* minor) except* nogil:
    global __nvJitLinkVersion
    _check_or_init_nvjitlink()
    if __nvJitLinkVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkVersion is not found")
    return (<nvJitLinkResult (*)(unsigned int*, unsigned int*) nogil>__nvJitLinkVersion)(
        major, minor)
