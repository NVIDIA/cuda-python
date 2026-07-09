# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.0.1 to 13.3.0. Do not modify it directly.
# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=2a8907b5ab8df8ecb19b6d0fd3014dea67a9e76e7a2a0ee81fdf23f449402dd6


# <<<< PREAMBLE CONTENT >>>>

cdef extern from "<dlfcn.h>":
    void* _cyb_dlsym "dlsym"(void*, const char*) nogil
    const void * _cyb_RTLD_DEFAULT "RTLD_DEFAULT"

from libc.stdint cimport intptr_t as _cyb_intptr_t

import threading as _cyb_threading

cdef bint _cyb___py_nvjitlink_init = False
cdef dict _cyb_func_ptrs = None
cdef object _cyb_symbol_lock = _cyb_threading.Lock()

# <<<< END OF PREAMBLE CONTENT >>>>

from libc.stdint cimport uintptr_t

from .utils import FunctionNotFoundError, NotSupportedError
from cuda.pathfinder import load_nvidia_dynamic_lib


###############################################################################
# Wrapper init
###############################################################################

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
cdef void* __nvJitLinkGetLinkedLTOIRSize = NULL
cdef void* __nvJitLinkGetLinkedLTOIR = NULL

cdef int _init_nvjitlink() except -1 nogil:
    global _cyb___py_nvjitlink_init
    cdef void* handle = NULL
    with gil, _cyb_symbol_lock:
        if _cyb___py_nvjitlink_init: return 0

        global __nvJitLinkCreate
        __nvJitLinkCreate = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvJitLinkCreate')
        if __nvJitLinkCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __nvJitLinkCreate = _cyb_dlsym(handle, 'nvJitLinkCreate')

        global __nvJitLinkDestroy
        __nvJitLinkDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvJitLinkDestroy')
        if __nvJitLinkDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __nvJitLinkDestroy = _cyb_dlsym(handle, 'nvJitLinkDestroy')

        global __nvJitLinkAddData
        __nvJitLinkAddData = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvJitLinkAddData')
        if __nvJitLinkAddData == NULL:
            if handle == NULL:
                handle = load_library()
            __nvJitLinkAddData = _cyb_dlsym(handle, 'nvJitLinkAddData')

        global __nvJitLinkAddFile
        __nvJitLinkAddFile = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvJitLinkAddFile')
        if __nvJitLinkAddFile == NULL:
            if handle == NULL:
                handle = load_library()
            __nvJitLinkAddFile = _cyb_dlsym(handle, 'nvJitLinkAddFile')

        global __nvJitLinkComplete
        __nvJitLinkComplete = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvJitLinkComplete')
        if __nvJitLinkComplete == NULL:
            if handle == NULL:
                handle = load_library()
            __nvJitLinkComplete = _cyb_dlsym(handle, 'nvJitLinkComplete')

        global __nvJitLinkGetLinkedCubinSize
        __nvJitLinkGetLinkedCubinSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvJitLinkGetLinkedCubinSize')
        if __nvJitLinkGetLinkedCubinSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvJitLinkGetLinkedCubinSize = _cyb_dlsym(handle, 'nvJitLinkGetLinkedCubinSize')

        global __nvJitLinkGetLinkedCubin
        __nvJitLinkGetLinkedCubin = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvJitLinkGetLinkedCubin')
        if __nvJitLinkGetLinkedCubin == NULL:
            if handle == NULL:
                handle = load_library()
            __nvJitLinkGetLinkedCubin = _cyb_dlsym(handle, 'nvJitLinkGetLinkedCubin')

        global __nvJitLinkGetLinkedPtxSize
        __nvJitLinkGetLinkedPtxSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvJitLinkGetLinkedPtxSize')
        if __nvJitLinkGetLinkedPtxSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvJitLinkGetLinkedPtxSize = _cyb_dlsym(handle, 'nvJitLinkGetLinkedPtxSize')

        global __nvJitLinkGetLinkedPtx
        __nvJitLinkGetLinkedPtx = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvJitLinkGetLinkedPtx')
        if __nvJitLinkGetLinkedPtx == NULL:
            if handle == NULL:
                handle = load_library()
            __nvJitLinkGetLinkedPtx = _cyb_dlsym(handle, 'nvJitLinkGetLinkedPtx')

        global __nvJitLinkGetErrorLogSize
        __nvJitLinkGetErrorLogSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvJitLinkGetErrorLogSize')
        if __nvJitLinkGetErrorLogSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvJitLinkGetErrorLogSize = _cyb_dlsym(handle, 'nvJitLinkGetErrorLogSize')

        global __nvJitLinkGetErrorLog
        __nvJitLinkGetErrorLog = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvJitLinkGetErrorLog')
        if __nvJitLinkGetErrorLog == NULL:
            if handle == NULL:
                handle = load_library()
            __nvJitLinkGetErrorLog = _cyb_dlsym(handle, 'nvJitLinkGetErrorLog')

        global __nvJitLinkGetInfoLogSize
        __nvJitLinkGetInfoLogSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvJitLinkGetInfoLogSize')
        if __nvJitLinkGetInfoLogSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvJitLinkGetInfoLogSize = _cyb_dlsym(handle, 'nvJitLinkGetInfoLogSize')

        global __nvJitLinkGetInfoLog
        __nvJitLinkGetInfoLog = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvJitLinkGetInfoLog')
        if __nvJitLinkGetInfoLog == NULL:
            if handle == NULL:
                handle = load_library()
            __nvJitLinkGetInfoLog = _cyb_dlsym(handle, 'nvJitLinkGetInfoLog')

        global __nvJitLinkVersion
        __nvJitLinkVersion = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvJitLinkVersion')
        if __nvJitLinkVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __nvJitLinkVersion = _cyb_dlsym(handle, 'nvJitLinkVersion')

        global __nvJitLinkGetLinkedLTOIRSize
        __nvJitLinkGetLinkedLTOIRSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvJitLinkGetLinkedLTOIRSize')
        if __nvJitLinkGetLinkedLTOIRSize == NULL:
            if handle == NULL:
                handle = load_library()
            __nvJitLinkGetLinkedLTOIRSize = _cyb_dlsym(handle, 'nvJitLinkGetLinkedLTOIRSize')

        global __nvJitLinkGetLinkedLTOIR
        __nvJitLinkGetLinkedLTOIR = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'nvJitLinkGetLinkedLTOIR')
        if __nvJitLinkGetLinkedLTOIR == NULL:
            if handle == NULL:
                handle = load_library()
            __nvJitLinkGetLinkedLTOIR = _cyb_dlsym(handle, 'nvJitLinkGetLinkedLTOIR')

        _cyb___py_nvjitlink_init = True
        return 0

cdef inline int _check_or_init_nvjitlink() except -1 nogil:
    if _cyb___py_nvjitlink_init:
        return 0

    return _init_nvjitlink()


cpdef dict _inspect_function_pointers():
    global _cyb_func_ptrs
    if _cyb_func_ptrs is not None:
        return _cyb_func_ptrs

    _check_or_init_nvjitlink()
    cdef dict data = {}
    global __nvJitLinkCreate
    data["__nvJitLinkCreate"] = <_cyb_intptr_t>__nvJitLinkCreate

    global __nvJitLinkDestroy
    data["__nvJitLinkDestroy"] = <_cyb_intptr_t>__nvJitLinkDestroy

    global __nvJitLinkAddData
    data["__nvJitLinkAddData"] = <_cyb_intptr_t>__nvJitLinkAddData

    global __nvJitLinkAddFile
    data["__nvJitLinkAddFile"] = <_cyb_intptr_t>__nvJitLinkAddFile

    global __nvJitLinkComplete
    data["__nvJitLinkComplete"] = <_cyb_intptr_t>__nvJitLinkComplete

    global __nvJitLinkGetLinkedCubinSize
    data["__nvJitLinkGetLinkedCubinSize"] = <_cyb_intptr_t>__nvJitLinkGetLinkedCubinSize

    global __nvJitLinkGetLinkedCubin
    data["__nvJitLinkGetLinkedCubin"] = <_cyb_intptr_t>__nvJitLinkGetLinkedCubin

    global __nvJitLinkGetLinkedPtxSize
    data["__nvJitLinkGetLinkedPtxSize"] = <_cyb_intptr_t>__nvJitLinkGetLinkedPtxSize

    global __nvJitLinkGetLinkedPtx
    data["__nvJitLinkGetLinkedPtx"] = <_cyb_intptr_t>__nvJitLinkGetLinkedPtx

    global __nvJitLinkGetErrorLogSize
    data["__nvJitLinkGetErrorLogSize"] = <_cyb_intptr_t>__nvJitLinkGetErrorLogSize

    global __nvJitLinkGetErrorLog
    data["__nvJitLinkGetErrorLog"] = <_cyb_intptr_t>__nvJitLinkGetErrorLog

    global __nvJitLinkGetInfoLogSize
    data["__nvJitLinkGetInfoLogSize"] = <_cyb_intptr_t>__nvJitLinkGetInfoLogSize

    global __nvJitLinkGetInfoLog
    data["__nvJitLinkGetInfoLog"] = <_cyb_intptr_t>__nvJitLinkGetInfoLog

    global __nvJitLinkVersion
    data["__nvJitLinkVersion"] = <_cyb_intptr_t>__nvJitLinkVersion

    global __nvJitLinkGetLinkedLTOIRSize
    data["__nvJitLinkGetLinkedLTOIRSize"] = <_cyb_intptr_t>__nvJitLinkGetLinkedLTOIRSize

    global __nvJitLinkGetLinkedLTOIR
    data["__nvJitLinkGetLinkedLTOIR"] = <_cyb_intptr_t>__nvJitLinkGetLinkedLTOIR
    _cyb_func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global _cyb_func_ptrs
    if _cyb_func_ptrs is None:
        _cyb_func_ptrs = _inspect_function_pointers()
    return _cyb_func_ptrs[name]




cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("nvJitLink")._handle_uint
    return <void*>handle


###############################################################################
# Wrapper functions
###############################################################################

cdef nvJitLinkResult _nvJitLinkCreate(nvJitLinkHandle* handle, uint32_t numOptions, const char** options) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvJitLinkCreate
    _check_or_init_nvjitlink()
    if __nvJitLinkCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkCreate is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle*, uint32_t, const char**) noexcept nogil>__nvJitLinkCreate)(
        handle, numOptions, options)


cdef nvJitLinkResult _nvJitLinkDestroy(nvJitLinkHandle* handle) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvJitLinkDestroy
    _check_or_init_nvjitlink()
    if __nvJitLinkDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkDestroy is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle*) noexcept nogil>__nvJitLinkDestroy)(
        handle)


cdef nvJitLinkResult _nvJitLinkAddData(nvJitLinkHandle handle, nvJitLinkInputType inputType, const void* data, size_t size, const char* name) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvJitLinkAddData
    _check_or_init_nvjitlink()
    if __nvJitLinkAddData == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkAddData is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, nvJitLinkInputType, const void*, size_t, const char*) noexcept nogil>__nvJitLinkAddData)(
        handle, inputType, data, size, name)


cdef nvJitLinkResult _nvJitLinkAddFile(nvJitLinkHandle handle, nvJitLinkInputType inputType, const char* fileName) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvJitLinkAddFile
    _check_or_init_nvjitlink()
    if __nvJitLinkAddFile == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkAddFile is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, nvJitLinkInputType, const char*) noexcept nogil>__nvJitLinkAddFile)(
        handle, inputType, fileName)


cdef nvJitLinkResult _nvJitLinkComplete(nvJitLinkHandle handle) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvJitLinkComplete
    _check_or_init_nvjitlink()
    if __nvJitLinkComplete == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkComplete is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle) noexcept nogil>__nvJitLinkComplete)(
        handle)


cdef nvJitLinkResult _nvJitLinkGetLinkedCubinSize(nvJitLinkHandle handle, size_t* size) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvJitLinkGetLinkedCubinSize
    _check_or_init_nvjitlink()
    if __nvJitLinkGetLinkedCubinSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkGetLinkedCubinSize is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, size_t*) noexcept nogil>__nvJitLinkGetLinkedCubinSize)(
        handle, size)


cdef nvJitLinkResult _nvJitLinkGetLinkedCubin(nvJitLinkHandle handle, void* cubin) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvJitLinkGetLinkedCubin
    _check_or_init_nvjitlink()
    if __nvJitLinkGetLinkedCubin == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkGetLinkedCubin is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, void*) noexcept nogil>__nvJitLinkGetLinkedCubin)(
        handle, cubin)


cdef nvJitLinkResult _nvJitLinkGetLinkedPtxSize(nvJitLinkHandle handle, size_t* size) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvJitLinkGetLinkedPtxSize
    _check_or_init_nvjitlink()
    if __nvJitLinkGetLinkedPtxSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkGetLinkedPtxSize is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, size_t*) noexcept nogil>__nvJitLinkGetLinkedPtxSize)(
        handle, size)


cdef nvJitLinkResult _nvJitLinkGetLinkedPtx(nvJitLinkHandle handle, char* ptx) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvJitLinkGetLinkedPtx
    _check_or_init_nvjitlink()
    if __nvJitLinkGetLinkedPtx == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkGetLinkedPtx is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, char*) noexcept nogil>__nvJitLinkGetLinkedPtx)(
        handle, ptx)


cdef nvJitLinkResult _nvJitLinkGetErrorLogSize(nvJitLinkHandle handle, size_t* size) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvJitLinkGetErrorLogSize
    _check_or_init_nvjitlink()
    if __nvJitLinkGetErrorLogSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkGetErrorLogSize is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, size_t*) noexcept nogil>__nvJitLinkGetErrorLogSize)(
        handle, size)


cdef nvJitLinkResult _nvJitLinkGetErrorLog(nvJitLinkHandle handle, char* log) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvJitLinkGetErrorLog
    _check_or_init_nvjitlink()
    if __nvJitLinkGetErrorLog == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkGetErrorLog is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, char*) noexcept nogil>__nvJitLinkGetErrorLog)(
        handle, log)


cdef nvJitLinkResult _nvJitLinkGetInfoLogSize(nvJitLinkHandle handle, size_t* size) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvJitLinkGetInfoLogSize
    _check_or_init_nvjitlink()
    if __nvJitLinkGetInfoLogSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkGetInfoLogSize is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, size_t*) noexcept nogil>__nvJitLinkGetInfoLogSize)(
        handle, size)


cdef nvJitLinkResult _nvJitLinkGetInfoLog(nvJitLinkHandle handle, char* log) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvJitLinkGetInfoLog
    _check_or_init_nvjitlink()
    if __nvJitLinkGetInfoLog == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkGetInfoLog is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, char*) noexcept nogil>__nvJitLinkGetInfoLog)(
        handle, log)


cdef nvJitLinkResult _nvJitLinkVersion(unsigned int* major, unsigned int* minor) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvJitLinkVersion
    _check_or_init_nvjitlink()
    if __nvJitLinkVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkVersion is not found")
    return (<nvJitLinkResult (*)(unsigned int*, unsigned int*) noexcept nogil>__nvJitLinkVersion)(
        major, minor)


cdef nvJitLinkResult _nvJitLinkGetLinkedLTOIRSize(nvJitLinkHandle handle, size_t* size) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvJitLinkGetLinkedLTOIRSize
    _check_or_init_nvjitlink()
    if __nvJitLinkGetLinkedLTOIRSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkGetLinkedLTOIRSize is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, size_t*) noexcept nogil>__nvJitLinkGetLinkedLTOIRSize)(
        handle, size)


cdef nvJitLinkResult _nvJitLinkGetLinkedLTOIR(nvJitLinkHandle handle, void* ltoir) except?_NVJITLINKRESULT_INTERNAL_LOADING_ERROR nogil:
    global __nvJitLinkGetLinkedLTOIR
    _check_or_init_nvjitlink()
    if __nvJitLinkGetLinkedLTOIR == NULL:
        with gil:
            raise FunctionNotFoundError("function nvJitLinkGetLinkedLTOIR is not found")
    return (<nvJitLinkResult (*)(nvJitLinkHandle, void*) noexcept nogil>__nvJitLinkGetLinkedLTOIR)(
        handle, ltoir)
