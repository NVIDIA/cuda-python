# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.9.0 to 12.9.1. Do not modify it directly.

from libc.stdint cimport intptr_t, uintptr_t
import threading

from .utils import FunctionNotFoundError, NotSupportedError

from cuda.pathfinder import load_nvidia_dynamic_lib

import cython


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

cdef object __symbol_lock = threading.Lock()
cdef bint __py_cufile_init = False
cdef void* __cuDriverGetVersion = NULL

cdef void* __cuFileHandleRegister = NULL
cdef void* __cuFileHandleDeregister = NULL
cdef void* __cuFileBufRegister = NULL
cdef void* __cuFileBufDeregister = NULL
cdef void* __cuFileRead = NULL
cdef void* __cuFileWrite = NULL
cdef void* __cuFileDriverOpen = NULL
cdef void* __cuFileDriverClose_v2 = NULL
cdef void* __cuFileUseCount = NULL
cdef void* __cuFileDriverGetProperties = NULL
cdef void* __cuFileDriverSetPollMode = NULL
cdef void* __cuFileDriverSetMaxDirectIOSize = NULL
cdef void* __cuFileDriverSetMaxCacheSize = NULL
cdef void* __cuFileDriverSetMaxPinnedMemSize = NULL
cdef void* __cuFileBatchIOSetUp = NULL
cdef void* __cuFileBatchIOSubmit = NULL
cdef void* __cuFileBatchIOGetStatus = NULL
cdef void* __cuFileBatchIOCancel = NULL
cdef void* __cuFileBatchIODestroy = NULL
cdef void* __cuFileReadAsync = NULL
cdef void* __cuFileWriteAsync = NULL
cdef void* __cuFileStreamRegister = NULL
cdef void* __cuFileStreamDeregister = NULL
cdef void* __cuFileGetVersion = NULL
cdef void* __cuFileGetParameterSizeT = NULL
cdef void* __cuFileGetParameterBool = NULL
cdef void* __cuFileGetParameterString = NULL
cdef void* __cuFileSetParameterSizeT = NULL
cdef void* __cuFileSetParameterBool = NULL
cdef void* __cuFileSetParameterString = NULL
cdef void* __cuFileDriverClose = NULL


cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("cufile")._handle_uint
    return <void*>handle


cdef int __check_or_init_cufile() except -1 nogil:
    global __py_cufile_init

    cdef void* handle = NULL

    with gil, __symbol_lock:
        # Load function
        global __cuFileHandleRegister
        __cuFileHandleRegister = dlsym(RTLD_DEFAULT, 'cuFileHandleRegister')
        if __cuFileHandleRegister == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileHandleRegister = dlsym(handle, 'cuFileHandleRegister')

        global __cuFileHandleDeregister
        __cuFileHandleDeregister = dlsym(RTLD_DEFAULT, 'cuFileHandleDeregister')
        if __cuFileHandleDeregister == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileHandleDeregister = dlsym(handle, 'cuFileHandleDeregister')

        global __cuFileBufRegister
        __cuFileBufRegister = dlsym(RTLD_DEFAULT, 'cuFileBufRegister')
        if __cuFileBufRegister == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileBufRegister = dlsym(handle, 'cuFileBufRegister')

        global __cuFileBufDeregister
        __cuFileBufDeregister = dlsym(RTLD_DEFAULT, 'cuFileBufDeregister')
        if __cuFileBufDeregister == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileBufDeregister = dlsym(handle, 'cuFileBufDeregister')

        global __cuFileRead
        __cuFileRead = dlsym(RTLD_DEFAULT, 'cuFileRead')
        if __cuFileRead == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileRead = dlsym(handle, 'cuFileRead')

        global __cuFileWrite
        __cuFileWrite = dlsym(RTLD_DEFAULT, 'cuFileWrite')
        if __cuFileWrite == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileWrite = dlsym(handle, 'cuFileWrite')

        global __cuFileDriverOpen
        __cuFileDriverOpen = dlsym(RTLD_DEFAULT, 'cuFileDriverOpen')
        if __cuFileDriverOpen == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileDriverOpen = dlsym(handle, 'cuFileDriverOpen')

        global __cuFileDriverClose_v2
        __cuFileDriverClose_v2 = dlsym(RTLD_DEFAULT, 'cuFileDriverClose_v2')
        if __cuFileDriverClose_v2 == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileDriverClose_v2 = dlsym(handle, 'cuFileDriverClose_v2')

        global __cuFileUseCount
        __cuFileUseCount = dlsym(RTLD_DEFAULT, 'cuFileUseCount')
        if __cuFileUseCount == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileUseCount = dlsym(handle, 'cuFileUseCount')

        global __cuFileDriverGetProperties
        __cuFileDriverGetProperties = dlsym(RTLD_DEFAULT, 'cuFileDriverGetProperties')
        if __cuFileDriverGetProperties == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileDriverGetProperties = dlsym(handle, 'cuFileDriverGetProperties')

        global __cuFileDriverSetPollMode
        __cuFileDriverSetPollMode = dlsym(RTLD_DEFAULT, 'cuFileDriverSetPollMode')
        if __cuFileDriverSetPollMode == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileDriverSetPollMode = dlsym(handle, 'cuFileDriverSetPollMode')

        global __cuFileDriverSetMaxDirectIOSize
        __cuFileDriverSetMaxDirectIOSize = dlsym(RTLD_DEFAULT, 'cuFileDriverSetMaxDirectIOSize')
        if __cuFileDriverSetMaxDirectIOSize == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileDriverSetMaxDirectIOSize = dlsym(handle, 'cuFileDriverSetMaxDirectIOSize')

        global __cuFileDriverSetMaxCacheSize
        __cuFileDriverSetMaxCacheSize = dlsym(RTLD_DEFAULT, 'cuFileDriverSetMaxCacheSize')
        if __cuFileDriverSetMaxCacheSize == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileDriverSetMaxCacheSize = dlsym(handle, 'cuFileDriverSetMaxCacheSize')

        global __cuFileDriverSetMaxPinnedMemSize
        __cuFileDriverSetMaxPinnedMemSize = dlsym(RTLD_DEFAULT, 'cuFileDriverSetMaxPinnedMemSize')
        if __cuFileDriverSetMaxPinnedMemSize == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileDriverSetMaxPinnedMemSize = dlsym(handle, 'cuFileDriverSetMaxPinnedMemSize')

        global __cuFileBatchIOSetUp
        __cuFileBatchIOSetUp = dlsym(RTLD_DEFAULT, 'cuFileBatchIOSetUp')
        if __cuFileBatchIOSetUp == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileBatchIOSetUp = dlsym(handle, 'cuFileBatchIOSetUp')

        global __cuFileBatchIOSubmit
        __cuFileBatchIOSubmit = dlsym(RTLD_DEFAULT, 'cuFileBatchIOSubmit')
        if __cuFileBatchIOSubmit == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileBatchIOSubmit = dlsym(handle, 'cuFileBatchIOSubmit')

        global __cuFileBatchIOGetStatus
        __cuFileBatchIOGetStatus = dlsym(RTLD_DEFAULT, 'cuFileBatchIOGetStatus')
        if __cuFileBatchIOGetStatus == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileBatchIOGetStatus = dlsym(handle, 'cuFileBatchIOGetStatus')

        global __cuFileBatchIOCancel
        __cuFileBatchIOCancel = dlsym(RTLD_DEFAULT, 'cuFileBatchIOCancel')
        if __cuFileBatchIOCancel == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileBatchIOCancel = dlsym(handle, 'cuFileBatchIOCancel')

        global __cuFileBatchIODestroy
        __cuFileBatchIODestroy = dlsym(RTLD_DEFAULT, 'cuFileBatchIODestroy')
        if __cuFileBatchIODestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileBatchIODestroy = dlsym(handle, 'cuFileBatchIODestroy')

        global __cuFileReadAsync
        __cuFileReadAsync = dlsym(RTLD_DEFAULT, 'cuFileReadAsync')
        if __cuFileReadAsync == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileReadAsync = dlsym(handle, 'cuFileReadAsync')

        global __cuFileWriteAsync
        __cuFileWriteAsync = dlsym(RTLD_DEFAULT, 'cuFileWriteAsync')
        if __cuFileWriteAsync == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileWriteAsync = dlsym(handle, 'cuFileWriteAsync')

        global __cuFileStreamRegister
        __cuFileStreamRegister = dlsym(RTLD_DEFAULT, 'cuFileStreamRegister')
        if __cuFileStreamRegister == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileStreamRegister = dlsym(handle, 'cuFileStreamRegister')

        global __cuFileStreamDeregister
        __cuFileStreamDeregister = dlsym(RTLD_DEFAULT, 'cuFileStreamDeregister')
        if __cuFileStreamDeregister == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileStreamDeregister = dlsym(handle, 'cuFileStreamDeregister')

        global __cuFileGetVersion
        __cuFileGetVersion = dlsym(RTLD_DEFAULT, 'cuFileGetVersion')
        if __cuFileGetVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileGetVersion = dlsym(handle, 'cuFileGetVersion')

        global __cuFileGetParameterSizeT
        __cuFileGetParameterSizeT = dlsym(RTLD_DEFAULT, 'cuFileGetParameterSizeT')
        if __cuFileGetParameterSizeT == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileGetParameterSizeT = dlsym(handle, 'cuFileGetParameterSizeT')

        global __cuFileGetParameterBool
        __cuFileGetParameterBool = dlsym(RTLD_DEFAULT, 'cuFileGetParameterBool')
        if __cuFileGetParameterBool == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileGetParameterBool = dlsym(handle, 'cuFileGetParameterBool')

        global __cuFileGetParameterString
        __cuFileGetParameterString = dlsym(RTLD_DEFAULT, 'cuFileGetParameterString')
        if __cuFileGetParameterString == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileGetParameterString = dlsym(handle, 'cuFileGetParameterString')

        global __cuFileSetParameterSizeT
        __cuFileSetParameterSizeT = dlsym(RTLD_DEFAULT, 'cuFileSetParameterSizeT')
        if __cuFileSetParameterSizeT == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileSetParameterSizeT = dlsym(handle, 'cuFileSetParameterSizeT')

        global __cuFileSetParameterBool
        __cuFileSetParameterBool = dlsym(RTLD_DEFAULT, 'cuFileSetParameterBool')
        if __cuFileSetParameterBool == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileSetParameterBool = dlsym(handle, 'cuFileSetParameterBool')

        global __cuFileSetParameterString
        __cuFileSetParameterString = dlsym(RTLD_DEFAULT, 'cuFileSetParameterString')
        if __cuFileSetParameterString == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileSetParameterString = dlsym(handle, 'cuFileSetParameterString')

        global __cuFileDriverClose
        __cuFileDriverClose = dlsym(RTLD_DEFAULT, 'cuFileDriverClose')
        if __cuFileDriverClose == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileDriverClose = dlsym(handle, 'cuFileDriverClose')

        __py_cufile_init = True
        return 0


cdef inline int _check_or_init_cufile() except -1 nogil:
    if __py_cufile_init:
        return 0

    return __check_or_init_cufile()


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_cufile()
    cdef dict data = {}

    global __cuFileHandleRegister
    data["__cuFileHandleRegister"] = <intptr_t>__cuFileHandleRegister

    global __cuFileHandleDeregister
    data["__cuFileHandleDeregister"] = <intptr_t>__cuFileHandleDeregister

    global __cuFileBufRegister
    data["__cuFileBufRegister"] = <intptr_t>__cuFileBufRegister

    global __cuFileBufDeregister
    data["__cuFileBufDeregister"] = <intptr_t>__cuFileBufDeregister

    global __cuFileRead
    data["__cuFileRead"] = <intptr_t>__cuFileRead

    global __cuFileWrite
    data["__cuFileWrite"] = <intptr_t>__cuFileWrite

    global __cuFileDriverOpen
    data["__cuFileDriverOpen"] = <intptr_t>__cuFileDriverOpen

    global __cuFileDriverClose_v2
    data["__cuFileDriverClose_v2"] = <intptr_t>__cuFileDriverClose_v2

    global __cuFileUseCount
    data["__cuFileUseCount"] = <intptr_t>__cuFileUseCount

    global __cuFileDriverGetProperties
    data["__cuFileDriverGetProperties"] = <intptr_t>__cuFileDriverGetProperties

    global __cuFileDriverSetPollMode
    data["__cuFileDriverSetPollMode"] = <intptr_t>__cuFileDriverSetPollMode

    global __cuFileDriverSetMaxDirectIOSize
    data["__cuFileDriverSetMaxDirectIOSize"] = <intptr_t>__cuFileDriverSetMaxDirectIOSize

    global __cuFileDriverSetMaxCacheSize
    data["__cuFileDriverSetMaxCacheSize"] = <intptr_t>__cuFileDriverSetMaxCacheSize

    global __cuFileDriverSetMaxPinnedMemSize
    data["__cuFileDriverSetMaxPinnedMemSize"] = <intptr_t>__cuFileDriverSetMaxPinnedMemSize

    global __cuFileBatchIOSetUp
    data["__cuFileBatchIOSetUp"] = <intptr_t>__cuFileBatchIOSetUp

    global __cuFileBatchIOSubmit
    data["__cuFileBatchIOSubmit"] = <intptr_t>__cuFileBatchIOSubmit

    global __cuFileBatchIOGetStatus
    data["__cuFileBatchIOGetStatus"] = <intptr_t>__cuFileBatchIOGetStatus

    global __cuFileBatchIOCancel
    data["__cuFileBatchIOCancel"] = <intptr_t>__cuFileBatchIOCancel

    global __cuFileBatchIODestroy
    data["__cuFileBatchIODestroy"] = <intptr_t>__cuFileBatchIODestroy

    global __cuFileReadAsync
    data["__cuFileReadAsync"] = <intptr_t>__cuFileReadAsync

    global __cuFileWriteAsync
    data["__cuFileWriteAsync"] = <intptr_t>__cuFileWriteAsync

    global __cuFileStreamRegister
    data["__cuFileStreamRegister"] = <intptr_t>__cuFileStreamRegister

    global __cuFileStreamDeregister
    data["__cuFileStreamDeregister"] = <intptr_t>__cuFileStreamDeregister

    global __cuFileGetVersion
    data["__cuFileGetVersion"] = <intptr_t>__cuFileGetVersion

    global __cuFileGetParameterSizeT
    data["__cuFileGetParameterSizeT"] = <intptr_t>__cuFileGetParameterSizeT

    global __cuFileGetParameterBool
    data["__cuFileGetParameterBool"] = <intptr_t>__cuFileGetParameterBool

    global __cuFileGetParameterString
    data["__cuFileGetParameterString"] = <intptr_t>__cuFileGetParameterString

    global __cuFileSetParameterSizeT
    data["__cuFileSetParameterSizeT"] = <intptr_t>__cuFileSetParameterSizeT

    global __cuFileSetParameterBool
    data["__cuFileSetParameterBool"] = <intptr_t>__cuFileSetParameterBool

    global __cuFileSetParameterString
    data["__cuFileSetParameterString"] = <intptr_t>__cuFileSetParameterString

    global __cuFileDriverClose
    data["__cuFileDriverClose"] = <intptr_t>__cuFileDriverClose

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

cdef CUfileError_t _cuFileHandleRegister(CUfileHandle_t* fh, CUfileDescr_t* descr) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileHandleRegister
    _check_or_init_cufile()
    if __cuFileHandleRegister == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileHandleRegister is not found")
    return (<CUfileError_t (*)(CUfileHandle_t*, CUfileDescr_t*) noexcept nogil>__cuFileHandleRegister)(
        fh, descr)


@cython.show_performance_hints(False)
cdef void _cuFileHandleDeregister(CUfileHandle_t fh) except* nogil:
    global __cuFileHandleDeregister
    _check_or_init_cufile()
    if __cuFileHandleDeregister == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileHandleDeregister is not found")
    (<void (*)(CUfileHandle_t) noexcept nogil>__cuFileHandleDeregister)(
        fh)


cdef CUfileError_t _cuFileBufRegister(const void* bufPtr_base, size_t length, int flags) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileBufRegister
    _check_or_init_cufile()
    if __cuFileBufRegister == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileBufRegister is not found")
    return (<CUfileError_t (*)(const void*, size_t, int) noexcept nogil>__cuFileBufRegister)(
        bufPtr_base, length, flags)


cdef CUfileError_t _cuFileBufDeregister(const void* bufPtr_base) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileBufDeregister
    _check_or_init_cufile()
    if __cuFileBufDeregister == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileBufDeregister is not found")
    return (<CUfileError_t (*)(const void*) noexcept nogil>__cuFileBufDeregister)(
        bufPtr_base)


cdef ssize_t _cuFileRead(CUfileHandle_t fh, void* bufPtr_base, size_t size, off_t file_offset, off_t bufPtr_offset) except* nogil:
    global __cuFileRead
    _check_or_init_cufile()
    if __cuFileRead == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileRead is not found")
    return (<ssize_t (*)(CUfileHandle_t, void*, size_t, off_t, off_t) noexcept nogil>__cuFileRead)(
        fh, bufPtr_base, size, file_offset, bufPtr_offset)


cdef ssize_t _cuFileWrite(CUfileHandle_t fh, const void* bufPtr_base, size_t size, off_t file_offset, off_t bufPtr_offset) except* nogil:
    global __cuFileWrite
    _check_or_init_cufile()
    if __cuFileWrite == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileWrite is not found")
    return (<ssize_t (*)(CUfileHandle_t, const void*, size_t, off_t, off_t) noexcept nogil>__cuFileWrite)(
        fh, bufPtr_base, size, file_offset, bufPtr_offset)


cdef CUfileError_t _cuFileDriverOpen() except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileDriverOpen
    _check_or_init_cufile()
    if __cuFileDriverOpen == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileDriverOpen is not found")
    return (<CUfileError_t (*)() noexcept nogil>__cuFileDriverOpen)(
        )


cdef CUfileError_t _cuFileDriverClose_v2() except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileDriverClose_v2
    _check_or_init_cufile()
    if __cuFileDriverClose_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileDriverClose_v2 is not found")
    return (<CUfileError_t (*)() noexcept nogil>__cuFileDriverClose_v2)(
        )


cdef long _cuFileUseCount() except* nogil:
    global __cuFileUseCount
    _check_or_init_cufile()
    if __cuFileUseCount == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileUseCount is not found")
    return (<long (*)() noexcept nogil>__cuFileUseCount)(
        )


cdef CUfileError_t _cuFileDriverGetProperties(CUfileDrvProps_t* props) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileDriverGetProperties
    _check_or_init_cufile()
    if __cuFileDriverGetProperties == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileDriverGetProperties is not found")
    return (<CUfileError_t (*)(CUfileDrvProps_t*) noexcept nogil>__cuFileDriverGetProperties)(
        props)


cdef CUfileError_t _cuFileDriverSetPollMode(cpp_bool poll, size_t poll_threshold_size) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileDriverSetPollMode
    _check_or_init_cufile()
    if __cuFileDriverSetPollMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileDriverSetPollMode is not found")
    return (<CUfileError_t (*)(cpp_bool, size_t) noexcept nogil>__cuFileDriverSetPollMode)(
        poll, poll_threshold_size)


cdef CUfileError_t _cuFileDriverSetMaxDirectIOSize(size_t max_direct_io_size) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileDriverSetMaxDirectIOSize
    _check_or_init_cufile()
    if __cuFileDriverSetMaxDirectIOSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileDriverSetMaxDirectIOSize is not found")
    return (<CUfileError_t (*)(size_t) noexcept nogil>__cuFileDriverSetMaxDirectIOSize)(
        max_direct_io_size)


cdef CUfileError_t _cuFileDriverSetMaxCacheSize(size_t max_cache_size) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileDriverSetMaxCacheSize
    _check_or_init_cufile()
    if __cuFileDriverSetMaxCacheSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileDriverSetMaxCacheSize is not found")
    return (<CUfileError_t (*)(size_t) noexcept nogil>__cuFileDriverSetMaxCacheSize)(
        max_cache_size)


cdef CUfileError_t _cuFileDriverSetMaxPinnedMemSize(size_t max_pinned_size) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileDriverSetMaxPinnedMemSize
    _check_or_init_cufile()
    if __cuFileDriverSetMaxPinnedMemSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileDriverSetMaxPinnedMemSize is not found")
    return (<CUfileError_t (*)(size_t) noexcept nogil>__cuFileDriverSetMaxPinnedMemSize)(
        max_pinned_size)


cdef CUfileError_t _cuFileBatchIOSetUp(CUfileBatchHandle_t* batch_idp, unsigned nr) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileBatchIOSetUp
    _check_or_init_cufile()
    if __cuFileBatchIOSetUp == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileBatchIOSetUp is not found")
    return (<CUfileError_t (*)(CUfileBatchHandle_t*, unsigned) noexcept nogil>__cuFileBatchIOSetUp)(
        batch_idp, nr)


cdef CUfileError_t _cuFileBatchIOSubmit(CUfileBatchHandle_t batch_idp, unsigned nr, CUfileIOParams_t* iocbp, unsigned int flags) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileBatchIOSubmit
    _check_or_init_cufile()
    if __cuFileBatchIOSubmit == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileBatchIOSubmit is not found")
    return (<CUfileError_t (*)(CUfileBatchHandle_t, unsigned, CUfileIOParams_t*, unsigned int) noexcept nogil>__cuFileBatchIOSubmit)(
        batch_idp, nr, iocbp, flags)


cdef CUfileError_t _cuFileBatchIOGetStatus(CUfileBatchHandle_t batch_idp, unsigned min_nr, unsigned* nr, CUfileIOEvents_t* iocbp, timespec* timeout) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileBatchIOGetStatus
    _check_or_init_cufile()
    if __cuFileBatchIOGetStatus == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileBatchIOGetStatus is not found")
    return (<CUfileError_t (*)(CUfileBatchHandle_t, unsigned, unsigned*, CUfileIOEvents_t*, timespec*) noexcept nogil>__cuFileBatchIOGetStatus)(
        batch_idp, min_nr, nr, iocbp, timeout)


cdef CUfileError_t _cuFileBatchIOCancel(CUfileBatchHandle_t batch_idp) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileBatchIOCancel
    _check_or_init_cufile()
    if __cuFileBatchIOCancel == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileBatchIOCancel is not found")
    return (<CUfileError_t (*)(CUfileBatchHandle_t) noexcept nogil>__cuFileBatchIOCancel)(
        batch_idp)


@cython.show_performance_hints(False)
cdef void _cuFileBatchIODestroy(CUfileBatchHandle_t batch_idp) except* nogil:
    global __cuFileBatchIODestroy
    _check_or_init_cufile()
    if __cuFileBatchIODestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileBatchIODestroy is not found")
    (<void (*)(CUfileBatchHandle_t) noexcept nogil>__cuFileBatchIODestroy)(
        batch_idp)


cdef CUfileError_t _cuFileReadAsync(CUfileHandle_t fh, void* bufPtr_base, size_t* size_p, off_t* file_offset_p, off_t* bufPtr_offset_p, ssize_t* bytes_read_p, CUstream stream) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileReadAsync
    _check_or_init_cufile()
    if __cuFileReadAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileReadAsync is not found")
    return (<CUfileError_t (*)(CUfileHandle_t, void*, size_t*, off_t*, off_t*, ssize_t*, CUstream) noexcept nogil>__cuFileReadAsync)(
        fh, bufPtr_base, size_p, file_offset_p, bufPtr_offset_p, bytes_read_p, stream)


cdef CUfileError_t _cuFileWriteAsync(CUfileHandle_t fh, void* bufPtr_base, size_t* size_p, off_t* file_offset_p, off_t* bufPtr_offset_p, ssize_t* bytes_written_p, CUstream stream) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileWriteAsync
    _check_or_init_cufile()
    if __cuFileWriteAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileWriteAsync is not found")
    return (<CUfileError_t (*)(CUfileHandle_t, void*, size_t*, off_t*, off_t*, ssize_t*, CUstream) noexcept nogil>__cuFileWriteAsync)(
        fh, bufPtr_base, size_p, file_offset_p, bufPtr_offset_p, bytes_written_p, stream)


cdef CUfileError_t _cuFileStreamRegister(CUstream stream, unsigned flags) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileStreamRegister
    _check_or_init_cufile()
    if __cuFileStreamRegister == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileStreamRegister is not found")
    return (<CUfileError_t (*)(CUstream, unsigned) noexcept nogil>__cuFileStreamRegister)(
        stream, flags)


cdef CUfileError_t _cuFileStreamDeregister(CUstream stream) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileStreamDeregister
    _check_or_init_cufile()
    if __cuFileStreamDeregister == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileStreamDeregister is not found")
    return (<CUfileError_t (*)(CUstream) noexcept nogil>__cuFileStreamDeregister)(
        stream)


cdef CUfileError_t _cuFileGetVersion(int* version) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileGetVersion
    _check_or_init_cufile()
    if __cuFileGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileGetVersion is not found")
    return (<CUfileError_t (*)(int*) noexcept nogil>__cuFileGetVersion)(
        version)


cdef CUfileError_t _cuFileGetParameterSizeT(CUFileSizeTConfigParameter_t param, size_t* value) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileGetParameterSizeT
    _check_or_init_cufile()
    if __cuFileGetParameterSizeT == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileGetParameterSizeT is not found")
    return (<CUfileError_t (*)(CUFileSizeTConfigParameter_t, size_t*) noexcept nogil>__cuFileGetParameterSizeT)(
        param, value)


cdef CUfileError_t _cuFileGetParameterBool(CUFileBoolConfigParameter_t param, cpp_bool* value) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileGetParameterBool
    _check_or_init_cufile()
    if __cuFileGetParameterBool == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileGetParameterBool is not found")
    return (<CUfileError_t (*)(CUFileBoolConfigParameter_t, cpp_bool*) noexcept nogil>__cuFileGetParameterBool)(
        param, value)


cdef CUfileError_t _cuFileGetParameterString(CUFileStringConfigParameter_t param, char* desc_str, int len) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileGetParameterString
    _check_or_init_cufile()
    if __cuFileGetParameterString == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileGetParameterString is not found")
    return (<CUfileError_t (*)(CUFileStringConfigParameter_t, char*, int) noexcept nogil>__cuFileGetParameterString)(
        param, desc_str, len)


cdef CUfileError_t _cuFileSetParameterSizeT(CUFileSizeTConfigParameter_t param, size_t value) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileSetParameterSizeT
    _check_or_init_cufile()
    if __cuFileSetParameterSizeT == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileSetParameterSizeT is not found")
    return (<CUfileError_t (*)(CUFileSizeTConfigParameter_t, size_t) noexcept nogil>__cuFileSetParameterSizeT)(
        param, value)


cdef CUfileError_t _cuFileSetParameterBool(CUFileBoolConfigParameter_t param, cpp_bool value) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileSetParameterBool
    _check_or_init_cufile()
    if __cuFileSetParameterBool == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileSetParameterBool is not found")
    return (<CUfileError_t (*)(CUFileBoolConfigParameter_t, cpp_bool) noexcept nogil>__cuFileSetParameterBool)(
        param, value)


cdef CUfileError_t _cuFileSetParameterString(CUFileStringConfigParameter_t param, const char* desc_str) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileSetParameterString
    _check_or_init_cufile()
    if __cuFileSetParameterString == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileSetParameterString is not found")
    return (<CUfileError_t (*)(CUFileStringConfigParameter_t, const char*) noexcept nogil>__cuFileSetParameterString)(
        param, desc_str)


cdef CUfileError_t _cuFileDriverClose() except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileDriverClose
    _check_or_init_cufile()
    if __cuFileDriverClose == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileDriverClose is not found")
    return (<CUfileError_t (*)() noexcept nogil>__cuFileDriverClose)(
        )
