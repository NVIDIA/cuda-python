# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.9.1 to 13.3.0. Do not modify it directly.
# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=5294c8a99ae13f8d69ca1c3ab4cec4e855b8c99aee32d558f267b10dd04d8e00

# <<<< PREAMBLE CONTENT >>>>

cdef extern from "<dlfcn.h>":
    void* _cyb_dlsym "dlsym"(void*, const char*) nogil
    const void * _cyb_RTLD_DEFAULT "RTLD_DEFAULT"

cimport cython as _cyb_cython
from libc.stdint cimport intptr_t as _cyb_intptr_t

import threading as _cyb_threading

cdef bint _cyb___py_cufile_init = False
cdef dict _cyb_func_ptrs = None
cdef object _cyb_symbol_lock = _cyb_threading.Lock()

# <<<< END OF PREAMBLE CONTENT >>>>


from libc.stdint cimport uintptr_t

from .utils import FunctionNotFoundError, NotSupportedError
from cuda.pathfinder import load_nvidia_dynamic_lib


###############################################################################
# Wrapper init
###############################################################################


cdef void* __cuFileHandleRegister = NULL
cdef void* __cuFileHandleDeregister = NULL
cdef void* __cuFileBufRegister = NULL
cdef void* __cuFileBufDeregister = NULL
cdef void* __cuFileRead = NULL
cdef void* __cuFileWrite = NULL
cdef void* __cuFileDriverOpen = NULL
cdef void* __cuFileDriverClose = NULL
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
cdef void* __cuFileGetParameterMinMaxValue = NULL
cdef void* __cuFileSetStatsLevel = NULL
cdef void* __cuFileGetStatsLevel = NULL
cdef void* __cuFileStatsStart = NULL
cdef void* __cuFileStatsStop = NULL
cdef void* __cuFileStatsReset = NULL
cdef void* __cuFileGetStatsL1 = NULL
cdef void* __cuFileGetStatsL2 = NULL
cdef void* __cuFileGetStatsL3 = NULL
cdef void* __cuFileGetBARSizeInKB = NULL
cdef void* __cuFileSetParameterPosixPoolSlabArray = NULL
cdef void* __cuFileGetParameterPosixPoolSlabArray = NULL

cdef int _init_cufile() except -1 nogil:
    global _cyb___py_cufile_init
    cdef void* handle = NULL
    with gil, _cyb_symbol_lock:
        if _cyb___py_cufile_init: return 0

        global __cuFileHandleRegister
        __cuFileHandleRegister = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileHandleRegister')
        if __cuFileHandleRegister == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileHandleRegister = _cyb_dlsym(handle, 'cuFileHandleRegister')

        global __cuFileHandleDeregister
        __cuFileHandleDeregister = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileHandleDeregister')
        if __cuFileHandleDeregister == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileHandleDeregister = _cyb_dlsym(handle, 'cuFileHandleDeregister')

        global __cuFileBufRegister
        __cuFileBufRegister = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileBufRegister')
        if __cuFileBufRegister == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileBufRegister = _cyb_dlsym(handle, 'cuFileBufRegister')

        global __cuFileBufDeregister
        __cuFileBufDeregister = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileBufDeregister')
        if __cuFileBufDeregister == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileBufDeregister = _cyb_dlsym(handle, 'cuFileBufDeregister')

        global __cuFileRead
        __cuFileRead = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileRead')
        if __cuFileRead == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileRead = _cyb_dlsym(handle, 'cuFileRead')

        global __cuFileWrite
        __cuFileWrite = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileWrite')
        if __cuFileWrite == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileWrite = _cyb_dlsym(handle, 'cuFileWrite')

        global __cuFileDriverOpen
        __cuFileDriverOpen = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileDriverOpen')
        if __cuFileDriverOpen == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileDriverOpen = _cyb_dlsym(handle, 'cuFileDriverOpen')

        global __cuFileDriverClose
        __cuFileDriverClose = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileDriverClose')
        if __cuFileDriverClose == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileDriverClose = _cyb_dlsym(handle, 'cuFileDriverClose')

        global __cuFileDriverClose_v2
        __cuFileDriverClose_v2 = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileDriverClose_v2')
        if __cuFileDriverClose_v2 == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileDriverClose_v2 = _cyb_dlsym(handle, 'cuFileDriverClose_v2')

        global __cuFileUseCount
        __cuFileUseCount = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileUseCount')
        if __cuFileUseCount == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileUseCount = _cyb_dlsym(handle, 'cuFileUseCount')

        global __cuFileDriverGetProperties
        __cuFileDriverGetProperties = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileDriverGetProperties')
        if __cuFileDriverGetProperties == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileDriverGetProperties = _cyb_dlsym(handle, 'cuFileDriverGetProperties')

        global __cuFileDriverSetPollMode
        __cuFileDriverSetPollMode = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileDriverSetPollMode')
        if __cuFileDriverSetPollMode == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileDriverSetPollMode = _cyb_dlsym(handle, 'cuFileDriverSetPollMode')

        global __cuFileDriverSetMaxDirectIOSize
        __cuFileDriverSetMaxDirectIOSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileDriverSetMaxDirectIOSize')
        if __cuFileDriverSetMaxDirectIOSize == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileDriverSetMaxDirectIOSize = _cyb_dlsym(handle, 'cuFileDriverSetMaxDirectIOSize')

        global __cuFileDriverSetMaxCacheSize
        __cuFileDriverSetMaxCacheSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileDriverSetMaxCacheSize')
        if __cuFileDriverSetMaxCacheSize == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileDriverSetMaxCacheSize = _cyb_dlsym(handle, 'cuFileDriverSetMaxCacheSize')

        global __cuFileDriverSetMaxPinnedMemSize
        __cuFileDriverSetMaxPinnedMemSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileDriverSetMaxPinnedMemSize')
        if __cuFileDriverSetMaxPinnedMemSize == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileDriverSetMaxPinnedMemSize = _cyb_dlsym(handle, 'cuFileDriverSetMaxPinnedMemSize')

        global __cuFileBatchIOSetUp
        __cuFileBatchIOSetUp = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileBatchIOSetUp')
        if __cuFileBatchIOSetUp == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileBatchIOSetUp = _cyb_dlsym(handle, 'cuFileBatchIOSetUp')

        global __cuFileBatchIOSubmit
        __cuFileBatchIOSubmit = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileBatchIOSubmit')
        if __cuFileBatchIOSubmit == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileBatchIOSubmit = _cyb_dlsym(handle, 'cuFileBatchIOSubmit')

        global __cuFileBatchIOGetStatus
        __cuFileBatchIOGetStatus = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileBatchIOGetStatus')
        if __cuFileBatchIOGetStatus == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileBatchIOGetStatus = _cyb_dlsym(handle, 'cuFileBatchIOGetStatus')

        global __cuFileBatchIOCancel
        __cuFileBatchIOCancel = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileBatchIOCancel')
        if __cuFileBatchIOCancel == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileBatchIOCancel = _cyb_dlsym(handle, 'cuFileBatchIOCancel')

        global __cuFileBatchIODestroy
        __cuFileBatchIODestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileBatchIODestroy')
        if __cuFileBatchIODestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileBatchIODestroy = _cyb_dlsym(handle, 'cuFileBatchIODestroy')

        global __cuFileReadAsync
        __cuFileReadAsync = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileReadAsync')
        if __cuFileReadAsync == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileReadAsync = _cyb_dlsym(handle, 'cuFileReadAsync')

        global __cuFileWriteAsync
        __cuFileWriteAsync = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileWriteAsync')
        if __cuFileWriteAsync == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileWriteAsync = _cyb_dlsym(handle, 'cuFileWriteAsync')

        global __cuFileStreamRegister
        __cuFileStreamRegister = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileStreamRegister')
        if __cuFileStreamRegister == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileStreamRegister = _cyb_dlsym(handle, 'cuFileStreamRegister')

        global __cuFileStreamDeregister
        __cuFileStreamDeregister = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileStreamDeregister')
        if __cuFileStreamDeregister == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileStreamDeregister = _cyb_dlsym(handle, 'cuFileStreamDeregister')

        global __cuFileGetVersion
        __cuFileGetVersion = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileGetVersion')
        if __cuFileGetVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileGetVersion = _cyb_dlsym(handle, 'cuFileGetVersion')

        global __cuFileGetParameterSizeT
        __cuFileGetParameterSizeT = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileGetParameterSizeT')
        if __cuFileGetParameterSizeT == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileGetParameterSizeT = _cyb_dlsym(handle, 'cuFileGetParameterSizeT')

        global __cuFileGetParameterBool
        __cuFileGetParameterBool = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileGetParameterBool')
        if __cuFileGetParameterBool == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileGetParameterBool = _cyb_dlsym(handle, 'cuFileGetParameterBool')

        global __cuFileGetParameterString
        __cuFileGetParameterString = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileGetParameterString')
        if __cuFileGetParameterString == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileGetParameterString = _cyb_dlsym(handle, 'cuFileGetParameterString')

        global __cuFileSetParameterSizeT
        __cuFileSetParameterSizeT = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileSetParameterSizeT')
        if __cuFileSetParameterSizeT == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileSetParameterSizeT = _cyb_dlsym(handle, 'cuFileSetParameterSizeT')

        global __cuFileSetParameterBool
        __cuFileSetParameterBool = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileSetParameterBool')
        if __cuFileSetParameterBool == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileSetParameterBool = _cyb_dlsym(handle, 'cuFileSetParameterBool')

        global __cuFileSetParameterString
        __cuFileSetParameterString = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileSetParameterString')
        if __cuFileSetParameterString == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileSetParameterString = _cyb_dlsym(handle, 'cuFileSetParameterString')

        global __cuFileGetParameterMinMaxValue
        __cuFileGetParameterMinMaxValue = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileGetParameterMinMaxValue')
        if __cuFileGetParameterMinMaxValue == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileGetParameterMinMaxValue = _cyb_dlsym(handle, 'cuFileGetParameterMinMaxValue')

        global __cuFileSetStatsLevel
        __cuFileSetStatsLevel = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileSetStatsLevel')
        if __cuFileSetStatsLevel == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileSetStatsLevel = _cyb_dlsym(handle, 'cuFileSetStatsLevel')

        global __cuFileGetStatsLevel
        __cuFileGetStatsLevel = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileGetStatsLevel')
        if __cuFileGetStatsLevel == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileGetStatsLevel = _cyb_dlsym(handle, 'cuFileGetStatsLevel')

        global __cuFileStatsStart
        __cuFileStatsStart = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileStatsStart')
        if __cuFileStatsStart == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileStatsStart = _cyb_dlsym(handle, 'cuFileStatsStart')

        global __cuFileStatsStop
        __cuFileStatsStop = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileStatsStop')
        if __cuFileStatsStop == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileStatsStop = _cyb_dlsym(handle, 'cuFileStatsStop')

        global __cuFileStatsReset
        __cuFileStatsReset = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileStatsReset')
        if __cuFileStatsReset == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileStatsReset = _cyb_dlsym(handle, 'cuFileStatsReset')

        global __cuFileGetStatsL1
        __cuFileGetStatsL1 = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileGetStatsL1')
        if __cuFileGetStatsL1 == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileGetStatsL1 = _cyb_dlsym(handle, 'cuFileGetStatsL1')

        global __cuFileGetStatsL2
        __cuFileGetStatsL2 = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileGetStatsL2')
        if __cuFileGetStatsL2 == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileGetStatsL2 = _cyb_dlsym(handle, 'cuFileGetStatsL2')

        global __cuFileGetStatsL3
        __cuFileGetStatsL3 = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileGetStatsL3')
        if __cuFileGetStatsL3 == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileGetStatsL3 = _cyb_dlsym(handle, 'cuFileGetStatsL3')

        global __cuFileGetBARSizeInKB
        __cuFileGetBARSizeInKB = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileGetBARSizeInKB')
        if __cuFileGetBARSizeInKB == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileGetBARSizeInKB = _cyb_dlsym(handle, 'cuFileGetBARSizeInKB')

        global __cuFileSetParameterPosixPoolSlabArray
        __cuFileSetParameterPosixPoolSlabArray = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileSetParameterPosixPoolSlabArray')
        if __cuFileSetParameterPosixPoolSlabArray == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileSetParameterPosixPoolSlabArray = _cyb_dlsym(handle, 'cuFileSetParameterPosixPoolSlabArray')

        global __cuFileGetParameterPosixPoolSlabArray
        __cuFileGetParameterPosixPoolSlabArray = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cuFileGetParameterPosixPoolSlabArray')
        if __cuFileGetParameterPosixPoolSlabArray == NULL:
            if handle == NULL:
                handle = load_library()
            __cuFileGetParameterPosixPoolSlabArray = _cyb_dlsym(handle, 'cuFileGetParameterPosixPoolSlabArray')

        _cyb___py_cufile_init = True
        return 0

cdef inline int _check_or_init_cufile() except -1 nogil:
    if _cyb___py_cufile_init:
        return 0

    return _init_cufile()


cpdef dict _inspect_function_pointers():
    global _cyb_func_ptrs
    if _cyb_func_ptrs is not None:
        return _cyb_func_ptrs

    _check_or_init_cufile()
    cdef dict data = {}
    global __cuFileHandleRegister
    data["__cuFileHandleRegister"] = <_cyb_intptr_t>__cuFileHandleRegister

    global __cuFileHandleDeregister
    data["__cuFileHandleDeregister"] = <_cyb_intptr_t>__cuFileHandleDeregister

    global __cuFileBufRegister
    data["__cuFileBufRegister"] = <_cyb_intptr_t>__cuFileBufRegister

    global __cuFileBufDeregister
    data["__cuFileBufDeregister"] = <_cyb_intptr_t>__cuFileBufDeregister

    global __cuFileRead
    data["__cuFileRead"] = <_cyb_intptr_t>__cuFileRead

    global __cuFileWrite
    data["__cuFileWrite"] = <_cyb_intptr_t>__cuFileWrite

    global __cuFileDriverOpen
    data["__cuFileDriverOpen"] = <_cyb_intptr_t>__cuFileDriverOpen

    global __cuFileDriverClose
    data["__cuFileDriverClose"] = <_cyb_intptr_t>__cuFileDriverClose

    global __cuFileDriverClose_v2
    data["__cuFileDriverClose_v2"] = <_cyb_intptr_t>__cuFileDriverClose_v2

    global __cuFileUseCount
    data["__cuFileUseCount"] = <_cyb_intptr_t>__cuFileUseCount

    global __cuFileDriverGetProperties
    data["__cuFileDriverGetProperties"] = <_cyb_intptr_t>__cuFileDriverGetProperties

    global __cuFileDriverSetPollMode
    data["__cuFileDriverSetPollMode"] = <_cyb_intptr_t>__cuFileDriverSetPollMode

    global __cuFileDriverSetMaxDirectIOSize
    data["__cuFileDriverSetMaxDirectIOSize"] = <_cyb_intptr_t>__cuFileDriverSetMaxDirectIOSize

    global __cuFileDriverSetMaxCacheSize
    data["__cuFileDriverSetMaxCacheSize"] = <_cyb_intptr_t>__cuFileDriverSetMaxCacheSize

    global __cuFileDriverSetMaxPinnedMemSize
    data["__cuFileDriverSetMaxPinnedMemSize"] = <_cyb_intptr_t>__cuFileDriverSetMaxPinnedMemSize

    global __cuFileBatchIOSetUp
    data["__cuFileBatchIOSetUp"] = <_cyb_intptr_t>__cuFileBatchIOSetUp

    global __cuFileBatchIOSubmit
    data["__cuFileBatchIOSubmit"] = <_cyb_intptr_t>__cuFileBatchIOSubmit

    global __cuFileBatchIOGetStatus
    data["__cuFileBatchIOGetStatus"] = <_cyb_intptr_t>__cuFileBatchIOGetStatus

    global __cuFileBatchIOCancel
    data["__cuFileBatchIOCancel"] = <_cyb_intptr_t>__cuFileBatchIOCancel

    global __cuFileBatchIODestroy
    data["__cuFileBatchIODestroy"] = <_cyb_intptr_t>__cuFileBatchIODestroy

    global __cuFileReadAsync
    data["__cuFileReadAsync"] = <_cyb_intptr_t>__cuFileReadAsync

    global __cuFileWriteAsync
    data["__cuFileWriteAsync"] = <_cyb_intptr_t>__cuFileWriteAsync

    global __cuFileStreamRegister
    data["__cuFileStreamRegister"] = <_cyb_intptr_t>__cuFileStreamRegister

    global __cuFileStreamDeregister
    data["__cuFileStreamDeregister"] = <_cyb_intptr_t>__cuFileStreamDeregister

    global __cuFileGetVersion
    data["__cuFileGetVersion"] = <_cyb_intptr_t>__cuFileGetVersion

    global __cuFileGetParameterSizeT
    data["__cuFileGetParameterSizeT"] = <_cyb_intptr_t>__cuFileGetParameterSizeT

    global __cuFileGetParameterBool
    data["__cuFileGetParameterBool"] = <_cyb_intptr_t>__cuFileGetParameterBool

    global __cuFileGetParameterString
    data["__cuFileGetParameterString"] = <_cyb_intptr_t>__cuFileGetParameterString

    global __cuFileSetParameterSizeT
    data["__cuFileSetParameterSizeT"] = <_cyb_intptr_t>__cuFileSetParameterSizeT

    global __cuFileSetParameterBool
    data["__cuFileSetParameterBool"] = <_cyb_intptr_t>__cuFileSetParameterBool

    global __cuFileSetParameterString
    data["__cuFileSetParameterString"] = <_cyb_intptr_t>__cuFileSetParameterString

    global __cuFileGetParameterMinMaxValue
    data["__cuFileGetParameterMinMaxValue"] = <_cyb_intptr_t>__cuFileGetParameterMinMaxValue

    global __cuFileSetStatsLevel
    data["__cuFileSetStatsLevel"] = <_cyb_intptr_t>__cuFileSetStatsLevel

    global __cuFileGetStatsLevel
    data["__cuFileGetStatsLevel"] = <_cyb_intptr_t>__cuFileGetStatsLevel

    global __cuFileStatsStart
    data["__cuFileStatsStart"] = <_cyb_intptr_t>__cuFileStatsStart

    global __cuFileStatsStop
    data["__cuFileStatsStop"] = <_cyb_intptr_t>__cuFileStatsStop

    global __cuFileStatsReset
    data["__cuFileStatsReset"] = <_cyb_intptr_t>__cuFileStatsReset

    global __cuFileGetStatsL1
    data["__cuFileGetStatsL1"] = <_cyb_intptr_t>__cuFileGetStatsL1

    global __cuFileGetStatsL2
    data["__cuFileGetStatsL2"] = <_cyb_intptr_t>__cuFileGetStatsL2

    global __cuFileGetStatsL3
    data["__cuFileGetStatsL3"] = <_cyb_intptr_t>__cuFileGetStatsL3

    global __cuFileGetBARSizeInKB
    data["__cuFileGetBARSizeInKB"] = <_cyb_intptr_t>__cuFileGetBARSizeInKB

    global __cuFileSetParameterPosixPoolSlabArray
    data["__cuFileSetParameterPosixPoolSlabArray"] = <_cyb_intptr_t>__cuFileSetParameterPosixPoolSlabArray

    global __cuFileGetParameterPosixPoolSlabArray
    data["__cuFileGetParameterPosixPoolSlabArray"] = <_cyb_intptr_t>__cuFileGetParameterPosixPoolSlabArray
    _cyb_func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global _cyb_func_ptrs
    if _cyb_func_ptrs is None:
        _cyb_func_ptrs = _inspect_function_pointers()
    return _cyb_func_ptrs[name]




cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("cufile")._handle_uint
    return <void*>handle


###############################################################################
# Wrapper functions

cdef CUfileError_t _cuFileHandleRegister(CUfileHandle_t* fh, CUfileDescr_t* descr) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileHandleRegister
    _check_or_init_cufile()
    if __cuFileHandleRegister == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileHandleRegister is not found")
    return (<CUfileError_t (*)(CUfileHandle_t*, CUfileDescr_t*) noexcept nogil>__cuFileHandleRegister)(
        fh, descr)


@_cyb_cython.show_performance_hints(False)
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


cdef CUfileError_t _cuFileDriverClose() except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileDriverClose
    _check_or_init_cufile()
    if __cuFileDriverClose == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileDriverClose is not found")
    return (<CUfileError_t (*)() noexcept nogil>__cuFileDriverClose)(
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


@_cyb_cython.show_performance_hints(False)
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


cdef CUfileError_t _cuFileGetParameterMinMaxValue(CUFileSizeTConfigParameter_t param, size_t* min_value, size_t* max_value) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileGetParameterMinMaxValue
    _check_or_init_cufile()
    if __cuFileGetParameterMinMaxValue == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileGetParameterMinMaxValue is not found")
    return (<CUfileError_t (*)(CUFileSizeTConfigParameter_t, size_t*, size_t*) noexcept nogil>__cuFileGetParameterMinMaxValue)(
        param, min_value, max_value)


cdef CUfileError_t _cuFileSetStatsLevel(int level) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileSetStatsLevel
    _check_or_init_cufile()
    if __cuFileSetStatsLevel == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileSetStatsLevel is not found")
    return (<CUfileError_t (*)(int) noexcept nogil>__cuFileSetStatsLevel)(
        level)


cdef CUfileError_t _cuFileGetStatsLevel(int* level) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileGetStatsLevel
    _check_or_init_cufile()
    if __cuFileGetStatsLevel == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileGetStatsLevel is not found")
    return (<CUfileError_t (*)(int*) noexcept nogil>__cuFileGetStatsLevel)(
        level)


cdef CUfileError_t _cuFileStatsStart() except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileStatsStart
    _check_or_init_cufile()
    if __cuFileStatsStart == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileStatsStart is not found")
    return (<CUfileError_t (*)() noexcept nogil>__cuFileStatsStart)(
        )


cdef CUfileError_t _cuFileStatsStop() except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileStatsStop
    _check_or_init_cufile()
    if __cuFileStatsStop == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileStatsStop is not found")
    return (<CUfileError_t (*)() noexcept nogil>__cuFileStatsStop)(
        )


cdef CUfileError_t _cuFileStatsReset() except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileStatsReset
    _check_or_init_cufile()
    if __cuFileStatsReset == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileStatsReset is not found")
    return (<CUfileError_t (*)() noexcept nogil>__cuFileStatsReset)(
        )


cdef CUfileError_t _cuFileGetStatsL1(CUfileStatsLevel1_t* stats) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileGetStatsL1
    _check_or_init_cufile()
    if __cuFileGetStatsL1 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileGetStatsL1 is not found")
    return (<CUfileError_t (*)(CUfileStatsLevel1_t*) noexcept nogil>__cuFileGetStatsL1)(
        stats)


cdef CUfileError_t _cuFileGetStatsL2(CUfileStatsLevel2_t* stats) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileGetStatsL2
    _check_or_init_cufile()
    if __cuFileGetStatsL2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileGetStatsL2 is not found")
    return (<CUfileError_t (*)(CUfileStatsLevel2_t*) noexcept nogil>__cuFileGetStatsL2)(
        stats)


cdef CUfileError_t _cuFileGetStatsL3(CUfileStatsLevel3_t* stats) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileGetStatsL3
    _check_or_init_cufile()
    if __cuFileGetStatsL3 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileGetStatsL3 is not found")
    return (<CUfileError_t (*)(CUfileStatsLevel3_t*) noexcept nogil>__cuFileGetStatsL3)(
        stats)


cdef CUfileError_t _cuFileGetBARSizeInKB(int gpuIndex, size_t* barSize) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileGetBARSizeInKB
    _check_or_init_cufile()
    if __cuFileGetBARSizeInKB == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileGetBARSizeInKB is not found")
    return (<CUfileError_t (*)(int, size_t*) noexcept nogil>__cuFileGetBARSizeInKB)(
        gpuIndex, barSize)


cdef CUfileError_t _cuFileSetParameterPosixPoolSlabArray(const size_t* size_values, const size_t* count_values, int len) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileSetParameterPosixPoolSlabArray
    _check_or_init_cufile()
    if __cuFileSetParameterPosixPoolSlabArray == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileSetParameterPosixPoolSlabArray is not found")
    return (<CUfileError_t (*)(const size_t*, const size_t*, int) noexcept nogil>__cuFileSetParameterPosixPoolSlabArray)(
        size_values, count_values, len)


cdef CUfileError_t _cuFileGetParameterPosixPoolSlabArray(size_t* size_values, size_t* count_values, int len) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileGetParameterPosixPoolSlabArray
    _check_or_init_cufile()
    if __cuFileGetParameterPosixPoolSlabArray == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileGetParameterPosixPoolSlabArray is not found")
    return (<CUfileError_t (*)(size_t*, size_t*, int) noexcept nogil>__cuFileGetParameterPosixPoolSlabArray)(
        size_values, count_values, len)
