# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 12.9.1. Do not modify it directly.
# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=35f027c37095d4df851e8339878e4d5b770eeda1e3a205649014a155c6d3d3c6


# <<<< PREAMBLE CONTENT >>>>

cimport cython as _cyb_cython
from libc.stdint cimport intptr_t
from libcpp cimport bool as _cyb_bool

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

import threading as _cyb_threading

cdef int _cyb___py_cufile_init = 0
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

cdef enum: _cyb_cufile_n_functions = 31
cdef void** _cyb_cufile_slots[_cyb_cufile_n_functions]
cdef const char* _cyb_cufile_names[_cyb_cufile_n_functions]
_cyb_cufile_slots[0] = &__cuFileHandleRegister; _cyb_cufile_names[0] = b'cuFileHandleRegister'
_cyb_cufile_slots[1] = &__cuFileHandleDeregister; _cyb_cufile_names[1] = b'cuFileHandleDeregister'
_cyb_cufile_slots[2] = &__cuFileBufRegister; _cyb_cufile_names[2] = b'cuFileBufRegister'
_cyb_cufile_slots[3] = &__cuFileBufDeregister; _cyb_cufile_names[3] = b'cuFileBufDeregister'
_cyb_cufile_slots[4] = &__cuFileRead; _cyb_cufile_names[4] = b'cuFileRead'
_cyb_cufile_slots[5] = &__cuFileWrite; _cyb_cufile_names[5] = b'cuFileWrite'
_cyb_cufile_slots[6] = &__cuFileDriverOpen; _cyb_cufile_names[6] = b'cuFileDriverOpen'
_cyb_cufile_slots[7] = &__cuFileDriverClose; _cyb_cufile_names[7] = b'cuFileDriverClose'
_cyb_cufile_slots[8] = &__cuFileDriverClose_v2; _cyb_cufile_names[8] = b'cuFileDriverClose_v2'
_cyb_cufile_slots[9] = &__cuFileUseCount; _cyb_cufile_names[9] = b'cuFileUseCount'
_cyb_cufile_slots[10] = &__cuFileDriverGetProperties; _cyb_cufile_names[10] = b'cuFileDriverGetProperties'
_cyb_cufile_slots[11] = &__cuFileDriverSetPollMode; _cyb_cufile_names[11] = b'cuFileDriverSetPollMode'
_cyb_cufile_slots[12] = &__cuFileDriverSetMaxDirectIOSize; _cyb_cufile_names[12] = b'cuFileDriverSetMaxDirectIOSize'
_cyb_cufile_slots[13] = &__cuFileDriverSetMaxCacheSize; _cyb_cufile_names[13] = b'cuFileDriverSetMaxCacheSize'
_cyb_cufile_slots[14] = &__cuFileDriverSetMaxPinnedMemSize; _cyb_cufile_names[14] = b'cuFileDriverSetMaxPinnedMemSize'
_cyb_cufile_slots[15] = &__cuFileBatchIOSetUp; _cyb_cufile_names[15] = b'cuFileBatchIOSetUp'
_cyb_cufile_slots[16] = &__cuFileBatchIOSubmit; _cyb_cufile_names[16] = b'cuFileBatchIOSubmit'
_cyb_cufile_slots[17] = &__cuFileBatchIOGetStatus; _cyb_cufile_names[17] = b'cuFileBatchIOGetStatus'
_cyb_cufile_slots[18] = &__cuFileBatchIOCancel; _cyb_cufile_names[18] = b'cuFileBatchIOCancel'
_cyb_cufile_slots[19] = &__cuFileBatchIODestroy; _cyb_cufile_names[19] = b'cuFileBatchIODestroy'
_cyb_cufile_slots[20] = &__cuFileReadAsync; _cyb_cufile_names[20] = b'cuFileReadAsync'
_cyb_cufile_slots[21] = &__cuFileWriteAsync; _cyb_cufile_names[21] = b'cuFileWriteAsync'
_cyb_cufile_slots[22] = &__cuFileStreamRegister; _cyb_cufile_names[22] = b'cuFileStreamRegister'
_cyb_cufile_slots[23] = &__cuFileStreamDeregister; _cyb_cufile_names[23] = b'cuFileStreamDeregister'
_cyb_cufile_slots[24] = &__cuFileGetVersion; _cyb_cufile_names[24] = b'cuFileGetVersion'
_cyb_cufile_slots[25] = &__cuFileGetParameterSizeT; _cyb_cufile_names[25] = b'cuFileGetParameterSizeT'
_cyb_cufile_slots[26] = &__cuFileGetParameterBool; _cyb_cufile_names[26] = b'cuFileGetParameterBool'
_cyb_cufile_slots[27] = &__cuFileGetParameterString; _cyb_cufile_names[27] = b'cuFileGetParameterString'
_cyb_cufile_slots[28] = &__cuFileSetParameterSizeT; _cyb_cufile_names[28] = b'cuFileSetParameterSizeT'
_cyb_cufile_slots[29] = &__cuFileSetParameterBool; _cyb_cufile_names[29] = b'cuFileSetParameterBool'
_cyb_cufile_slots[30] = &__cuFileSetParameterString; _cyb_cufile_names[30] = b'cuFileSetParameterString'

cdef int _init_cufile() except -1 nogil:
    global _cyb___py_cufile_init
    cdef void* handle = NULL
    cdef Py_ssize_t i
    with gil, _cyb_symbol_lock:
        if _cyb___py_cufile_init: return 0

        for i in range(_cyb_cufile_n_functions):
            _cyb_cufile_slots[i][0] = _cyb_dlsym(_cyb_RTLD_DEFAULT, _cyb_cufile_names[i])
            if _cyb_cufile_slots[i][0] == NULL:
                if handle == NULL:
                    handle = load_library()
                _cyb_cufile_slots[i][0] = _cyb_dlsym(handle, _cyb_cufile_names[i])

        _cyb_atomic_int_store(<int *>&_cyb___py_cufile_init, 1)
        return 0

cdef inline int _check_or_init_cufile() except -1 nogil:
    if _cyb_atomic_int_load(<int *>&_cyb___py_cufile_init):
        return 0

    return _init_cufile()


cpdef dict _inspect_function_pointers():
    global _cyb_func_ptrs
    if _cyb_func_ptrs is not None:
        return _cyb_func_ptrs

    _check_or_init_cufile()
    cdef dict data = {}
    cdef Py_ssize_t i
    for i in range(_cyb_cufile_n_functions):
        data['__' + _cyb_cufile_names[i].decode()] = <intptr_t>_cyb_cufile_slots[i][0]
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


cdef CUfileError_t _cuFileDriverSetPollMode(_cyb_bool poll, size_t poll_threshold_size) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileDriverSetPollMode
    _check_or_init_cufile()
    if __cuFileDriverSetPollMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileDriverSetPollMode is not found")
    return (<CUfileError_t (*)(_cyb_bool, size_t) noexcept nogil>__cuFileDriverSetPollMode)(
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


cdef CUfileError_t _cuFileGetParameterBool(CUFileBoolConfigParameter_t param, _cyb_bool* value) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileGetParameterBool
    _check_or_init_cufile()
    if __cuFileGetParameterBool == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileGetParameterBool is not found")
    return (<CUfileError_t (*)(CUFileBoolConfigParameter_t, _cyb_bool*) noexcept nogil>__cuFileGetParameterBool)(
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


cdef CUfileError_t _cuFileSetParameterBool(CUFileBoolConfigParameter_t param, _cyb_bool value) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileSetParameterBool
    _check_or_init_cufile()
    if __cuFileSetParameterBool == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileSetParameterBool is not found")
    return (<CUfileError_t (*)(CUFileBoolConfigParameter_t, _cyb_bool) noexcept nogil>__cuFileSetParameterBool)(
        param, value)


cdef CUfileError_t _cuFileSetParameterString(CUFileStringConfigParameter_t param, const char* desc_str) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    global __cuFileSetParameterString
    _check_or_init_cufile()
    if __cuFileSetParameterString == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFileSetParameterString is not found")
    return (<CUfileError_t (*)(CUFileStringConfigParameter_t, const char*) noexcept nogil>__cuFileSetParameterString)(
        param, desc_str)
