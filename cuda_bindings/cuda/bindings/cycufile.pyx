# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.9.0 to 12.9.1. Do not modify it directly.

from ._internal cimport cufile as _cufile

import cython

###############################################################################
# Wrapper functions
###############################################################################

cdef CUfileError_t cuFileHandleRegister(CUfileHandle_t* fh, CUfileDescr_t* descr) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileHandleRegister(fh, descr)


@cython.show_performance_hints(False)
cdef void cuFileHandleDeregister(CUfileHandle_t fh) except* nogil:
    _cufile._cuFileHandleDeregister(fh)


cdef CUfileError_t cuFileBufRegister(const void* bufPtr_base, size_t length, int flags) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileBufRegister(bufPtr_base, length, flags)


cdef CUfileError_t cuFileBufDeregister(const void* bufPtr_base) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileBufDeregister(bufPtr_base)


cdef ssize_t cuFileRead(CUfileHandle_t fh, void* bufPtr_base, size_t size, off_t file_offset, off_t bufPtr_offset) except* nogil:
    return _cufile._cuFileRead(fh, bufPtr_base, size, file_offset, bufPtr_offset)


cdef ssize_t cuFileWrite(CUfileHandle_t fh, const void* bufPtr_base, size_t size, off_t file_offset, off_t bufPtr_offset) except* nogil:
    return _cufile._cuFileWrite(fh, bufPtr_base, size, file_offset, bufPtr_offset)


cdef CUfileError_t cuFileDriverOpen() except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileDriverOpen()


cdef CUfileError_t cuFileDriverClose_v2() except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileDriverClose_v2()


cdef long cuFileUseCount() except* nogil:
    return _cufile._cuFileUseCount()


cdef CUfileError_t cuFileDriverGetProperties(CUfileDrvProps_t* props) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileDriverGetProperties(props)


cdef CUfileError_t cuFileDriverSetPollMode(cpp_bool poll, size_t poll_threshold_size) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileDriverSetPollMode(poll, poll_threshold_size)


cdef CUfileError_t cuFileDriverSetMaxDirectIOSize(size_t max_direct_io_size) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileDriverSetMaxDirectIOSize(max_direct_io_size)


cdef CUfileError_t cuFileDriverSetMaxCacheSize(size_t max_cache_size) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileDriverSetMaxCacheSize(max_cache_size)


cdef CUfileError_t cuFileDriverSetMaxPinnedMemSize(size_t max_pinned_size) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileDriverSetMaxPinnedMemSize(max_pinned_size)


cdef CUfileError_t cuFileBatchIOSetUp(CUfileBatchHandle_t* batch_idp, unsigned nr) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileBatchIOSetUp(batch_idp, nr)


cdef CUfileError_t cuFileBatchIOSubmit(CUfileBatchHandle_t batch_idp, unsigned nr, CUfileIOParams_t* iocbp, unsigned int flags) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileBatchIOSubmit(batch_idp, nr, iocbp, flags)


cdef CUfileError_t cuFileBatchIOGetStatus(CUfileBatchHandle_t batch_idp, unsigned min_nr, unsigned* nr, CUfileIOEvents_t* iocbp, timespec* timeout) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileBatchIOGetStatus(batch_idp, min_nr, nr, iocbp, timeout)


cdef CUfileError_t cuFileBatchIOCancel(CUfileBatchHandle_t batch_idp) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileBatchIOCancel(batch_idp)


@cython.show_performance_hints(False)
cdef void cuFileBatchIODestroy(CUfileBatchHandle_t batch_idp) except* nogil:
    _cufile._cuFileBatchIODestroy(batch_idp)


cdef CUfileError_t cuFileReadAsync(CUfileHandle_t fh, void* bufPtr_base, size_t* size_p, off_t* file_offset_p, off_t* bufPtr_offset_p, ssize_t* bytes_read_p, CUstream stream) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileReadAsync(fh, bufPtr_base, size_p, file_offset_p, bufPtr_offset_p, bytes_read_p, stream)


cdef CUfileError_t cuFileWriteAsync(CUfileHandle_t fh, void* bufPtr_base, size_t* size_p, off_t* file_offset_p, off_t* bufPtr_offset_p, ssize_t* bytes_written_p, CUstream stream) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileWriteAsync(fh, bufPtr_base, size_p, file_offset_p, bufPtr_offset_p, bytes_written_p, stream)


cdef CUfileError_t cuFileStreamRegister(CUstream stream, unsigned flags) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileStreamRegister(stream, flags)


cdef CUfileError_t cuFileStreamDeregister(CUstream stream) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileStreamDeregister(stream)


cdef CUfileError_t cuFileGetVersion(int* version) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileGetVersion(version)


cdef CUfileError_t cuFileGetParameterSizeT(CUFileSizeTConfigParameter_t param, size_t* value) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileGetParameterSizeT(param, value)


cdef CUfileError_t cuFileGetParameterBool(CUFileBoolConfigParameter_t param, cpp_bool* value) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileGetParameterBool(param, value)


cdef CUfileError_t cuFileGetParameterString(CUFileStringConfigParameter_t param, char* desc_str, int len) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileGetParameterString(param, desc_str, len)


cdef CUfileError_t cuFileSetParameterSizeT(CUFileSizeTConfigParameter_t param, size_t value) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileSetParameterSizeT(param, value)


cdef CUfileError_t cuFileSetParameterBool(CUFileBoolConfigParameter_t param, cpp_bool value) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileSetParameterBool(param, value)


cdef CUfileError_t cuFileSetParameterString(CUFileStringConfigParameter_t param, const char* desc_str) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileSetParameterString(param, desc_str)


cdef CUfileError_t cuFileDriverClose() except?<CUfileError_t>CUFILE_LOADING_ERROR nogil:
    return _cufile._cuFileDriverClose()
