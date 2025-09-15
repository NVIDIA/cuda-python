# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.9.0 to 13.0.1. Do not modify it directly.

from ..cycufile cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef CUfileError_t _cuFileHandleRegister(CUfileHandle_t* fh, CUfileDescr_t* descr) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef void _cuFileHandleDeregister(CUfileHandle_t fh) except* nogil
cdef CUfileError_t _cuFileBufRegister(const void* bufPtr_base, size_t length, int flags) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileBufDeregister(const void* bufPtr_base) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef ssize_t _cuFileRead(CUfileHandle_t fh, void* bufPtr_base, size_t size, off_t file_offset, off_t bufPtr_offset) except* nogil
cdef ssize_t _cuFileWrite(CUfileHandle_t fh, const void* bufPtr_base, size_t size, off_t file_offset, off_t bufPtr_offset) except* nogil
cdef CUfileError_t _cuFileDriverOpen() except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileDriverClose_v2() except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef long _cuFileUseCount() except* nogil
cdef CUfileError_t _cuFileDriverGetProperties(CUfileDrvProps_t* props) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileDriverSetPollMode(cpp_bool poll, size_t poll_threshold_size) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileDriverSetMaxDirectIOSize(size_t max_direct_io_size) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileDriverSetMaxCacheSize(size_t max_cache_size) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileDriverSetMaxPinnedMemSize(size_t max_pinned_size) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileBatchIOSetUp(CUfileBatchHandle_t* batch_idp, unsigned nr) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileBatchIOSubmit(CUfileBatchHandle_t batch_idp, unsigned nr, CUfileIOParams_t* iocbp, unsigned int flags) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileBatchIOGetStatus(CUfileBatchHandle_t batch_idp, unsigned min_nr, unsigned* nr, CUfileIOEvents_t* iocbp, timespec* timeout) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileBatchIOCancel(CUfileBatchHandle_t batch_idp) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef void _cuFileBatchIODestroy(CUfileBatchHandle_t batch_idp) except* nogil
cdef CUfileError_t _cuFileReadAsync(CUfileHandle_t fh, void* bufPtr_base, size_t* size_p, off_t* file_offset_p, off_t* bufPtr_offset_p, ssize_t* bytes_read_p, CUstream stream) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileWriteAsync(CUfileHandle_t fh, void* bufPtr_base, size_t* size_p, off_t* file_offset_p, off_t* bufPtr_offset_p, ssize_t* bytes_written_p, CUstream stream) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileStreamRegister(CUstream stream, unsigned flags) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileStreamDeregister(CUstream stream) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileGetVersion(int* version) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileGetParameterSizeT(CUFileSizeTConfigParameter_t param, size_t* value) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileGetParameterBool(CUFileBoolConfigParameter_t param, cpp_bool* value) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileGetParameterString(CUFileStringConfigParameter_t param, char* desc_str, int len) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileSetParameterSizeT(CUFileSizeTConfigParameter_t param, size_t value) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileSetParameterBool(CUFileBoolConfigParameter_t param, cpp_bool value) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileSetParameterString(CUFileStringConfigParameter_t param, const char* desc_str) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileDriverClose() except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileGetParameterMinMaxValue(CUFileSizeTConfigParameter_t param, size_t* min_value, size_t* max_value) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileSetStatsLevel(int level) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileGetStatsLevel(int* level) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileStatsStart() except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileStatsStop() except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileStatsReset() except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileGetStatsL1(CUfileStatsLevel1_t* stats) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileGetStatsL2(CUfileStatsLevel2_t* stats) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileGetStatsL3(CUfileStatsLevel3_t* stats) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileGetBARSizeInKB(int gpuIndex, size_t* barSize) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileSetParameterPosixPoolSlabArray(const size_t* size_values, const size_t* count_values, int len) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
cdef CUfileError_t _cuFileGetParameterPosixPoolSlabArray(size_t* size_values, size_t* count_values, int len) except?<CUfileError_t>CUFILE_LOADING_ERROR nogil
