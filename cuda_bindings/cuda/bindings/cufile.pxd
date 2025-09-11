# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.9.0 to 13.0.1. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cycufile cimport *


###############################################################################
# Types
###############################################################################

ctypedef CUfileHandle_t Handle
ctypedef CUfileBatchHandle_t BatchHandle
ctypedef CUfileError_t Error
ctypedef cufileRDMAInfo_t RDMAInfo
ctypedef CUfileFSOps_t FSOps
ctypedef CUfileOpCounter_t OpCounter
ctypedef CUfilePerGpuStats_t PerGpuStats
ctypedef CUfileDrvProps_t DrvProps
ctypedef CUfileStatsLevel1_t StatsLevel1
ctypedef CUfileStatsLevel2_t StatsLevel2
ctypedef CUfileStatsLevel3_t StatsLevel3


###############################################################################
# Enum
###############################################################################

ctypedef CUfileOpError _OpError
ctypedef CUfileDriverStatusFlags_t _DriverStatusFlags
ctypedef CUfileDriverControlFlags_t _DriverControlFlags
ctypedef CUfileFeatureFlags_t _FeatureFlags
ctypedef CUfileFileHandleType _FileHandleType
ctypedef CUfileOpcode_t _Opcode
ctypedef CUfileStatus_t _Status
ctypedef CUfileBatchMode_t _BatchMode
ctypedef CUFileSizeTConfigParameter_t _SizeTConfigParameter
ctypedef CUFileBoolConfigParameter_t _BoolConfigParameter
ctypedef CUFileStringConfigParameter_t _StringConfigParameter
ctypedef CUFileArrayConfigParameter_t _ArrayConfigParameter


###############################################################################
# Functions
###############################################################################

cpdef intptr_t handle_register(intptr_t descr) except? 0
cpdef void handle_deregister(intptr_t fh) except*
cpdef buf_register(intptr_t buf_ptr_base, size_t length, int flags)
cpdef buf_deregister(intptr_t buf_ptr_base)
cpdef read(intptr_t fh, intptr_t buf_ptr_base, size_t size, off_t file_offset, off_t buf_ptr_offset)
cpdef write(intptr_t fh, intptr_t buf_ptr_base, size_t size, off_t file_offset, off_t buf_ptr_offset)
cpdef driver_open()
cpdef use_count()
cpdef driver_get_properties(intptr_t props)
cpdef driver_set_poll_mode(bint poll, size_t poll_threshold_size)
cpdef driver_set_max_direct_io_size(size_t max_direct_io_size)
cpdef driver_set_max_cache_size(size_t max_cache_size)
cpdef driver_set_max_pinned_mem_size(size_t max_pinned_size)
cpdef intptr_t batch_io_set_up(unsigned nr) except? 0
cpdef batch_io_submit(intptr_t batch_idp, unsigned nr, intptr_t iocbp, unsigned int flags)
cpdef batch_io_get_status(intptr_t batch_idp, unsigned min_nr, intptr_t nr, intptr_t iocbp, intptr_t timeout)
cpdef batch_io_cancel(intptr_t batch_idp)
cpdef void batch_io_destroy(intptr_t batch_idp) except*
cpdef read_async(intptr_t fh, intptr_t buf_ptr_base, intptr_t size_p, intptr_t file_offset_p, intptr_t buf_ptr_offset_p, intptr_t bytes_read_p, intptr_t stream)
cpdef write_async(intptr_t fh, intptr_t buf_ptr_base, intptr_t size_p, intptr_t file_offset_p, intptr_t buf_ptr_offset_p, intptr_t bytes_written_p, intptr_t stream)
cpdef stream_register(intptr_t stream, unsigned flags)
cpdef stream_deregister(intptr_t stream)
cpdef int get_version() except? 0
cpdef size_t get_parameter_size_t(int param) except? 0
cpdef bint get_parameter_bool(int param) except? 0
cpdef str get_parameter_string(int param, int len)
cpdef set_parameter_size_t(int param, size_t value)
cpdef set_parameter_bool(int param, bint value)
cpdef set_parameter_string(int param, intptr_t desc_str)
