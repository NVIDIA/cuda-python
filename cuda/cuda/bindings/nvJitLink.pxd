# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.0.1 to 12.4.1. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cynvJitLink cimport *


###############################################################################
# Types
###############################################################################



ctypedef cudaStream_t Stream
ctypedef cudaDataType DataType
ctypedef libraryPropertyType_t LibraryPropertyType


###############################################################################
# Enum
###############################################################################




###############################################################################
# Functions
###############################################################################

cpdef create(intptr_t handle, uint32_t num_options, intptr_t options)
cpdef destroy(intptr_t handle)
cpdef add_data(nvJitLinkHandle handle, nvJitLinkInputType input_type, intptr_t data, size_t size, intptr_t name)
cpdef add_file(nvJitLinkHandle handle, nvJitLinkInputType input_type, intptr_t file_name)
cpdef complete(nvJitLinkHandle handle)
cpdef get_linked_cubin_size(nvJitLinkHandle handle, intptr_t size)
cpdef get_linked_cubin(nvJitLinkHandle handle, intptr_t cubin)
cpdef get_linked_ptx_size(nvJitLinkHandle handle, intptr_t size)
cpdef get_linked_ptx(nvJitLinkHandle handle, intptr_t ptx)
cpdef get_error_log_size(nvJitLinkHandle handle, intptr_t size)
cpdef get_error_log(nvJitLinkHandle handle, intptr_t log)
cpdef get_info_log_size(nvJitLinkHandle handle, intptr_t size)
cpdef get_info_log(nvJitLinkHandle handle, intptr_t log)
