# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.0.76 to 12.6.77. Do not modify it directly.

from libc.stdint cimport intptr_t, uint32_t

from .cynvjitlink cimport *


###############################################################################
# Types
###############################################################################

ctypedef nvJitLinkHandle Handle


###############################################################################
# Enum
###############################################################################

ctypedef nvJitLinkResult _NvJitLinkResult
ctypedef nvJitLinkInputType _NvJitLinkInputType


###############################################################################
# Functions
###############################################################################

cpdef create(intptr_t handle, uint32_t num_options, intptr_t options)
cpdef destroy(intptr_t handle)
cpdef add_data(intptr_t handle, int input_type, intptr_t data, size_t size, intptr_t name)
cpdef add_file(intptr_t handle, int input_type, intptr_t file_name)
cpdef complete(intptr_t handle)
cpdef get_linked_cubin_size(intptr_t handle, intptr_t size)
cpdef get_linked_cubin(intptr_t handle, intptr_t cubin)
cpdef get_linked_ptx_size(intptr_t handle, intptr_t size)
cpdef get_linked_ptx(intptr_t handle, intptr_t ptx)
cpdef get_error_log_size(intptr_t handle, intptr_t size)
cpdef get_error_log(intptr_t handle, intptr_t log)
cpdef get_info_log_size(intptr_t handle, intptr_t size)
cpdef get_info_log(intptr_t handle, intptr_t log)