# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.0.1 to 12.6.2. Do not modify it directly.

from libc.stdint cimport intptr_t, uint32_t

from .cynvjitlink cimport *


###############################################################################
# Types
###############################################################################

ctypedef nvJitLinkHandle Handle


###############################################################################
# Enum
###############################################################################

ctypedef nvJitLinkResult _Result
ctypedef nvJitLinkInputType _InputType


###############################################################################
# Functions
###############################################################################

cpdef intptr_t create(uint32_t num_options, options) except -1
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