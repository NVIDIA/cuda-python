# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# CYBIND-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=679e46cfd685c708f5ea57e8063e088710420cf92d6f98c53aa3edef1bd8cfac
#
# This code was automatically generated across versions from 12.0.1 to 13.4.0, generator version 0.3.1.dev1881+g248da917e. Do not modify it directly.

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
cpdef add_data(intptr_t handle, int input_type, data, size_t size, name)
cpdef add_file(intptr_t handle, int input_type, file_name)
cpdef complete(intptr_t handle)
cpdef size_t get_linked_cubin_size(intptr_t handle) except? 0
cpdef get_linked_cubin(intptr_t handle, cubin)
cpdef size_t get_linked_ptx_size(intptr_t handle) except? 0
cpdef get_linked_ptx(intptr_t handle, ptx)
cpdef size_t get_error_log_size(intptr_t handle) except? 0
cpdef get_error_log(intptr_t handle, log)
cpdef size_t get_info_log_size(intptr_t handle) except? 0
cpdef get_info_log(intptr_t handle, log)
cpdef tuple version()
cpdef size_t get_linked_ltoir_size(intptr_t handle) except? 0
cpdef get_linked_ltoir(intptr_t handle, ltoir)
