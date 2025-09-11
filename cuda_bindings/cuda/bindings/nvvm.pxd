# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.0.1 to 13.0.1. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cynvvm cimport *


###############################################################################
# Types
###############################################################################

ctypedef nvvmProgram Program


###############################################################################
# Enum
###############################################################################

ctypedef nvvmResult _Result


###############################################################################
# Functions
###############################################################################

cpdef str get_error_string(int result)
cpdef tuple version()
cpdef tuple ir_version()
cpdef intptr_t create_program() except? 0
cpdef add_module_to_program(intptr_t prog, buffer, size_t size, name)
cpdef lazy_add_module_to_program(intptr_t prog, buffer, size_t size, name)
cpdef compile_program(intptr_t prog, int num_options, options)
cpdef verify_program(intptr_t prog, int num_options, options)
cpdef size_t get_compiled_result_size(intptr_t prog) except? 0
cpdef get_compiled_result(intptr_t prog, buffer)
cpdef size_t get_program_log_size(intptr_t prog) except? 0
cpdef get_program_log(intptr_t prog, buffer)
