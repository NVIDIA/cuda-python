# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated with version 12.6.1. Do not modify it directly.

from libc.stdint cimport intptr_t, uint32_t

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

cpdef tuple version()
cpdef tuple ir_version()
cpdef intptr_t create_program() except? 0
cpdef add_module_to_program(intptr_t prog, buffer, size_t size, name)
cpdef compile_program(intptr_t prog, int num_options, options)
cpdef verify_program(intptr_t prog, int num_options, options)
