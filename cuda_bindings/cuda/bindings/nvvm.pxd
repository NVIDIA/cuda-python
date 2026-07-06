# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# CYBIND-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=836bcb954d3db4313e3476bfd2e80f1318724c478a277e2d0194f4ad714e02ef
#
# This code was automatically generated across versions from 12.0.1 to 13.4.0, generator version 0.3.1.dev1881+g248da917e. Do not modify it directly.

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
cpdef int llvm_version(arch) except? 0
