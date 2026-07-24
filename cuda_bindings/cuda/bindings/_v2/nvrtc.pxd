# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.9.0 to 13.3.0. Do not modify it directly.

# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=223716a3d6961dfe7231f2c88e76a9ab4c0a8d888cc4bf3e889bfbcbf8580346
from libc.stdint cimport intptr_t

from ..cynvrtc cimport *


###############################################################################
# Types
###############################################################################

ctypedef nvrtcProgram Program


###############################################################################
# Enum
###############################################################################

ctypedef nvrtcResult _Result


###############################################################################
# Functions
###############################################################################

cpdef intptr_t create_program(bytes src, name, headers=*, include_names=*) except? 0
cpdef compile_program(intptr_t prog, options=*)
cpdef set_flow_callback(intptr_t prog, intptr_t callback, intptr_t payload)
cpdef bytes get_lowered_name(intptr_t prog, bytes name_expression)
cpdef install_bundled_headers(bytes install_path, unsigned int flags)
cpdef tuple get_bundled_headers_info()
cpdef remove_bundled_headers(bytes install_path)

cpdef str get_error_string(int result)
cpdef tuple version()
cpdef int get_num_supported_archs() except? -1
cpdef object get_supported_archs()
cpdef destroy_program(intptr_t prog)
cpdef size_t get_ptx_size(intptr_t prog) except? 0
cpdef bytes get_ptx(intptr_t prog)
cpdef size_t get_cubin_size(intptr_t prog) except? 0
cpdef bytes get_cubin(intptr_t prog)
cpdef size_t get_ltoir_size(intptr_t prog) except? 0
cpdef bytes get_ltoir(intptr_t prog)
cpdef size_t get_optix_ir_size(intptr_t prog) except? 0
cpdef bytes get_optix_ir(intptr_t prog)
cpdef size_t get_program_log_size(intptr_t prog) except? 0
cpdef bytes get_program_log(intptr_t prog)
cpdef add_name_expression(intptr_t prog, name_expression)
cpdef size_t get_pch_heap_size() except? 0
cpdef set_pch_heap_size(size_t size)
cpdef int get_pch_create_status(intptr_t prog) except? -1
cpdef size_t get_pch_heap_size_required(intptr_t prog) except? 0
cpdef size_t get_tile_ir_size(intptr_t prog) except? 0
cpdef bytes get_tile_ir(intptr_t prog)
