# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.4.1 to 13.3.0. Do not modify it directly.
# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=d9fd5ffb6adedf403c2fee0594d979c6ad2221c94f886cdaaa5efb01c1fa1421



# <<<< PREAMBLE CONTENT >>>>

from libc.stdint cimport intptr_t


# <<<< END OF PREAMBLE CONTENT >>>>

from .cynvfatbin cimport *


###############################################################################
# Types
###############################################################################

ctypedef nvFatbinHandle Handle


###############################################################################
# Enum
###############################################################################

ctypedef nvFatbinResult _Result


###############################################################################
# Functions
###############################################################################

cpdef intptr_t create(options, size_t options_count) except -1
cpdef add_ptx(intptr_t handle, code, size_t size, arch, identifier, options_cmd_line)
cpdef add_cubin(intptr_t handle, code, size_t size, arch, identifier)
cpdef add_ltoir(intptr_t handle, code, size_t size, arch, identifier, options_cmd_line)
cpdef size_t size(intptr_t handle) except? 0
cpdef get(intptr_t handle, buffer)
cpdef tuple version()
cpdef add_index(intptr_t handle, code, size_t size, identifier)
cpdef add_reloc(intptr_t handle, code, size_t size)
cpdef add_tile_ir(intptr_t handle, code, size_t size, identifier, options_cmd_line)
