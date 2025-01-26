# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated with version 12.6.1. Do not modify it directly.

from libc.stdint cimport intptr_t, uint32_t


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum nvvmResult "nvvmResult":
    NVVM_SUCCESS "NVVM_SUCCESS" = 0
    NVVM_ERROR_OUT_OF_MEMORY "NVVM_ERROR_OUT_OF_MEMORY" = 1
    NVVM_ERROR_PROGRAM_CREATION_FAILURE "NVVM_ERROR_PROGRAM_CREATION_FAILURE" = 2
    NVVM_ERROR_IR_VERSION_MISMATCH "NVVM_ERROR_IR_VERSION_MISMATCH" = 3
    NVVM_ERROR_INVALID_INPUT "NVVM_ERROR_INVALID_INPUT" = 4
    NVVM_ERROR_INVALID_PROGRAM "NVVM_ERROR_INVALID_PROGRAM" = 5
    NVVM_ERROR_INVALID_IR "NVVM_ERROR_INVALID_IR" = 6
    NVVM_ERROR_INVALID_OPTION "NVVM_ERROR_INVALID_OPTION" = 7
    NVVM_ERROR_NO_MODULE_IN_PROGRAM "NVVM_ERROR_NO_MODULE_IN_PROGRAM" = 8
    NVVM_ERROR_COMPILATION "NVVM_ERROR_COMPILATION" = 9


# types
ctypedef void* nvvmProgram 'nvvmProgram'


###############################################################################
# Functions
###############################################################################

cdef nvvmResult nvvmVersion(int* major, int* minor) except* nogil
cdef nvvmResult nvvmIRVersion(int* majorIR, int* minorIR, int* majorDbg, int* minorDbg) except* nogil
cdef nvvmResult nvvmCreateProgram(nvvmProgram* prog) except* nogil
cdef nvvmResult nvvmDestroyProgram(nvvmProgram* prog) except* nogil
