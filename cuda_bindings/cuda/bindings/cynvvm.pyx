# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated with version 12.6.1. Do not modify it directly.

from ._internal cimport nvvm as _nvvm


###############################################################################
# Wrapper functions
###############################################################################

cdef nvvmResult nvvmVersion(int* major, int* minor) except* nogil:
    return _nvvm._nvvmVersion(major, minor)


cdef nvvmResult nvvmIRVersion(int* majorIR, int* minorIR, int* majorDbg, int* minorDbg) except* nogil:
    return _nvvm._nvvmIRVersion(majorIR, minorIR, majorDbg, minorDbg)


cdef nvvmResult nvvmCreateProgram(nvvmProgram* prog) except* nogil:
    return _nvvm._nvvmCreateProgram(prog)
