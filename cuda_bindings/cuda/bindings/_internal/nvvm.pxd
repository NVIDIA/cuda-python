# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated with version 12.6.1. Do not modify it directly.

from ..cynvvm cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef nvvmResult _nvvmVersion(int* major, int* minor) except* nogil
cdef nvvmResult _nvvmIRVersion(int* majorIR, int* minorIR, int* majorDbg, int* minorDbg) except* nogil
cdef nvvmResult _nvvmCreateProgram(nvvmProgram* prog) except* nogil
cdef nvvmResult _nvvmDestroyProgram(nvvmProgram* prog) except* nogil
cdef nvvmResult _nvvmAddModuleToProgram(nvvmProgram prog, const char* buffer, size_t size, const char* name) except* nogil
cdef nvvmResult _nvvmCompileProgram(nvvmProgram prog, int numOptions, const char** options) except* nogil
