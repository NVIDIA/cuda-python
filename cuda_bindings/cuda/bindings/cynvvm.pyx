# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.0.1 to 13.0.1. Do not modify it directly.

from ._internal cimport nvvm as _nvvm


###############################################################################
# Wrapper functions
###############################################################################

cdef const char* nvvmGetErrorString(nvvmResult result) except?NULL nogil:
    return _nvvm._nvvmGetErrorString(result)


cdef nvvmResult nvvmVersion(int* major, int* minor) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvvm._nvvmVersion(major, minor)


cdef nvvmResult nvvmIRVersion(int* majorIR, int* minorIR, int* majorDbg, int* minorDbg) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvvm._nvvmIRVersion(majorIR, minorIR, majorDbg, minorDbg)


cdef nvvmResult nvvmCreateProgram(nvvmProgram* prog) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvvm._nvvmCreateProgram(prog)


cdef nvvmResult nvvmDestroyProgram(nvvmProgram* prog) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvvm._nvvmDestroyProgram(prog)


cdef nvvmResult nvvmAddModuleToProgram(nvvmProgram prog, const char* buffer, size_t size, const char* name) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvvm._nvvmAddModuleToProgram(prog, buffer, size, name)


cdef nvvmResult nvvmLazyAddModuleToProgram(nvvmProgram prog, const char* buffer, size_t size, const char* name) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvvm._nvvmLazyAddModuleToProgram(prog, buffer, size, name)


cdef nvvmResult nvvmCompileProgram(nvvmProgram prog, int numOptions, const char** options) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvvm._nvvmCompileProgram(prog, numOptions, options)


cdef nvvmResult nvvmVerifyProgram(nvvmProgram prog, int numOptions, const char** options) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvvm._nvvmVerifyProgram(prog, numOptions, options)


cdef nvvmResult nvvmGetCompiledResultSize(nvvmProgram prog, size_t* bufferSizeRet) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvvm._nvvmGetCompiledResultSize(prog, bufferSizeRet)


cdef nvvmResult nvvmGetCompiledResult(nvvmProgram prog, char* buffer) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvvm._nvvmGetCompiledResult(prog, buffer)


cdef nvvmResult nvvmGetProgramLogSize(nvvmProgram prog, size_t* bufferSizeRet) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvvm._nvvmGetProgramLogSize(prog, bufferSizeRet)


cdef nvvmResult nvvmGetProgramLog(nvvmProgram prog, char* buffer) except?_NVVMRESULT_INTERNAL_LOADING_ERROR nogil:
    return _nvvm._nvvmGetProgramLog(prog, buffer)
