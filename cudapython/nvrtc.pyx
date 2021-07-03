# Copyright 2021 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
from typing import List, Tuple, Any
from enum import Enum
import cython
import ctypes
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy
from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t
from libc.stddef cimport wchar_t
from libcpp.vector cimport vector
from cpython.buffer cimport PyObject_CheckBuffer, PyObject_GetBuffer, PyBuffer_Release, PyBUF_SIMPLE, PyBUF_ANY_CONTIGUOUS

ctypedef unsigned long long signed_char_ptr
ctypedef unsigned long long unsigned_char_ptr
ctypedef unsigned long long char_ptr
ctypedef unsigned long long short_ptr
ctypedef unsigned long long unsigned_short_ptr
ctypedef unsigned long long int_ptr
ctypedef unsigned long long long_int_ptr
ctypedef unsigned long long long_long_int_ptr
ctypedef unsigned long long unsigned_int_ptr
ctypedef unsigned long long unsigned_long_int_ptr
ctypedef unsigned long long unsigned_long_long_int_ptr
ctypedef unsigned long long uint32_t_ptr
ctypedef unsigned long long uint64_t_ptr
ctypedef unsigned long long int32_t_ptr
ctypedef unsigned long long int64_t_ptr
ctypedef unsigned long long unsigned_ptr
ctypedef unsigned long long unsigned_long_long_ptr
ctypedef unsigned long long size_t_ptr
ctypedef unsigned long long float_ptr
ctypedef unsigned long long double_ptr
ctypedef unsigned long long void_ptr


class nvrtcResult(Enum):
    NVRTC_SUCCESS = cnvrtc.nvrtcResult.NVRTC_SUCCESS
    NVRTC_ERROR_OUT_OF_MEMORY = cnvrtc.nvrtcResult.NVRTC_ERROR_OUT_OF_MEMORY
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = cnvrtc.nvrtcResult.NVRTC_ERROR_PROGRAM_CREATION_FAILURE
    NVRTC_ERROR_INVALID_INPUT = cnvrtc.nvrtcResult.NVRTC_ERROR_INVALID_INPUT
    NVRTC_ERROR_INVALID_PROGRAM = cnvrtc.nvrtcResult.NVRTC_ERROR_INVALID_PROGRAM
    NVRTC_ERROR_INVALID_OPTION = cnvrtc.nvrtcResult.NVRTC_ERROR_INVALID_OPTION
    NVRTC_ERROR_COMPILATION = cnvrtc.nvrtcResult.NVRTC_ERROR_COMPILATION
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = cnvrtc.nvrtcResult.NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = cnvrtc.nvrtcResult.NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = cnvrtc.nvrtcResult.NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = cnvrtc.nvrtcResult.NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID
    NVRTC_ERROR_INTERNAL_ERROR = cnvrtc.nvrtcResult.NVRTC_ERROR_INTERNAL_ERROR


cdef class nvrtcProgram:
    def __cinit__(self, void_ptr init_value = 0, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <cnvrtc.nvrtcProgram *>calloc(1, sizeof(cnvrtc.nvrtcProgram))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(cnvrtc.nvrtcProgram)))
            self._ptr[0] = <cnvrtc.nvrtcProgram>init_value
        else:
            self._ptr_owner = False
            self._ptr = <cnvrtc.nvrtcProgram *>_ptr
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
    def __repr__(self):
        return '<nvrtcProgram ' + str(hex(self.__int__())) + '>'
    def __index__(self):
        return self.__int__()
    def __int__(self):
        return <void_ptr>self._ptr[0]
    def getPtr(self):
        return <void_ptr>self._ptr

@cython.embedsignature(True)
def nvrtcGetErrorString(result not None : nvrtcResult):
    cdef cnvrtc.nvrtcResult cresult = result.value
    err = cnvrtc.nvrtcGetErrorString(cresult)
    return (nvrtcResult.NVRTC_SUCCESS, err)

@cython.embedsignature(True)
def nvrtcVersion():
    cdef int major = 0
    cdef int minor = 0
    err = cnvrtc.nvrtcVersion(&major, &minor)
    return (nvrtcResult(err), major, minor)

@cython.embedsignature(True)
def nvrtcGetNumSupportedArchs():
    cdef int numArchs = 0
    err = cnvrtc.nvrtcGetNumSupportedArchs(&numArchs)
    return (nvrtcResult(err), numArchs)

@cython.embedsignature(True)
def nvrtcGetSupportedArchs():
    cdef vector[int] supportedArchs
    _, s = nvrtcGetNumSupportedArchs()
    supportedArchs.resize(s)
    err = cnvrtc.nvrtcGetSupportedArchs(supportedArchs.data())
    return (nvrtcResult(err), supportedArchs)

@cython.embedsignature(True)
def nvrtcCreateProgram(char* src, char* name, int numHeaders, list headers, list includeNames):
    cdef nvrtcProgram prog = nvrtcProgram()
    if numHeaders > len(headers): raise RuntimeError("List is too small: " + str(len(headers)) + " < " + str(numHeaders))
    if numHeaders > len(includeNames): raise RuntimeError("List is too small: " + str(len(includeNames)) + " < " + str(numHeaders))
    cdef vector[const char*] cheaders = headers
    cdef vector[const char*] cincludeNames = includeNames
    err = cnvrtc.nvrtcCreateProgram(prog._ptr, src, name, numHeaders, cheaders.data(), cincludeNames.data())
    return (nvrtcResult(err), prog)

@cython.embedsignature(True)
def nvrtcDestroyProgram(prog : nvrtcProgram):
    cdef cnvrtc.nvrtcProgram* cprog_ptr = prog._ptr if prog != None else NULL
    err = cnvrtc.nvrtcDestroyProgram(cprog_ptr)
    return (nvrtcResult(err),)

@cython.embedsignature(True)
def nvrtcCompileProgram(prog not None : nvrtcProgram, int numOptions, list options):
    if numOptions > len(options): raise RuntimeError("List is too small: " + str(len(options)) + " < " + str(numOptions))
    cdef vector[const char*] coptions = options
    err = cnvrtc.nvrtcCompileProgram(prog._ptr[0], numOptions, coptions.data())
    return (nvrtcResult(err),)

@cython.embedsignature(True)
def nvrtcGetPTXSize(prog not None : nvrtcProgram):
    cdef size_t ptxSizeRet = 0
    err = cnvrtc.nvrtcGetPTXSize(prog._ptr[0], &ptxSizeRet)
    return (nvrtcResult(err), ptxSizeRet)

@cython.embedsignature(True)
def nvrtcGetPTX(prog not None : nvrtcProgram, char* ptx):
    err = cnvrtc.nvrtcGetPTX(prog._ptr[0], ptx)
    return (nvrtcResult(err),)

@cython.embedsignature(True)
def nvrtcGetCUBINSize(prog not None : nvrtcProgram):
    cdef size_t cubinSizeRet = 0
    err = cnvrtc.nvrtcGetCUBINSize(prog._ptr[0], &cubinSizeRet)
    return (nvrtcResult(err), cubinSizeRet)

@cython.embedsignature(True)
def nvrtcGetCUBIN(prog not None : nvrtcProgram, char* cubin):
    err = cnvrtc.nvrtcGetCUBIN(prog._ptr[0], cubin)
    return (nvrtcResult(err),)

@cython.embedsignature(True)
def nvrtcGetNVVMSize(prog not None : nvrtcProgram):
    cdef size_t nvvmSizeRet = 0
    err = cnvrtc.nvrtcGetNVVMSize(prog._ptr[0], &nvvmSizeRet)
    return (nvrtcResult(err), nvvmSizeRet)

@cython.embedsignature(True)
def nvrtcGetNVVM(prog not None : nvrtcProgram, char* nvvm):
    err = cnvrtc.nvrtcGetNVVM(prog._ptr[0], nvvm)
    return (nvrtcResult(err),)

@cython.embedsignature(True)
def nvrtcGetProgramLogSize(prog not None : nvrtcProgram):
    cdef size_t logSizeRet = 0
    err = cnvrtc.nvrtcGetProgramLogSize(prog._ptr[0], &logSizeRet)
    return (nvrtcResult(err), logSizeRet)

@cython.embedsignature(True)
def nvrtcGetProgramLog(prog not None : nvrtcProgram, char* log):
    err = cnvrtc.nvrtcGetProgramLog(prog._ptr[0], log)
    return (nvrtcResult(err),)

@cython.embedsignature(True)
def nvrtcAddNameExpression(prog not None : nvrtcProgram, char* name_expression):
    err = cnvrtc.nvrtcAddNameExpression(prog._ptr[0], name_expression)
    return (nvrtcResult(err),)

@cython.embedsignature(True)
def nvrtcGetLoweredName(prog not None : nvrtcProgram, char* name_expression):
    cdef const char* lowered_name = NULL
    err = cnvrtc.nvrtcGetLoweredName(prog._ptr[0], name_expression, &lowered_name)
    return (nvrtcResult(err), <bytes>lowered_name)

@cython.embedsignature(True)
def sizeof(objType):
    if objType == nvrtcProgram:
        return sizeof(cnvrtc.nvrtcProgram)
    raise TypeError("Unknown type: " + str(objType))
