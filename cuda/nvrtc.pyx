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
from cpython.bytes cimport PyBytes_FromStringAndSize

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
    """
    The enumerated type nvrtcResult defines API call result codes.
    NVRTC API functions return nvrtcResult to indicate the call result.
    """
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
    """


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
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
    def __init__(self, *args, **kwargs):
        pass
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
    """ nvrtcGetErrorString is a helper function that returns a string describing the given nvrtcResult code, e.g., NVRTC_SUCCESS to `"NVRTC_SUCCESS"`. For unrecognized enumeration values, it returns `"NVRTC_ERROR unknown"`.

    Parameters
    ----------
    result : nvrtcResult
        CUDA Runtime Compilation API result code.

    Returns
    -------
    nvrtcResult
        Message string for the given nvrtcResult code.
    None
        None
    """
    cdef cnvrtc.nvrtcResult cresult = result.value
    err = cnvrtc.nvrtcGetErrorString(cresult)
    return (nvrtcResult.NVRTC_SUCCESS, err)

@cython.embedsignature(True)
def nvrtcVersion():
    """ nvrtcVersion sets the output parameters `major` and `minor` with the CUDA Runtime Compilation version number.

    Returns
    -------
    nvrtcResult
        NVRTC_SUCCESS
        NVRTC_ERROR_INVALID_INPUT
    major : int
        CUDA Runtime Compilation major version number.
    minor : int
        CUDA Runtime Compilation minor version number.
    """
    cdef int major = 0
    cdef int minor = 0
    err = cnvrtc.nvrtcVersion(&major, &minor)
    return (nvrtcResult(err), major, minor)

@cython.embedsignature(True)
def nvrtcGetNumSupportedArchs():
    """ nvrtcGetNumSupportedArchs sets the output parameter `numArchs` with the number of architectures supported by NVRTC. This can then be used to pass an array to nvrtcGetSupportedArchs to get the supported architectures.

    Returns
    -------
    nvrtcResult
        NVRTC_SUCCESS
        NVRTC_ERROR_INVALID_INPUT
    numArchs : int
        number of supported architectures.
    """
    cdef int numArchs = 0
    err = cnvrtc.nvrtcGetNumSupportedArchs(&numArchs)
    return (nvrtcResult(err), numArchs)

@cython.embedsignature(True)
def nvrtcGetSupportedArchs():
    """ nvrtcGetSupportedArchs populates the array passed via the output parameter `supportedArchs` with the architectures supported by NVRTC. The array is sorted in the ascending order. The size of the array to be passed can be determined using nvrtcGetNumSupportedArchs.

    Returns
    -------
    nvrtcResult
        NVRTC_SUCCESS
        NVRTC_ERROR_INVALID_INPUT
    supportedArchs : List[Int]
        sorted array of supported architectures.
    """
    cdef vector[int] supportedArchs
    _, s = nvrtcGetNumSupportedArchs()
    supportedArchs.resize(s)
    err = cnvrtc.nvrtcGetSupportedArchs(supportedArchs.data())
    return (nvrtcResult(err), supportedArchs)

@cython.embedsignature(True)
def nvrtcCreateProgram(char* src, char* name, int numHeaders, list headers, list includeNames):
    """ nvrtcCreateProgram creates an instance of nvrtcProgram with the given input parameters, and sets the output parameter `prog` with it.

    Parameters
    ----------
    src : bytes
        CUDA program source.
    name : bytes
        CUDA program name. `name` can be `NULL`; `"default_program"` is
        used when `name` is `NULL` or "".
    numHeaders : int
        Number of headers used. `numHeaders` must be greater than or equal
        to 0.
    headers : list
        Sources of the headers. `headers` can be `NULL` when `numHeaders`
        is 0.
    includeNames : list
        Name of each header by which they can be included in the CUDA
        program source. `includeNames` can be `NULL` when `numHeaders` is
        0.

    Returns
    -------
    nvrtcResult
        NVRTC_SUCCESS
        NVRTC_ERROR_OUT_OF_MEMORY
        NVRTC_ERROR_PROGRAM_CREATION_FAILURE
        NVRTC_ERROR_INVALID_INPUT
        NVRTC_ERROR_INVALID_PROGRAM
    prog : nvrtcProgram
        CUDA Runtime Compilation program.

    See Also
    --------
    nvrtcDestroyProgram
    """
    cdef nvrtcProgram prog = nvrtcProgram()
    if numHeaders > len(headers): raise RuntimeError("List is too small: " + str(len(headers)) + " < " + str(numHeaders))
    if numHeaders > len(includeNames): raise RuntimeError("List is too small: " + str(len(includeNames)) + " < " + str(numHeaders))
    cdef vector[const char*] cheaders = headers
    cdef vector[const char*] cincludeNames = includeNames
    err = cnvrtc.nvrtcCreateProgram(prog._ptr, src, name, numHeaders, cheaders.data(), cincludeNames.data())
    return (nvrtcResult(err), prog)

@cython.embedsignature(True)
def nvrtcDestroyProgram(prog : nvrtcProgram):
    """ nvrtcDestroyProgram destroys the given program.

    Parameters
    ----------
    prog : nvrtcProgram
        CUDA Runtime Compilation program.

    Returns
    -------
    nvrtcResult
        NVRTC_SUCCESS
        NVRTC_ERROR_INVALID_PROGRAM
    None
        None

    See Also
    --------
    nvrtcCreateProgram
    """
    cdef cnvrtc.nvrtcProgram* cprog_ptr = prog._ptr if prog != None else NULL
    err = cnvrtc.nvrtcDestroyProgram(cprog_ptr)
    return (nvrtcResult(err),)

@cython.embedsignature(True)
def nvrtcCompileProgram(prog not None : nvrtcProgram, int numOptions, list options):
    """ nvrtcCompileProgram compiles the given program.

    Parameters
    ----------
    prog : nvrtcProgram
        CUDA Runtime Compilation program.
    numOptions : int
        Number of compiler options passed.
    options : list
        Compiler options in the form of C string array. `options` can be
        `NULL` when `numOptions` is 0.

    Returns
    -------
    nvrtcResult
        NVRTC_SUCCESS
        NVRTC_ERROR_OUT_OF_MEMORY
        NVRTC_ERROR_INVALID_INPUT
        NVRTC_ERROR_INVALID_PROGRAM
        NVRTC_ERROR_INVALID_OPTION
        NVRTC_ERROR_COMPILATION
        NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
    None
        None
    """
    if numOptions > len(options): raise RuntimeError("List is too small: " + str(len(options)) + " < " + str(numOptions))
    cdef vector[const char*] coptions = options
    err = cnvrtc.nvrtcCompileProgram(prog._ptr[0], numOptions, coptions.data())
    return (nvrtcResult(err),)

@cython.embedsignature(True)
def nvrtcGetPTXSize(prog not None : nvrtcProgram):
    """ nvrtcGetPTXSize sets `ptxSizeRet` with the size of the PTX generated by the previous compilation of `prog` (including the trailing `NULL`).

    Parameters
    ----------
    prog : nvrtcProgram
        CUDA Runtime Compilation program.

    Returns
    -------
    nvrtcResult
        NVRTC_SUCCESS
        NVRTC_ERROR_INVALID_INPUT
        NVRTC_ERROR_INVALID_PROGRAM
    ptxSizeRet : int
        Size of the generated PTX (including the trailing `NULL`).

    See Also
    --------
    nvrtcGetPTX
    """
    cdef size_t ptxSizeRet = 0
    err = cnvrtc.nvrtcGetPTXSize(prog._ptr[0], &ptxSizeRet)
    return (nvrtcResult(err), ptxSizeRet)

@cython.embedsignature(True)
def nvrtcGetPTX(prog not None : nvrtcProgram, char* ptx):
    """ nvrtcGetPTX stores the PTX generated by the previous compilation of `prog` in the memory pointed by `ptx`.

    Parameters
    ----------
    prog : nvrtcProgram
        CUDA Runtime Compilation program.
    ptx : bytes
        Compiled result.

    Returns
    -------
    nvrtcResult
        NVRTC_SUCCESS
        NVRTC_ERROR_INVALID_INPUT
        NVRTC_ERROR_INVALID_PROGRAM
    None
        None

    See Also
    --------
    nvrtcGetPTXSize
    """
    err = cnvrtc.nvrtcGetPTX(prog._ptr[0], ptx)
    return (nvrtcResult(err),)

@cython.embedsignature(True)
def nvrtcGetCUBINSize(prog not None : nvrtcProgram):
    """ nvrtcGetCUBINSize sets `cubinSizeRet` with the size of the cubin generated by the previous compilation of `prog`. The value of cubinSizeRet is set to 0 if the value specified to `-arch` is a virtual architecture instead of an actual architecture.

    Parameters
    ----------
    prog : nvrtcProgram
        CUDA Runtime Compilation program.

    Returns
    -------
    nvrtcResult
        NVRTC_SUCCESS
        NVRTC_ERROR_INVALID_INPUT
        NVRTC_ERROR_INVALID_PROGRAM
    cubinSizeRet : int
        Size of the generated cubin.

    See Also
    --------
    nvrtcGetCUBIN
    """
    cdef size_t cubinSizeRet = 0
    err = cnvrtc.nvrtcGetCUBINSize(prog._ptr[0], &cubinSizeRet)
    return (nvrtcResult(err), cubinSizeRet)

@cython.embedsignature(True)
def nvrtcGetCUBIN(prog not None : nvrtcProgram, char* cubin):
    """ nvrtcGetCUBIN stores the cubin generated by the previous compilation of `prog` in the memory pointed by `cubin`. No cubin is available if the value specified to `-arch` is a virtual architecture instead of an actual architecture.

    Parameters
    ----------
    prog : nvrtcProgram
        CUDA Runtime Compilation program.
    cubin : bytes
        Compiled and assembled result.

    Returns
    -------
    nvrtcResult
        NVRTC_SUCCESS
        NVRTC_ERROR_INVALID_INPUT
        NVRTC_ERROR_INVALID_PROGRAM
    None
        None

    See Also
    --------
    nvrtcGetCUBINSize
    """
    err = cnvrtc.nvrtcGetCUBIN(prog._ptr[0], cubin)
    return (nvrtcResult(err),)

@cython.embedsignature(True)
def nvrtcGetNVVMSize(prog not None : nvrtcProgram):
    """ nvrtcGetNVVMSize sets `nvvmSizeRet` with the size of the NVVM generated by the previous compilation of `prog`. The value of nvvmSizeRet is set to 0 if the program was not compiled with `-dlto`.

    Parameters
    ----------
    prog : nvrtcProgram
        CUDA Runtime Compilation program.

    Returns
    -------
    nvrtcResult
        NVRTC_SUCCESS
        NVRTC_ERROR_INVALID_INPUT
        NVRTC_ERROR_INVALID_PROGRAM
    nvvmSizeRet : int
        Size of the generated NVVM.

    See Also
    --------
    nvrtcGetNVVM
    """
    cdef size_t nvvmSizeRet = 0
    err = cnvrtc.nvrtcGetNVVMSize(prog._ptr[0], &nvvmSizeRet)
    return (nvrtcResult(err), nvvmSizeRet)

@cython.embedsignature(True)
def nvrtcGetNVVM(prog not None : nvrtcProgram, char* nvvm):
    """ nvrtcGetNVVM stores the NVVM generated by the previous compilation of `prog` in the memory pointed by `nvvm`. The program must have been compiled with -dlto, otherwise will return an error.

    Parameters
    ----------
    prog : nvrtcProgram
        CUDA Runtime Compilation program.
    nvvm : bytes
        Compiled result.

    Returns
    -------
    nvrtcResult
        NVRTC_SUCCESS
        NVRTC_ERROR_INVALID_INPUT
        NVRTC_ERROR_INVALID_PROGRAM
    None
        None

    See Also
    --------
    nvrtcGetNVVMSize
    """
    err = cnvrtc.nvrtcGetNVVM(prog._ptr[0], nvvm)
    return (nvrtcResult(err),)

@cython.embedsignature(True)
def nvrtcGetProgramLogSize(prog not None : nvrtcProgram):
    """ nvrtcGetProgramLogSize sets `logSizeRet` with the size of the log generated by the previous compilation of `prog` (including the trailing `NULL`).

    Note that compilation log may be generated with warnings and
    informative messages, even when the compilation of `prog` succeeds.

    Parameters
    ----------
    prog : nvrtcProgram
        CUDA Runtime Compilation program.

    Returns
    -------
    nvrtcResult
        NVRTC_SUCCESS
        NVRTC_ERROR_INVALID_INPUT
        NVRTC_ERROR_INVALID_PROGRAM
    logSizeRet : int
        Size of the compilation log (including the trailing `NULL`).

    See Also
    --------
    nvrtcGetProgramLog
    """
    cdef size_t logSizeRet = 0
    err = cnvrtc.nvrtcGetProgramLogSize(prog._ptr[0], &logSizeRet)
    return (nvrtcResult(err), logSizeRet)

@cython.embedsignature(True)
def nvrtcGetProgramLog(prog not None : nvrtcProgram, char* log):
    """ nvrtcGetProgramLog stores the log generated by the previous compilation of `prog` in the memory pointed by `log`.

    Parameters
    ----------
    prog : nvrtcProgram
        CUDA Runtime Compilation program.
    log : bytes
        Compilation log.

    Returns
    -------
    nvrtcResult
        NVRTC_SUCCESS
        NVRTC_ERROR_INVALID_INPUT
        NVRTC_ERROR_INVALID_PROGRAM
    None
        None

    See Also
    --------
    nvrtcGetProgramLogSize
    """
    err = cnvrtc.nvrtcGetProgramLog(prog._ptr[0], log)
    return (nvrtcResult(err),)

@cython.embedsignature(True)
def nvrtcAddNameExpression(prog not None : nvrtcProgram, char* name_expression):
    """ nvrtcAddNameExpression notes the given name expression denoting the address of a global function or device/__constant__ variable.

    The identical name expression string must be provided on a subsequent
    call to nvrtcGetLoweredName to extract the lowered name.

    Parameters
    ----------
    prog : nvrtcProgram
        CUDA Runtime Compilation program.
    name_expression : bytes
        constant expression denoting the address of a global function or
        device/__constant__ variable.

    Returns
    -------
    nvrtcResult
        NVRTC_SUCCESS
        NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
    None
        None

    See Also
    --------
    nvrtcGetLoweredName
    """
    err = cnvrtc.nvrtcAddNameExpression(prog._ptr[0], name_expression)
    return (nvrtcResult(err),)

@cython.embedsignature(True)
def nvrtcGetLoweredName(prog not None : nvrtcProgram, char* name_expression):
    """ nvrtcGetLoweredName extracts the lowered (mangled) name for a global function or device/__constant__ variable, and updates lowered_name to point to it. The memory containing the name is released when the NVRTC program is destroyed by nvrtcDestroyProgram. The identical name expression must have been previously provided to nvrtcAddNameExpression.

    Parameters
    ----------
    prog : nvrtcProgram
        CUDA Runtime Compilation program.
    name_expression : bytes
        constant expression denoting the address of a global function or
        device/__constant__ variable.

    Returns
    -------
    nvrtcResult
        NVRTC_SUCCESS
        NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
        NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID
    lowered_name : bytes
        initialized by the function to point to a C string containing the
        lowered (mangled) name corresponding to the provided name
        expression.

    See Also
    --------
    nvrtcAddNameExpression
    """
    cdef const char* lowered_name = NULL
    err = cnvrtc.nvrtcGetLoweredName(prog._ptr[0], name_expression, &lowered_name)
    return (nvrtcResult(err), <bytes>lowered_name)

@cython.embedsignature(True)
def sizeof(objType):
    """ Returns the size of provided CUDA Python structure in bytes

    Parameters
    ----------
    objType : Any
        CUDA Python object

    Returns
    -------
    lowered_name : int
        The size of `objType` in bytes
    """
    if objType == nvrtcProgram:
        return sizeof(cnvrtc.nvrtcProgram)
    raise TypeError("Unknown type: " + str(objType))
