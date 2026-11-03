# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.0.1 to 13.4.1. Do not modify it directly.
# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=92a06aa55bccfabc1d5bb1216489bb147dfb71d6e628480128fee207b8f3794f


# <<<< PREAMBLE CONTENT >>>>

cimport cpython as _cyb_cpython
from libc.stdint cimport intptr_t

from cuda.bindings._internal._fast_enum import FastEnum as _cyb_FastEnum

cdef intptr_t _cyb_get_buffer_pointer(buf, Py_ssize_t size, readonly=True) except?-1:
    cdef intptr_t ptr
    cdef int flags = _cyb_cpython.PyBUF_ANY_CONTIGUOUS
    if not readonly:
        flags |= _cyb_cpython.PyBUF_WRITABLE
    cdef int status = -1
    cdef _cyb_cpython.Py_buffer view
    if buf is None:
        ptr = 0
    elif isinstance(buf, int):
        ptr = <intptr_t>buf
    else:
        try:
            status = _cyb_cpython.PyObject_GetBuffer(buf, &view, flags)
            if size != -1:
                assert view.len == size
            assert view.ndim == 1
        except Exception as e:
            adj = "writable " if not readonly else ""
            raise ValueError(
                "buf must be None, a Python int representing the pointer "
                f"address to a valid buffer, or a 1D contiguous {adj}"
                f"buffer, of size {size}"
            ) from e
        else:
            ptr = <intptr_t>view.buf
        finally:
            if status == 0:
                _cyb_cpython.PyBuffer_Release(&view)
    return ptr


# <<<< END OF PREAMBLE CONTENT >>>>

cimport cython  # NOQA

from ._internal.utils cimport (get_nested_resource_ptr,
                               nested_resource)


###############################################################################
# Enum
###############################################################################

class Result(_cyb_FastEnum):
    """
    NVVM API call result code.

    See `nvvmResult`.
    """
    SUCCESS = NVVM_SUCCESS
    ERROR_OUT_OF_MEMORY = NVVM_ERROR_OUT_OF_MEMORY
    ERROR_PROGRAM_CREATION_FAILURE = NVVM_ERROR_PROGRAM_CREATION_FAILURE
    ERROR_IR_VERSION_MISMATCH = NVVM_ERROR_IR_VERSION_MISMATCH
    ERROR_INVALID_INPUT = NVVM_ERROR_INVALID_INPUT
    ERROR_INVALID_PROGRAM = NVVM_ERROR_INVALID_PROGRAM
    ERROR_INVALID_IR = NVVM_ERROR_INVALID_IR
    ERROR_INVALID_OPTION = NVVM_ERROR_INVALID_OPTION
    ERROR_NO_MODULE_IN_PROGRAM = NVVM_ERROR_NO_MODULE_IN_PROGRAM
    ERROR_COMPILATION = NVVM_ERROR_COMPILATION
    ERROR_CANCELLED = NVVM_ERROR_CANCELLED


###############################################################################
# Error handling
###############################################################################

class nvvmError(Exception):

    def __init__(self, status):
        self.status = status
        s = Result(status)
        cdef str err = f"{s.name} ({s.value})"
        super(nvvmError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cdef int check_status(int status) except 1 nogil:
    if status != 0:
        with gil:
            raise nvvmError(status)
    return status


###############################################################################
# Wrapper functions
###############################################################################

cpdef destroy_program(intptr_t prog):
    """Destroy a program.

    Args:
        prog (intptr_t): nvvm prog.

    .. seealso:: `nvvmDestroyProgram`
    """
    cdef Program p = <Program>prog
    with nogil:
        status = nvvmDestroyProgram(&p)
    check_status(status)


cpdef str get_error_string(int result):
    """Get the message string for the given ``nvvmResult`` code.


    Args:
        result (Result): NVVM API result code.

    .. seealso:: `nvvmGetErrorString`
    """
    cdef const char *_output_cstr_
    cdef bytes _output_
    with nogil:
        _output_cstr_ = nvvmGetErrorString(<_Result>result)
    _output_ = _output_cstr_
    return _output_.decode()


cpdef tuple version():
    """Get the NVVM version.


    Returns:
        A 2-tuple containing:
        - int: NVVM major version number.
        - int: NVVM minor version number.

    .. seealso:: `nvvmVersion`
    """
    cdef int major
    cdef int minor
    with nogil:
        __status__ = nvvmVersion(&major, &minor)
    check_status(__status__)
    return (major, minor)


cpdef tuple ir_version():
    """Get the NVVM IR version.


    Returns:
        A 4-tuple containing:
        - int: NVVM IR major version number.
        - int: NVVM IR minor version number.
        - int: NVVM IR debug metadata major version number.
        - int: NVVM IR debug metadata minor version number.

    .. seealso:: `nvvmIRVersion`
    """
    cdef int major_ir
    cdef int minor_ir
    cdef int major_dbg
    cdef int minor_dbg
    with nogil:
        __status__ = nvvmIRVersion(&major_ir, &minor_ir, &major_dbg, &minor_dbg)
    check_status(__status__)
    return (major_ir, minor_ir, major_dbg, minor_dbg)


cpdef intptr_t create_program() except? 0:
    """Create a program, and set the value of its handle to ``*prog``.


    Returns:
        intptr_t: NVVM program.

    .. seealso:: `nvvmCreateProgram`
    """
    cdef Program prog
    with nogil:
        __status__ = nvvmCreateProgram(&prog)
    check_status(__status__)
    return <intptr_t>prog


cpdef add_module_to_program(intptr_t prog, buffer, size_t size, name):
    """Add a module level NVVM IR to a program.

    The ``buffer`` should contain an NVVM IR module. The module should have NVVM IR
    either in the LLVM 7.0.1 bitcode representation or in the LLVM 7.0.1 text
    representation. Support for reading the text representation of NVVM IR is
    deprecated and may be removed in a later version.

    Args:
        prog (intptr_t): NVVM program.
        buffer (bytes): NVVM IR module in the bitcode or text
            representation.
        size (size_t): Size of the NVVM IR module.
        name (str): Name of the NVVM IR module. If NULL, "<unnamed>" is
            used as the name.

    .. seealso:: `nvvmAddModuleToProgram`
    """
    cdef void* _buffer_ = <void *>_cyb_get_buffer_pointer(buffer, size, readonly=True)
    if not isinstance(name, str):
        raise TypeError("name must be a Python str")
    cdef bytes _temp_name_ = (<str>name).encode()
    cdef char* _name_ = _temp_name_
    with nogil:
        __status__ = nvvmAddModuleToProgram(<Program>prog, <const char*>_buffer_, size, <const char*>_name_)
    check_status(__status__)


cpdef lazy_add_module_to_program(intptr_t prog, buffer, size_t size, name):
    """Add a module level NVVM IR to a program.

    The ``buffer`` should contain an NVVM IR module. The module should have NVVM IR
    in the LLVM 7.0.1 bitcode representation.
    A module added using this API is lazily loaded - the only symbols loaded are
    those that are required by module(s) loaded using nvvmAddModuleToProgram. It is
    an error for a program to have all modules loaded using this API. Compiler may
    also optimize entities in this module by making them internal to the linked
    NVVM IR module, making them eligible for other optimizations. Due to these
    optimizations, this API to load a module is more efficient and should be used
    where possible.

    Args:
        prog (intptr_t): NVVM program.
        buffer (bytes): NVVM IR module in the bitcode representation.
        size (size_t): Size of the NVVM IR module.
        name (str): Name of the NVVM IR module. If NULL, "<unnamed>" is
            used as the name.

    .. seealso:: `nvvmLazyAddModuleToProgram`
    """
    cdef void* _buffer_ = <void *>_cyb_get_buffer_pointer(buffer, size, readonly=True)
    if not isinstance(name, str):
        raise TypeError("name must be a Python str")
    cdef bytes _temp_name_ = (<str>name).encode()
    cdef char* _name_ = _temp_name_
    with nogil:
        __status__ = nvvmLazyAddModuleToProgram(<Program>prog, <const char*>_buffer_, size, <const char*>_name_)
    check_status(__status__)


cpdef compile_program(intptr_t prog, int num_options, options):
    """Compile the NVVM program.

    The NVVM IR modules in the program will be linked at the IR level. The linked
    IR program is compiled to PTX.
    The target datalayout in the linked IR program is used to determine the address
    size (32bit vs 64bit).
    The valid compiler options are:.

    - -g (enable generation of full debugging information). Full debug support is
      only valid with '-opt=0'. Debug support requires the input module to utilize
      NVVM IR Debug Metadata. Line number (line info) only generation is also
      enabled via NVVM IR Debug Metadata, there is no specific libNVVM API flag for
      that case.
    - -opt=.
    - 0 (disable optimizations).
    - 3 (default, enable optimizations).
    - -arch=.
    - compute_75 (default).
    - compute_80.
    - compute_87.
    - compute_89.
    - compute_90.
    - compute_90a.
    - compute_100.
    - compute_100a.
    - compute_100f.
    - compute_103.
    - compute_103a.
    - compute_103f.
    - compute_110.
    - compute_110a.
    - compute_110f.
    - compute_120.
    - compute_120a.
    - compute_120f.
    - compute_121.
    - compute_121a.
    - compute_121f.
    - -ftz=.
    - 0 (default, preserve denormal values, when performing single-precision
      floating-point operations).
    - 1 (flush denormal values to zero, when performing single-precision
      floating-point operations).
    - -prec-sqrt=.
    - 0 (use a faster approximation for single-precision floating-point square
      root).
    - 1 (default, use IEEE round-to-nearest mode for single-precision floating-
      point square root).
    - -prec-div=.
    - 0 (use a faster approximation for single-precision floating-point division
      and reciprocals).
    - 1 (default, use IEEE round-to-nearest mode for single-precision floating-
      point division and reciprocals).
    - -fma=.
    - 0 (disable FMA contraction).
    - 1 (default, enable FMA contraction).
    - -jump-table-density=[0-101] Specify the case density percentage in switch
      statements, and use it as a minimal threshold to determine whether jump
      table(brx.idx instruction) will be used to implement a switch statement.
      Default value is 101. The percentage ranges from 0 to 101 inclusively.
    - -gen-lto (Generate LTO IR instead of PTX).
    - -ptx-version-target=[86-94] Specify the target PTX version as (MAJOR VERSION
      \* 10 + MINOR VERSION). This is supported only for Blackwell and later
      architectures (compute capability compute_100 or greater). Using this option
      implies that the operations in incoming ``prog`` are compliant with targeted
      PTX version. If not set, highest available PTX version is used.

    Args:
        prog (intptr_t): NVVM program.
        num_options (int): Number of compiler ``options`` passed.
        options (object): Compiler options in the form of C string array. It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence of 'char', or
            - a nested Python sequence of ``str``.

    .. seealso:: `nvvmCompileProgram`
    """
    cdef nested_resource[ char ] _options_
    get_nested_resource_ptr[char](_options_, options, <char*>NULL)
    with nogil:
        __status__ = nvvmCompileProgram(<Program>prog, num_options, <const char**>(_options_.ptrs.data()))
    check_status(__status__)


cpdef verify_program(intptr_t prog, int num_options, options):
    """Verify the NVVM program.

    The valid compiler options are:.
    Same as for :func:`compile_program`.

    Args:
        prog (intptr_t): NVVM program.
        num_options (int): Number of compiler ``options`` passed.
        options (object): Compiler options in the form of C string array. It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence of 'char', or
            - a nested Python sequence of ``str``.

    .. seealso:: `nvvmVerifyProgram`
    """
    cdef nested_resource[ char ] _options_
    get_nested_resource_ptr[char](_options_, options, <char*>NULL)
    with nogil:
        __status__ = nvvmVerifyProgram(<Program>prog, num_options, <const char**>(_options_.ptrs.data()))
    check_status(__status__)


cpdef size_t get_compiled_result_size(intptr_t prog) except? 0:
    """Get the size of the compiled result.


    Args:
        prog (intptr_t): NVVM program.

    Returns:
        size_t: Size of the compiled result (including the trailing NULL).

    .. seealso:: `nvvmGetCompiledResultSize`
    """
    cdef size_t buffer_size_ret
    with nogil:
        __status__ = nvvmGetCompiledResultSize(<Program>prog, &buffer_size_ret)
    check_status(__status__)
    return buffer_size_ret


cpdef get_compiled_result(intptr_t prog, buffer):
    """Get the compiled result.

    The result is stored in the memory pointed to by ``buffer``.

    Args:
        prog (intptr_t): NVVM program.
        buffer (bytes): Compiled result.

    .. seealso:: `nvvmGetCompiledResult`
    """
    cdef void* _buffer_ = <void *>_cyb_get_buffer_pointer(buffer, -1, readonly=False)
    with nogil:
        __status__ = nvvmGetCompiledResult(<Program>prog, <char*>_buffer_)
    check_status(__status__)


cpdef size_t get_program_log_size(intptr_t prog) except? 0:
    """Get the Size of Compiler/Verifier Message.

    The size of the message string (including the trailing NULL) is stored into
    ``buffer_size_ret`` when the return value is NVVM_SUCCESS.

    Args:
        prog (intptr_t): NVVM program.

    Returns:
        size_t: Size of the compilation/verification log (including the
            trailing NULL).

    .. seealso:: `nvvmGetProgramLogSize`
    """
    cdef size_t buffer_size_ret
    with nogil:
        __status__ = nvvmGetProgramLogSize(<Program>prog, &buffer_size_ret)
    check_status(__status__)
    return buffer_size_ret


cpdef get_program_log(intptr_t prog, buffer):
    """Get the Compiler/Verifier Message.

    The NULL terminated message string is stored in the memory pointed to by
    ``buffer`` when the return value is NVVM_SUCCESS.

    Args:
        prog (intptr_t): NVVM program.
        buffer (bytes): Compilation/Verification log.

    .. seealso:: `nvvmGetProgramLog`
    """
    cdef void* _buffer_ = <void *>_cyb_get_buffer_pointer(buffer, -1, readonly=False)
    with nogil:
        __status__ = nvvmGetProgramLog(<Program>prog, <char*>_buffer_)
    check_status(__status__)


cpdef int llvm_version(arch) except? 0:
    """Get the LLVM IR version guaranteed to be supported by NVVM.

    The valid arch strings are the ones supported by :func:`compile_program`.

    Args:
        arch (str): Architecture string.

    Returns:
        int: IR version number.

    .. seealso:: `nvvmLLVMVersion`
    """
    if not isinstance(arch, str):
        raise TypeError("arch must be a Python str")
    cdef bytes _temp_arch_ = (<str>arch).encode()
    cdef char* _arch_ = _temp_arch_
    cdef int major
    with nogil:
        __status__ = nvvmLLVMVersion(<const char*>_arch_, &major)
    check_status(__status__)
    return major
del _cyb_FastEnum
