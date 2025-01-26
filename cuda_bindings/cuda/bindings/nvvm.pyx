# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated with version 12.6.1. Do not modify it directly.

cimport cython  # NOQA

from ._internal.utils cimport get_nested_resource_ptr, nested_resource

from enum import IntEnum as _IntEnum


###############################################################################
# Enum
###############################################################################

class Result(_IntEnum):
    """See `nvvmResult`."""
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
        status = nvvmVersion(&major, &minor)
    check_status(status)
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
        status = nvvmIRVersion(&major_ir, &minor_ir, &major_dbg, &minor_dbg)
    check_status(status)
    return (major_ir, minor_ir, major_dbg, minor_dbg)


cpdef intptr_t create_program() except? 0:
    """Create a program, and set the value of its handle to ``*prog``.

    Returns:
        intptr_t: NVVM program.

    .. seealso:: `nvvmCreateProgram`
    """
    cdef Program prog
    with nogil:
        status = nvvmCreateProgram(&prog)
    check_status(status)
    return <intptr_t>prog


cpdef add_module_to_program(intptr_t prog, buffer, size_t size, name):
    """Add a module level NVVM IR to a program.

    Args:
        prog (intptr_t): NVVM program.
        buffer (object): NVVM IR module in the bitcode or text representation. It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence of 'char', or
            - a nested Python sequence of ``str``.

        size (size_t): Size of the NVVM IR module.
        name (str): Name of the NVVM IR module. If NULL, "<unnamed>" is used as the name.

    .. seealso:: `nvvmAddModuleToProgram`
    """
    cdef nested_resource[ char ] _buffer_
    get_nested_resource_ptr[char](_buffer_, buffer, <char*>NULL)
    if not isinstance(name, str):
        raise TypeError("name must be a Python str")
    cdef bytes _temp_name_ = (<str>name).encode()
    cdef char* _name_ = _temp_name_
    with nogil:
        status = nvvmAddModuleToProgram(<Program>prog, <const char*>(_buffer_.ptrs.data()), size, <const char*>_name_)
    check_status(status)


cpdef compile_program(intptr_t prog, int num_options, options):
    """Compile the NVVM program.

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
        status = nvvmCompileProgram(<Program>prog, num_options, <const char**>(_options_.ptrs.data()))
    check_status(status)


cpdef verify_program(intptr_t prog, int num_options, options):
    """Verify the NVVM program.

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
        status = nvvmVerifyProgram(<Program>prog, num_options, <const char**>(_options_.ptrs.data()))
    check_status(status)
