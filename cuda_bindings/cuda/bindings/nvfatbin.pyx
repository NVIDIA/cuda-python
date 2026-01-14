# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.4.1 to 13.1.0. Do not modify it directly.

cimport cython  # NOQA

from ._internal.utils cimport (get_resource_ptr, get_nested_resource_ptr, nested_resource, nullable_unique_ptr,
                               get_buffer_pointer, get_resource_ptrs)

from enum import IntEnum as _IntEnum
from libcpp.vector cimport vector


###############################################################################
# Enum
###############################################################################

class Result(_IntEnum):
    """See `nvFatbinResult`."""
    SUCCESS = NVFATBIN_SUCCESS
    ERROR_INTERNAL = NVFATBIN_ERROR_INTERNAL
    ERROR_ELF_ARCH_MISMATCH = NVFATBIN_ERROR_ELF_ARCH_MISMATCH
    ERROR_ELF_SIZE_MISMATCH = NVFATBIN_ERROR_ELF_SIZE_MISMATCH
    ERROR_MISSING_PTX_VERSION = NVFATBIN_ERROR_MISSING_PTX_VERSION
    ERROR_NULL_POINTER = NVFATBIN_ERROR_NULL_POINTER
    ERROR_COMPRESSION_FAILED = NVFATBIN_ERROR_COMPRESSION_FAILED
    ERROR_COMPRESSED_SIZE_EXCEEDED = NVFATBIN_ERROR_COMPRESSED_SIZE_EXCEEDED
    ERROR_UNRECOGNIZED_OPTION = NVFATBIN_ERROR_UNRECOGNIZED_OPTION
    ERROR_INVALID_ARCH = NVFATBIN_ERROR_INVALID_ARCH
    ERROR_INVALID_NVVM = NVFATBIN_ERROR_INVALID_NVVM
    ERROR_EMPTY_INPUT = NVFATBIN_ERROR_EMPTY_INPUT
    ERROR_MISSING_PTX_ARCH = NVFATBIN_ERROR_MISSING_PTX_ARCH
    ERROR_PTX_ARCH_MISMATCH = NVFATBIN_ERROR_PTX_ARCH_MISMATCH
    ERROR_MISSING_FATBIN = NVFATBIN_ERROR_MISSING_FATBIN
    ERROR_INVALID_INDEX = NVFATBIN_ERROR_INVALID_INDEX
    ERROR_IDENTIFIER_REUSE = NVFATBIN_ERROR_IDENTIFIER_REUSE
    ERROR_INTERNAL_PTX_OPTION = NVFATBIN_ERROR_INTERNAL_PTX_OPTION


###############################################################################
# Error handling
###############################################################################

class nvFatbinError(Exception):

    def __init__(self, status):
        self.status = status
        s = Result(status)
        cdef str err = f"{s.name} ({s.value})"
        super(nvFatbinError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cdef int check_status(int status) except 1 nogil:
    if status != 0:
        with gil:
            raise nvFatbinError(status)
    return status


###############################################################################
# Wrapper functions
###############################################################################

cpdef destroy(intptr_t handle):
    """nvFatbinDestroy frees the memory associated with the given handle.

    Args:
        handle (intptr_t): nvFatbin handle.

    .. seealso:: `nvFatbinDestroy`
    """
    cdef Handle h = <Handle>handle
    with nogil:
        status = nvFatbinDestroy(&h)
    check_status(status)


cpdef str get_error_string(int result):
    """nvFatbinGetErrorString returns an error description string for each error code.

    Args:
        result (Result): error code.

    .. seealso:: `nvFatbinGetErrorString`
    """
    cdef char* _output_
    cdef bytes _output_bytes_
    _output_ = nvFatbinGetErrorString(<_Result>result)

    if _output_ == NULL:
        return ""

    _output_bytes_ = <bytes>_output_
    return _output_bytes_.decode()


cpdef intptr_t create(options, size_t options_count) except -1:
    """nvFatbinCreate creates a new handle.

    Args:
        options (object): An array of strings, each containing a single option. It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence of 'char', or
            - a nested Python sequence of ``str``.

        options_count (size_t): Number of options.

    Returns:
        intptr_t: Address of nvFatbin handle.

    .. seealso:: `nvFatbinCreate`
    """
    cdef nested_resource[ char ] _options_
    get_nested_resource_ptr[char](_options_, options, <char*>NULL)
    cdef Handle handle_indirect
    with nogil:
        __status__ = nvFatbinCreate(&handle_indirect, <const char**>(_options_.ptrs.data()), options_count)
    check_status(__status__)
    return <intptr_t>handle_indirect


cpdef add_ptx(intptr_t handle, code, size_t size, arch, identifier, options_cmd_line):
    """nvFatbinAddPTX adds PTX to the fatbinary.

    Args:
        handle (intptr_t): nvFatbin handle.
        code (bytes): The PTX code.
        size (size_t): The size of the PTX code.
        arch (str): The numerical architecture that this PTX is for (the XX of any sm_XX, lto_XX, or compute_XX).
        identifier (str): Name of the PTX, useful when extracting the fatbin with tools like cuobjdump.
        options_cmd_line (str): Options used during JIT compilation.

    .. seealso:: `nvFatbinAddPTX`
    """
    cdef void* _code_ = get_buffer_pointer(code, size, readonly=True)
    if not isinstance(arch, str):
        raise TypeError("arch must be a Python str")
    cdef bytes _temp_arch_ = (<str>arch).encode()
    cdef char* _arch_ = _temp_arch_
    if not isinstance(identifier, str):
        raise TypeError("identifier must be a Python str")
    cdef bytes _temp_identifier_ = (<str>identifier).encode()
    cdef char* _identifier_ = _temp_identifier_
    if not isinstance(options_cmd_line, str):
        raise TypeError("options_cmd_line must be a Python str")
    cdef bytes _temp_options_cmd_line_ = (<str>options_cmd_line).encode()
    cdef char* _options_cmd_line_ = _temp_options_cmd_line_
    with nogil:
        __status__ = nvFatbinAddPTX(<Handle>handle, <const char*>_code_, size, <const char*>_arch_, <const char*>_identifier_, <const char*>_options_cmd_line_)
    check_status(__status__)


cpdef add_cubin(intptr_t handle, code, size_t size, arch, identifier):
    """nvFatbinAddCubin adds a CUDA binary to the fatbinary.

    Args:
        handle (intptr_t): nvFatbin handle.
        code (bytes): The cubin.
        size (size_t): The size of the cubin.
        arch (str): The numerical architecture that this cubin is for (the XX of any sm_XX, lto_XX, or compute_XX).
        identifier (str): Name of the cubin, useful when extracting the fatbin with tools like cuobjdump.

    .. seealso:: `nvFatbinAddCubin`
    """
    cdef void* _code_ = get_buffer_pointer(code, size, readonly=True)
    if not isinstance(arch, str):
        raise TypeError("arch must be a Python str")
    cdef bytes _temp_arch_ = (<str>arch).encode()
    cdef char* _arch_ = _temp_arch_
    if not isinstance(identifier, str):
        raise TypeError("identifier must be a Python str")
    cdef bytes _temp_identifier_ = (<str>identifier).encode()
    cdef char* _identifier_ = _temp_identifier_
    with nogil:
        __status__ = nvFatbinAddCubin(<Handle>handle, <const void*>_code_, size, <const char*>_arch_, <const char*>_identifier_)
    check_status(__status__)


cpdef add_ltoir(intptr_t handle, code, size_t size, arch, identifier, options_cmd_line):
    """nvFatbinAddLTOIR adds LTOIR to the fatbinary.

    Args:
        handle (intptr_t): nvFatbin handle.
        code (bytes): The LTOIR code.
        size (size_t): The size of the LTOIR code.
        arch (str): The numerical architecture that this LTOIR is for (the XX of any sm_XX, lto_XX, or compute_XX).
        identifier (str): Name of the LTOIR, useful when extracting the fatbin with tools like cuobjdump.
        options_cmd_line (str): Options used during JIT compilation.

    .. seealso:: `nvFatbinAddLTOIR`
    """
    cdef void* _code_ = get_buffer_pointer(code, size, readonly=True)
    if not isinstance(arch, str):
        raise TypeError("arch must be a Python str")
    cdef bytes _temp_arch_ = (<str>arch).encode()
    cdef char* _arch_ = _temp_arch_
    if not isinstance(identifier, str):
        raise TypeError("identifier must be a Python str")
    cdef bytes _temp_identifier_ = (<str>identifier).encode()
    cdef char* _identifier_ = _temp_identifier_
    if not isinstance(options_cmd_line, str):
        raise TypeError("options_cmd_line must be a Python str")
    cdef bytes _temp_options_cmd_line_ = (<str>options_cmd_line).encode()
    cdef char* _options_cmd_line_ = _temp_options_cmd_line_
    with nogil:
        __status__ = nvFatbinAddLTOIR(<Handle>handle, <const void*>_code_, size, <const char*>_arch_, <const char*>_identifier_, <const char*>_options_cmd_line_)
    check_status(__status__)


cpdef size_t size(intptr_t handle) except? 0:
    """nvFatbinSize returns the fatbinary's size.

    Args:
        handle (intptr_t): nvFatbin handle.

    Returns:
        size_t: The fatbinary's size.

    .. seealso:: `nvFatbinSize`
    """
    cdef size_t size
    with nogil:
        __status__ = nvFatbinSize(<Handle>handle, &size)
    check_status(__status__)
    return size


cpdef get(intptr_t handle, buffer):
    """nvFatbinGet returns the completed fatbinary.

    Args:
        handle (intptr_t): nvFatbin handle.
        buffer (bytes): memory to store fatbinary.

    .. seealso:: `nvFatbinGet`
    """
    cdef void* _buffer_ = get_buffer_pointer(buffer, -1, readonly=False)
    with nogil:
        __status__ = nvFatbinGet(<Handle>handle, <void*>_buffer_)
    check_status(__status__)


cpdef tuple version():
    """nvFatbinVersion returns the current version of nvFatbin.

    Returns:
        A 2-tuple containing:

        - unsigned int: The major version.
        - unsigned int: The minor version.

    .. seealso:: `nvFatbinVersion`
    """
    cdef unsigned int major
    cdef unsigned int minor
    with nogil:
        __status__ = nvFatbinVersion(&major, &minor)
    check_status(__status__)
    return (major, minor)


cpdef add_reloc(intptr_t handle, code, size_t size):
    """nvFatbinAddReloc adds relocatable PTX entries from a host object to the fatbinary.

    Args:
        handle (intptr_t): nvFatbin handle.
        code (bytes): The host object image.
        size (size_t): The size of the host object image code.

    .. seealso:: `nvFatbinAddReloc`
    """
    cdef void* _code_ = get_buffer_pointer(code, size, readonly=True)
    with nogil:
        __status__ = nvFatbinAddReloc(<Handle>handle, <const void*>_code_, size)
    check_status(__status__)


cpdef add_tile_ir(intptr_t handle, code, size_t size, identifier, options_cmd_line):
    """nvFatbinAddTileIR adds Tile IR to the fatbinary.

    Args:
        handle (intptr_t): nvFatbin handle.
        code (bytes): The Tile IR.
        size (size_t): The size of the Tile IR.
        identifier (str): Name of the Tile IR, useful when extracting the fatbin with tools like cuobjdump.
        options_cmd_line (str): Options used during JIT compilation.

    .. seealso:: `nvFatbinAddTileIR`
    """
    cdef void* _code_ = get_buffer_pointer(code, size, readonly=True)
    if not isinstance(identifier, str):
        raise TypeError("identifier must be a Python str")
    cdef bytes _temp_identifier_ = (<str>identifier).encode()
    cdef char* _identifier_ = _temp_identifier_
    if not isinstance(options_cmd_line, str):
        raise TypeError("options_cmd_line must be a Python str")
    cdef bytes _temp_options_cmd_line_ = (<str>options_cmd_line).encode()
    cdef char* _options_cmd_line_ = _temp_options_cmd_line_
    with nogil:
        __status__ = nvFatbinAddTileIR(<Handle>handle, <const void*>_code_, size, <const char*>_identifier_, <const char*>_options_cmd_line_)
    check_status(__status__)
