# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.0.1 to 13.1.1, generator version 0.3.1.dev1322+g646ce84ec. Do not modify it directly.

cimport cython  # NOQA

from ._internal.utils cimport (get_resource_ptr, get_nested_resource_ptr, nested_resource, nullable_unique_ptr,
                               get_buffer_pointer, get_resource_ptrs)

from cuda.bindings._internal._fast_enum import FastEnum as _FastEnum
from libcpp.vector cimport vector


###############################################################################
# Enum
###############################################################################

class Result(_FastEnum):
    """
    The enumerated type nvJitLinkResult defines API call result codes.
    nvJitLink APIs return nvJitLinkResult codes to indicate the result.

    See `nvJitLinkResult`.
    """
    SUCCESS = NVJITLINK_SUCCESS
    ERROR_UNRECOGNIZED_OPTION = (NVJITLINK_ERROR_UNRECOGNIZED_OPTION, 'Unrecognized Option')
    ERROR_MISSING_ARCH = (NVJITLINK_ERROR_MISSING_ARCH, 'Option `-arch=sm_NN` not specified')
    ERROR_INVALID_INPUT = (NVJITLINK_ERROR_INVALID_INPUT, 'Invalid Input')
    ERROR_PTX_COMPILE = (NVJITLINK_ERROR_PTX_COMPILE, 'Issue during PTX Compilation')
    ERROR_NVVM_COMPILE = (NVJITLINK_ERROR_NVVM_COMPILE, 'Issue during NVVM Compilation')
    ERROR_INTERNAL = (NVJITLINK_ERROR_INTERNAL, 'Internal Error')
    ERROR_THREADPOOL = (NVJITLINK_ERROR_THREADPOOL, 'Issue with Thread Pool')
    ERROR_UNRECOGNIZED_INPUT = (NVJITLINK_ERROR_UNRECOGNIZED_INPUT, 'Unrecognized Input')
    ERROR_FINALIZE = (NVJITLINK_ERROR_FINALIZE, 'Finalizer Error')
    ERROR_NULL_INPUT = (NVJITLINK_ERROR_NULL_INPUT, 'Null Input')
    ERROR_INCOMPATIBLE_OPTIONS = (NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS, 'Incompatible Options')
    ERROR_INCORRECT_INPUT_TYPE = (NVJITLINK_ERROR_INCORRECT_INPUT_TYPE, 'Incorrect Input Type')
    ERROR_ARCH_MISMATCH = (NVJITLINK_ERROR_ARCH_MISMATCH, 'Arch Mismatch')
    ERROR_OUTDATED_LIBRARY = (NVJITLINK_ERROR_OUTDATED_LIBRARY, 'Outdated Library')
    ERROR_MISSING_FATBIN = (NVJITLINK_ERROR_MISSING_FATBIN, 'Missing Fatbin')
    ERROR_UNRECOGNIZED_ARCH = (NVJITLINK_ERROR_UNRECOGNIZED_ARCH, 'Unrecognized -arch value')
    ERROR_UNSUPPORTED_ARCH = (NVJITLINK_ERROR_UNSUPPORTED_ARCH, 'Unsupported -arch value')
    ERROR_LTO_NOT_ENABLED = (NVJITLINK_ERROR_LTO_NOT_ENABLED, 'Requires -lto')

class InputType(_FastEnum):
    """
    The enumerated type nvJitLinkInputType defines the kind of inputs that
    can be passed to nvJitLinkAdd* APIs.

    See `nvJitLinkInputType`.
    """
    NONE = (NVJITLINK_INPUT_NONE, 'Error Type')
    CUBIN = (NVJITLINK_INPUT_CUBIN, 'For CUDA Binaries')
    PTX = (NVJITLINK_INPUT_PTX, 'For PTX')
    LTOIR = (NVJITLINK_INPUT_LTOIR, 'For LTO-IR')
    FATBIN = (NVJITLINK_INPUT_FATBIN, 'For Fatbin')
    OBJECT = (NVJITLINK_INPUT_OBJECT, 'For Host Object')
    LIBRARY = (NVJITLINK_INPUT_LIBRARY, 'For Host Library')
    INDEX = (NVJITLINK_INPUT_INDEX, 'For Index File')
    ANY = (NVJITLINK_INPUT_ANY, 'Dynamically chooses from the valid types')


###############################################################################
# Error handling
###############################################################################

class nvJitLinkError(Exception):

    def __init__(self, status):
        self.status = status
        s = Result(status)
        cdef str err = f"{s.name} ({s.value})"
        super(nvJitLinkError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cdef int check_status(int status) except 1 nogil:
    if status != 0:
        with gil:
            raise nvJitLinkError(status)
    return status


###############################################################################
# Wrapper functions
###############################################################################

cpdef destroy(intptr_t handle):
    """nvJitLinkDestroy frees the memory associated with the given handle.

    Args:
        handle (intptr_t): nvJitLink handle.

    .. seealso:: `nvJitLinkDestroy`
    """
    cdef Handle h = <Handle>handle
    with nogil:
        status = nvJitLinkDestroy(&h)
    check_status(status)


cpdef intptr_t create(uint32_t num_options, options) except -1:
    """nvJitLinkCreate creates an instance of nvJitLinkHandle with the given input options, and sets the output parameter ``handle``.

    Args:
        num_options (uint32_t): Number of options passed.
        options (object): Array of size ``num_options`` of option strings. It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence of 'char', or
            - a nested Python sequence of ``str``.


    Returns:
        intptr_t: Address of nvJitLink handle.

    .. seealso:: `nvJitLinkCreate`
    """
    cdef nested_resource[ char ] _options_
    get_nested_resource_ptr[char](_options_, options, <char*>NULL)
    cdef Handle handle
    with nogil:
        __status__ = nvJitLinkCreate(&handle, num_options, <const char**>(_options_.ptrs.data()))
    check_status(__status__)
    return <intptr_t>handle


cpdef add_data(intptr_t handle, int input_type, data, size_t size, name):
    """nvJitLinkAddData adds data image to the link.

    Args:
        handle (intptr_t): nvJitLink handle.
        input_type (InputType): kind of input.
        data (bytes): pointer to data image in memory.
        size (size_t): size of the data.
        name (str): name of input object.

    .. seealso:: `nvJitLinkAddData`
    """
    cdef void* _data_ = get_buffer_pointer(data, size, readonly=True)
    if not isinstance(name, str):
        raise TypeError("name must be a Python str")
    cdef bytes _temp_name_ = (<str>name).encode()
    cdef char* _name_ = _temp_name_
    with nogil:
        __status__ = nvJitLinkAddData(<Handle>handle, <_InputType>input_type, <const void*>_data_, size, <const char*>_name_)
    check_status(__status__)


cpdef add_file(intptr_t handle, int input_type, file_name):
    """nvJitLinkAddFile reads data from file and links it in.

    Args:
        handle (intptr_t): nvJitLink handle.
        input_type (InputType): kind of input.
        file_name (str): name of file.

    .. seealso:: `nvJitLinkAddFile`
    """
    if not isinstance(file_name, str):
        raise TypeError("file_name must be a Python str")
    cdef bytes _temp_file_name_ = (<str>file_name).encode()
    cdef char* _file_name_ = _temp_file_name_
    with nogil:
        __status__ = nvJitLinkAddFile(<Handle>handle, <_InputType>input_type, <const char*>_file_name_)
    check_status(__status__)


cpdef complete(intptr_t handle):
    """nvJitLinkComplete does the actual link.

    Args:
        handle (intptr_t): nvJitLink handle.

    .. seealso:: `nvJitLinkComplete`
    """
    with nogil:
        __status__ = nvJitLinkComplete(<Handle>handle)
    check_status(__status__)


cpdef size_t get_linked_cubin_size(intptr_t handle) except? 0:
    """nvJitLinkGetLinkedCubinSize gets the size of the linked cubin.

    Args:
        handle (intptr_t): nvJitLink handle.

    Returns:
        size_t: Size of the linked cubin.

    .. seealso:: `nvJitLinkGetLinkedCubinSize`
    """
    cdef size_t size
    with nogil:
        __status__ = nvJitLinkGetLinkedCubinSize(<Handle>handle, &size)
    check_status(__status__)
    return size


cpdef get_linked_cubin(intptr_t handle, cubin):
    """nvJitLinkGetLinkedCubin gets the linked cubin.

    Args:
        handle (intptr_t): nvJitLink handle.
        cubin (bytes): The linked cubin.

    .. seealso:: `nvJitLinkGetLinkedCubin`
    """
    cdef void* _cubin_ = get_buffer_pointer(cubin, -1, readonly=False)
    with nogil:
        __status__ = nvJitLinkGetLinkedCubin(<Handle>handle, <void*>_cubin_)
    check_status(__status__)


cpdef size_t get_linked_ptx_size(intptr_t handle) except? 0:
    """nvJitLinkGetLinkedPtxSize gets the size of the linked ptx.

    Args:
        handle (intptr_t): nvJitLink handle.

    Returns:
        size_t: Size of the linked PTX.

    .. seealso:: `nvJitLinkGetLinkedPtxSize`
    """
    cdef size_t size
    with nogil:
        __status__ = nvJitLinkGetLinkedPtxSize(<Handle>handle, &size)
    check_status(__status__)
    return size


cpdef get_linked_ptx(intptr_t handle, ptx):
    """nvJitLinkGetLinkedPtx gets the linked ptx.

    Args:
        handle (intptr_t): nvJitLink handle.
        ptx (bytes): The linked PTX.

    .. seealso:: `nvJitLinkGetLinkedPtx`
    """
    cdef void* _ptx_ = get_buffer_pointer(ptx, -1, readonly=False)
    with nogil:
        __status__ = nvJitLinkGetLinkedPtx(<Handle>handle, <char*>_ptx_)
    check_status(__status__)


cpdef size_t get_error_log_size(intptr_t handle) except? 0:
    """nvJitLinkGetErrorLogSize gets the size of the error log.

    Args:
        handle (intptr_t): nvJitLink handle.

    Returns:
        size_t: Size of the error log.

    .. seealso:: `nvJitLinkGetErrorLogSize`
    """
    cdef size_t size
    with nogil:
        __status__ = nvJitLinkGetErrorLogSize(<Handle>handle, &size)
    check_status(__status__)
    return size


cpdef get_error_log(intptr_t handle, log):
    """nvJitLinkGetErrorLog puts any error messages in the log.

    Args:
        handle (intptr_t): nvJitLink handle.
        log (bytes): The error log.

    .. seealso:: `nvJitLinkGetErrorLog`
    """
    cdef void* _log_ = get_buffer_pointer(log, -1, readonly=False)
    with nogil:
        __status__ = nvJitLinkGetErrorLog(<Handle>handle, <char*>_log_)
    check_status(__status__)


cpdef size_t get_info_log_size(intptr_t handle) except? 0:
    """nvJitLinkGetInfoLogSize gets the size of the info log.

    Args:
        handle (intptr_t): nvJitLink handle.

    Returns:
        size_t: Size of the info log.

    .. seealso:: `nvJitLinkGetInfoLogSize`
    """
    cdef size_t size
    with nogil:
        __status__ = nvJitLinkGetInfoLogSize(<Handle>handle, &size)
    check_status(__status__)
    return size


cpdef get_info_log(intptr_t handle, log):
    """nvJitLinkGetInfoLog puts any info messages in the log.

    Args:
        handle (intptr_t): nvJitLink handle.
        log (bytes): The info log.

    .. seealso:: `nvJitLinkGetInfoLog`
    """
    cdef void* _log_ = get_buffer_pointer(log, -1, readonly=False)
    with nogil:
        __status__ = nvJitLinkGetInfoLog(<Handle>handle, <char*>_log_)
    check_status(__status__)


cpdef tuple version():
    """nvJitLinkVersion returns the current version of nvJitLink.

    Returns:
        A 2-tuple containing:

        - unsigned int: The major version.
        - unsigned int: The minor version.

    .. seealso:: `nvJitLinkVersion`
    """
    cdef unsigned int major
    cdef unsigned int minor
    with nogil:
        __status__ = nvJitLinkVersion(&major, &minor)
    check_status(__status__)
    return (major, minor)
