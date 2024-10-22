# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.0.1 to 12.6.2. Do not modify it directly.

cimport cython  # NOQA

from ._internal.utils cimport (get_resource_ptr, get_nested_resource_ptr, nested_resource, nullable_unique_ptr,
                               get_buffer_pointer, get_resource_ptrs)

from enum import IntEnum as _IntEnum
from libcpp.vector cimport vector


###############################################################################
# Enum
###############################################################################

class Result(_IntEnum):
    """See `nvJitLinkResult`."""
    SUCCESS = NVJITLINK_SUCCESS
    ERROR_UNRECOGNIZED_OPTION = NVJITLINK_ERROR_UNRECOGNIZED_OPTION
    ERROR_MISSING_ARCH = NVJITLINK_ERROR_MISSING_ARCH
    ERROR_INVALID_INPUT = NVJITLINK_ERROR_INVALID_INPUT
    ERROR_PTX_COMPILE = NVJITLINK_ERROR_PTX_COMPILE
    ERROR_NVVM_COMPILE = NVJITLINK_ERROR_NVVM_COMPILE
    ERROR_INTERNAL = NVJITLINK_ERROR_INTERNAL
    ERROR_THREADPOOL = NVJITLINK_ERROR_THREADPOOL
    ERROR_UNRECOGNIZED_INPUT = NVJITLINK_ERROR_UNRECOGNIZED_INPUT
    ERROR_FINALIZE = NVJITLINK_ERROR_FINALIZE

class InputType(_IntEnum):
    """See `nvJitLinkInputType`."""
    NONE = NVJITLINK_INPUT_NONE
    CUBIN = NVJITLINK_INPUT_CUBIN
    PTX = NVJITLINK_INPUT_PTX
    LTOIR = NVJITLINK_INPUT_LTOIR
    FATBIN = NVJITLINK_INPUT_FATBIN
    OBJECT = NVJITLINK_INPUT_OBJECT
    LIBRARY = NVJITLINK_INPUT_LIBRARY
    INDEX = NVJITLINK_INPUT_INDEX
    ANY = NVJITLINK_INPUT_ANY


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
    cdef nested_resource[ char ] _options_
    get_nested_resource_ptr[char](_options_, options, <char*>NULL)
    cdef Handle handle
    with nogil:
        status = nvJitLinkCreate(&handle, num_options, <const char**>(_options_.ptrs.data()))
    check_status(status)
    return <intptr_t>handle


cpdef add_data(intptr_t handle, int input_type, intptr_t data, size_t size, name):
    if not isinstance(name, str):
        raise TypeError("name must be a Python str")
    cdef bytes _temp_name_ = (<str>name).encode()
    cdef char* _name_ = _temp_name_
    with nogil:
        status = nvJitLinkAddData(<Handle>handle, <_InputType>input_type, <const void*>data, size, <const char*>_name_)
    check_status(status)


cpdef add_file(intptr_t handle, int input_type, file_name):
    if not isinstance(file_name, str):
        raise TypeError("file_name must be a Python str")
    cdef bytes _temp_file_name_ = (<str>file_name).encode()
    cdef char* _file_name_ = _temp_file_name_
    with nogil:
        status = nvJitLinkAddFile(<Handle>handle, <_InputType>input_type, <const char*>_file_name_)
    check_status(status)


cpdef complete(intptr_t handle):
    with nogil:
        status = nvJitLinkComplete(<Handle>handle)
    check_status(status)


cpdef size_t get_linked_cubin_size(intptr_t handle) except? 0:
    cdef size_t size
    with nogil:
        status = nvJitLinkGetLinkedCubinSize(<Handle>handle, &size)
    check_status(status)
    return size


cpdef get_linked_cubin(intptr_t handle, cubin):
    cdef void* _cubin_ = get_buffer_pointer(cubin, -1, readonly=False)
    with nogil:
        status = nvJitLinkGetLinkedCubin(<Handle>handle, <void*>_cubin_)
    check_status(status)


cpdef size_t get_linked_ptx_size(intptr_t handle) except? 0:
    cdef size_t size
    with nogil:
        status = nvJitLinkGetLinkedPtxSize(<Handle>handle, &size)
    check_status(status)
    return size


cpdef get_linked_ptx(intptr_t handle, ptx):
    cdef void* _ptx_ = get_buffer_pointer(ptx, -1, readonly=False)
    with nogil:
        status = nvJitLinkGetLinkedPtx(<Handle>handle, <char*>_ptx_)
    check_status(status)


cpdef size_t get_error_log_size(intptr_t handle) except? 0:
    cdef size_t size
    with nogil:
        status = nvJitLinkGetErrorLogSize(<Handle>handle, &size)
    check_status(status)
    return size


cpdef get_error_log(intptr_t handle, log):
    cdef void* _log_ = get_buffer_pointer(log, -1, readonly=False)
    with nogil:
        status = nvJitLinkGetErrorLog(<Handle>handle, <char*>_log_)
    check_status(status)


cpdef size_t get_info_log_size(intptr_t handle) except? 0:
    cdef size_t size
    with nogil:
        status = nvJitLinkGetInfoLogSize(<Handle>handle, &size)
    check_status(status)
    return size


cpdef get_info_log(intptr_t handle, log):
    cdef void* _log_ = get_buffer_pointer(log, -1, readonly=False)
    with nogil:
        status = nvJitLinkGetInfoLog(<Handle>handle, <char*>_log_)
    check_status(status)


cpdef tuple version():
    cdef unsigned int major
    cdef unsigned int minor
    with nogil:
        status = nvJitLinkVersion(&major, &minor)
    check_status(status)
    return (major, minor)
