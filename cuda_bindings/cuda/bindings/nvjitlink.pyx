# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.0.76 to 12.6.77. Do not modify it directly.

cimport cython  # NOQA

from enum import IntEnum as _IntEnum


###############################################################################
# Enum
###############################################################################

class NvJitLinkResult(_IntEnum):
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

class NvJitLinkInputType(_IntEnum):
    """See `nvJitLinkInputType`."""
    INPUT_NONE = NVJITLINK_INPUT_NONE
    INPUT_CUBIN = NVJITLINK_INPUT_CUBIN
    INPUT_PTX = NVJITLINK_INPUT_PTX
    INPUT_LTOIR = NVJITLINK_INPUT_LTOIR
    INPUT_FATBIN = NVJITLINK_INPUT_FATBIN
    INPUT_OBJECT = NVJITLINK_INPUT_OBJECT
    INPUT_LIBRARY = NVJITLINK_INPUT_LIBRARY
    INPUT_INDEX = NVJITLINK_INPUT_INDEX
    INPUT_ANY = NVJITLINK_INPUT_ANY


###############################################################################
# Error handling
###############################################################################

cdef dict STATUS={
    NVJITLINK_SUCCESS                   : 'NVJITLINK_SUCCESS',
    NVJITLINK_ERROR_UNRECOGNIZED_OPTION : 'NVJITLINK_ERROR_UNRECOGNIZED_OPTION',
    NVJITLINK_ERROR_MISSING_ARCH        : 'NVJITLINK_ERROR_MISSING_ARCH',
    NVJITLINK_ERROR_INVALID_INPUT       : 'NVJITLINK_ERROR_INVALID_INPUT',
    NVJITLINK_ERROR_PTX_COMPILE         : 'NVJITLINK_ERROR_PTX_COMPILE',
    NVJITLINK_ERROR_NVVM_COMPILE        : 'NVJITLINK_ERROR_NVVM_COMPILE',
    NVJITLINK_ERROR_INTERNAL            : 'NVJITLINK_ERROR_INTERNAL'
}

class nvJitLinkError(Exception):

    def __init__(self, status):
        self.status = status
        cdef str err = STATUS[status]
        super(nvJitLinkError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cdef inline void check_status(int status) nogil:
    if status != 0:
        with gil:
            raise nvJitLinkError(status)


###############################################################################
# Wrapper functions
###############################################################################

cpdef create(intptr_t handle, uint32_t num_options, intptr_t options):
    with nogil:
        status = nvJitLinkCreate(<Handle*>handle, num_options, <const char**>options)
    check_status(status)


cpdef destroy(intptr_t handle):
    with nogil:
        status = nvJitLinkDestroy(<Handle*>handle)
    check_status(status)


cpdef add_data(intptr_t handle, int input_type, intptr_t data, size_t size, intptr_t name):
    with nogil:
        status = nvJitLinkAddData(<Handle>handle, <_NvJitLinkInputType>input_type, <const void*>data, size, <const char*>name)
    check_status(status)


cpdef add_file(intptr_t handle, int input_type, intptr_t file_name):
    with nogil:
        status = nvJitLinkAddFile(<Handle>handle, <_NvJitLinkInputType>input_type, <const char*>file_name)
    check_status(status)


cpdef complete(intptr_t handle):
    with nogil:
        status = nvJitLinkComplete(<Handle>handle)
    check_status(status)


cpdef get_linked_cubin_size(intptr_t handle, intptr_t size):
    with nogil:
        status = nvJitLinkGetLinkedCubinSize(<Handle>handle, <size_t*>size)
    check_status(status)


cpdef get_linked_cubin(intptr_t handle, intptr_t cubin):
    with nogil:
        status = nvJitLinkGetLinkedCubin(<Handle>handle, <void*>cubin)
    check_status(status)


cpdef get_linked_ptx_size(intptr_t handle, intptr_t size):
    with nogil:
        status = nvJitLinkGetLinkedPtxSize(<Handle>handle, <size_t*>size)
    check_status(status)


cpdef get_linked_ptx(intptr_t handle, intptr_t ptx):
    with nogil:
        status = nvJitLinkGetLinkedPtx(<Handle>handle, <char*>ptx)
    check_status(status)


cpdef get_error_log_size(intptr_t handle, intptr_t size):
    with nogil:
        status = nvJitLinkGetErrorLogSize(<Handle>handle, <size_t*>size)
    check_status(status)


cpdef get_error_log(intptr_t handle, intptr_t log):
    with nogil:
        status = nvJitLinkGetErrorLog(<Handle>handle, <char*>log)
    check_status(status)


cpdef get_info_log_size(intptr_t handle, intptr_t size):
    with nogil:
        status = nvJitLinkGetInfoLogSize(<Handle>handle, <size_t*>size)
    check_status(status)


cpdef get_info_log(intptr_t handle, intptr_t log):
    with nogil:
        status = nvJitLinkGetInfoLog(<Handle>handle, <char*>log)
    check_status(status)