# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.0.1 to 12.4.1. Do not modify it directly.

cimport cython  # NOQA

from enum import IntEnum as _IntEnum


###############################################################################
# Enum
###############################################################################




###############################################################################
# Error handling
###############################################################################

cdef dict STATUS={
    NVJITLINK_SUCCESS                   : 'NVJITLINK_SUCCESS',
    NVJITLINK_ERROR_UNRECOGNIZED_OPTION : 'NVJITLINK_ERROR_UNRECOGNIZED_OPTION',
    NVJITLINK_ERROR_MISSING_ARCH        : 'NVJITLINK_ERROR_MISSING_ARCH', // -arch=sm_NN option not specified
    NVJITLINK_ERROR_INVALID_INPUT       : 'NVJITLINK_ERROR_INVALID_INPUT',
    NVJITLINK_ERROR_PTX_COMPILE         : 'NVJITLINK_ERROR_PTX_COMPILE',
    NVJITLINK_ERROR_NVVM_COMPILE        : 'NVJITLINK_ERROR_NVVM_COMPILE',
    NVJITLINK_ERROR_INTERNAL            : 'NVJITLINK_ERROR_INTERNAL',
    NVJITLINK_ERROR_THREADPOOL          : 'NVJITLINK_ERROR_THREADPOOL',
    NVJITLINK_ERROR_UNRECOGNIZED_INPUT  : 'NVJITLINK_ERROR_UNRECOGNIZED_INPUT',
    NVJITLINK_ERROR_NULL_INPUT          : 'NVJITLINK_ERROR_NULL_INPUT',
    NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS: 'NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS',
    NVJITLINK_ERROR_INCORRECT_INPUT_TYPE: 'NVJITLINK_ERROR_INCORRECT_INPUT_TYPE',
    NVJITLINK_ERROR_ARCH_MISMATCH       : 'NVJITLINK_ERROR_ARCH_MISMATCH',
    NVJITLINK_ERROR_OUTDATED_LIBRARY    : 'NVJITLINK_ERROR_OUTDATED_LIBRARY',
    NVJITLINK_ERROR_MISSING_FATBIN      : 'NVJITLINK_ERROR_MISSING_FATBIN'
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
        status = nvJitLinkCreate(<nvJitLinkHandle*>handle, num_options, <const char**>options)
        _check_status(status)


cpdef destroy(intptr_t handle):
    with nogil:
        status = nvJitLinkDestroy(<nvJitLinkHandle*>handle)
        _check_status(status)


cpdef add_data(nvJitLinkHandle handle, nvJitLinkInputType input_type, intptr_t data, size_t size, intptr_t name):
    with nogil:
        status = nvJitLinkAddData(handle, input_type, <const void*>data, size, <const char*>name)
        _check_status(status)


cpdef add_file(nvJitLinkHandle handle, nvJitLinkInputType input_type, intptr_t file_name):
    with nogil:
        status = nvJitLinkAddFile(handle, input_type, <const char*>file_name)
        _check_status(status)


cpdef complete(nvJitLinkHandle handle):
    with nogil:
        status = nvJitLinkComplete(handle)
        _check_status(status)


cpdef get_linked_cubin_size(nvJitLinkHandle handle, intptr_t size):
    with nogil:
        status = nvJitLinkGetLinkedCubinSize(handle, <size_t*>size)
        _check_status(status)


cpdef get_linked_cubin(nvJitLinkHandle handle, intptr_t cubin):
    with nogil:
        status = nvJitLinkGetLinkedCubin(handle, <void*>cubin)
        _check_status(status)


cpdef get_linked_ptx_size(nvJitLinkHandle handle, intptr_t size):
    with nogil:
        status = nvJitLinkGetLinkedPtxSize(handle, <size_t*>size)
        _check_status(status)


cpdef get_linked_ptx(nvJitLinkHandle handle, intptr_t ptx):
    with nogil:
        status = nvJitLinkGetLinkedPtx(handle, <char*>ptx)
        _check_status(status)


cpdef get_error_log_size(nvJitLinkHandle handle, intptr_t size):
    with nogil:
        status = nvJitLinkGetErrorLogSize(handle, <size_t*>size)
        _check_status(status)


cpdef get_error_log(nvJitLinkHandle handle, intptr_t log):
    with nogil:
        status = nvJitLinkGetErrorLog(handle, <char*>log)
        _check_status(status)


cpdef get_info_log_size(nvJitLinkHandle handle, intptr_t size):
    with nogil:
        status = nvJitLinkGetInfoLogSize(handle, <size_t*>size)
        _check_status(status)


cpdef get_info_log(nvJitLinkHandle handle, intptr_t log):
    with nogil:
        status = nvJitLinkGetInfoLog(handle, <char*>log)
        _check_status(status)
