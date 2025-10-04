# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

cimport cpython
from libc.stdint cimport int64_t

from cuda.bindings cimport cydriver


ctypedef fused supported_error_type:
    cydriver.CUresult


# mimic CU_DEVICE_INVALID
cdef cydriver.CUcontext CU_CONTEXT_INVALID = <cydriver.CUcontext>(-2)


cdef cydriver.CUdevice get_device_from_ctx(
    cydriver.CUcontext target_ctx, cydriver.CUcontext curr_ctx) except?cydriver.CU_DEVICE_INVALID nogil


cdef int HANDLE_RETURN(supported_error_type err) except?-1 nogil


# TODO: stop exposing these within the codebase?
cpdef int _check_driver_error(cydriver.CUresult error) except?-1 nogil
cpdef int _check_runtime_error(error) except?-1
cpdef int _check_nvrtc_error(error) except?-1


cpdef check_or_create_options(type cls, options, str options_description=*, bint keep_none=*)


cdef inline tuple carray_int64_t_to_tuple(int64_t *ptr, int length):
    # Construct shape and strides tuples using the Python/C API for speed
    result = cpython.PyTuple_New(length)
    for i in range(length):
        cpython.PyTuple_SET_ITEM(result, i, cpython.PyLong_FromLongLong(ptr[i]))
    return result
