# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


cimport cpython
cimport libc.stdint


cpdef int _check_driver_error(error) except?-1
cpdef int _check_runtime_error(error) except?-1
cpdef int _check_nvrtc_error(error) except?-1
cpdef check_or_create_options(type cls, options, str options_description=*, bint keep_none=*)


cdef inline tuple carray_int64_t_to_tuple(libc.stdint.int64_t *ptr, int length):
    result = cpython.PyTuple_New(length)
    for i in range(length):
        cpython.PyTuple_SET_ITEM(result, i, cpython.PyLong_FromLongLong(ptr[i]))
    return result
