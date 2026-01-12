# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

cimport cpython
from cpython.object cimport PyObject
from libc.stdint cimport int64_t, int32_t

from cuda.bindings cimport cydriver


ctypedef fused supported_error_type:
    cydriver.CUresult


ctypedef fused integer_t:
    int64_t
    int32_t


# mimic CU_DEVICE_INVALID
cdef const cydriver.CUcontext CU_CONTEXT_INVALID = <cydriver.CUcontext>(-2)


cdef int HANDLE_RETURN(supported_error_type err) except?-1 nogil


# TODO: stop exposing these within the codebase?
cpdef int _check_driver_error(cydriver.CUresult error) except?-1 nogil
cpdef int _check_runtime_error(error) except?-1
cpdef int _check_nvrtc_error(error) except?-1


cpdef check_or_create_options(type cls, options, str options_description=*, bint keep_none=*)


# Create low-level externs so Cython won't "helpfully" handle reference counting
# for us.  Prefixing with an underscore to distinguish it from the definition in
# cpython.long.
cdef extern from "Python.h":
    PyObject *_PyLong_FromLongLong "PyLong_FromLongLong" (long long val) except NULL
    void _PyTuple_SET_ITEM "PyTuple_SET_ITEM" (object p, Py_ssize_t pos, PyObject *o)


cdef inline tuple carray_integer_t_to_tuple(integer_t *ptr, int length):
    # Construct shape and strides tuples using the Python/C API for speed
    cdef tuple result = cpython.PyTuple_New(length)
    for i in range(length):
        _PyTuple_SET_ITEM(result, i, _PyLong_FromLongLong(ptr[i]))
    return result
