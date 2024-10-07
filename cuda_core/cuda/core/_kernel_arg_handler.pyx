# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdint cimport (intptr_t,
                          int8_t, int16_t, int32_t, int64_t,
                          uint8_t, uint16_t, uint32_t, uint64_t,)
from libcpp cimport bool as cpp_bool
from libcpp.complex cimport complex as cpp_complex
from libcpp cimport nullptr
from libcpp cimport vector

import ctypes

# this might be an unnecessary assumption that NumPy does not exist...
try:
    import numpy
except ImportError:
    numpy = None

from cuda.core._memory import Buffer


ctypedef cpp_complex.complex[float] cpp_single_complex
ctypedef cpp_complex.complex[double] cpp_double_complex


ctypedef fused supported_type:
    cpp_bool
    int8_t
    int16_t
    int32_t
    int64_t
    uint8_t
    uint16_t
    uint32_t
    uint64_t
    float
    double
    intptr_t
    cpp_single_complex
    cpp_double_complex


# TODO: cache ctypes/numpy type objects to avoid attribute access


# limitation due to cython/cython#534
ctypedef void* voidptr


# Cython can't infer the overload without at least one input argument with fused type
cdef inline int prepare_arg(
        vector.vector[void*]& data,
        vector.vector[void*]& data_addresses,
        arg,  # important: keep it a Python object and don't cast
        const size_t idx,
        const supported_type* __unused=NULL) except -1:
    cdef void* ptr = PyMem_Malloc(sizeof(supported_type))
    # note: this should also work once ctypes has complex support:
    # python/cpython#121248
    if supported_type is cpp_single_complex:
        (<supported_type*>ptr)[0] = cpp_complex.complex[float](arg.real, arg.imag)
    elif supported_type is cpp_double_complex:
        (<supported_type*>ptr)[0] = cpp_complex.complex[double](arg.real, arg.imag)
    else:
        (<supported_type*>ptr)[0] = <supported_type>(arg)
    data_addresses[idx] = ptr  # take the address to the scalar
    data[idx] = ptr  # for later dealloc
    return 0


cdef inline int prepare_ctypes_arg(
        vector.vector[void*]& data,
        vector.vector[void*]& data_addresses,
        arg,
        const size_t idx) except -1:
    if isinstance(arg, ctypes.c_bool):
        return prepare_arg[cpp_bool](data, data_addresses, arg.value, idx)
    elif isinstance(arg, ctypes.c_int8):
        return prepare_arg[int8_t](data, data_addresses, arg.value, idx)
    elif isinstance(arg, ctypes.c_int16):
        return prepare_arg[int16_t](data, data_addresses, arg.value, idx)
    elif isinstance(arg, ctypes.c_int32):
        return prepare_arg[int32_t](data, data_addresses, arg.value, idx)
    elif isinstance(arg, ctypes.c_int64):
        return prepare_arg[int64_t](data, data_addresses, arg.value, idx)
    elif isinstance(arg, ctypes.c_uint8):
        return prepare_arg[uint8_t](data, data_addresses, arg.value, idx)
    elif isinstance(arg, ctypes.c_uint16):
        return prepare_arg[uint16_t](data, data_addresses, arg.value, idx)
    elif isinstance(arg, ctypes.c_uint32):
        return prepare_arg[uint32_t](data, data_addresses, arg.value, idx)
    elif isinstance(arg, ctypes.c_uint64):
        return prepare_arg[uint64_t](data, data_addresses, arg.value, idx)
    elif isinstance(arg, ctypes.c_float):
        return prepare_arg[float](data, data_addresses, arg.value, idx)
    elif isinstance(arg, ctypes.c_double):
        return prepare_arg[double](data, data_addresses, arg.value, idx)
    else:
        return 1


cdef inline int prepare_numpy_arg(
        vector.vector[void*]& data,
        vector.vector[void*]& data_addresses,
        arg,
        const size_t idx) except -1:
    if not numpy:
        return 1

    if isinstance(arg, numpy.bool_):
        return prepare_arg[cpp_bool](data, data_addresses, arg, idx)
    elif isinstance(arg, numpy.int8):
        return prepare_arg[int8_t](data, data_addresses, arg, idx)
    elif isinstance(arg, numpy.int16):
        return prepare_arg[int16_t](data, data_addresses, arg, idx)
    elif isinstance(arg, numpy.int32):
        return prepare_arg[int32_t](data, data_addresses, arg, idx)
    elif isinstance(arg, numpy.int64):
        return prepare_arg[int64_t](data, data_addresses, arg, idx)
    elif isinstance(arg, numpy.uint8):
        return prepare_arg[uint8_t](data, data_addresses, arg, idx)
    elif isinstance(arg, numpy.uint16):
        return prepare_arg[uint16_t](data, data_addresses, arg, idx)
    elif isinstance(arg, numpy.uint32):
        return prepare_arg[uint32_t](data, data_addresses, arg, idx)
    elif isinstance(arg, numpy.uint64):
        return prepare_arg[uint64_t](data, data_addresses, arg, idx)
    elif isinstance(arg, numpy.float16):
        # use int16 as a proxy
        return prepare_arg[int16_t](data, data_addresses, arg, idx)
    elif isinstance(arg, numpy.float32):
        return prepare_arg[float](data, data_addresses, arg, idx)
    elif isinstance(arg, numpy.float64):
        return prepare_arg[double](data, data_addresses, arg, idx)
    elif isinstance(arg, numpy.complex64):
        return prepare_arg[cpp_single_complex](data, data_addresses, arg, idx)
    elif isinstance(arg, numpy.complex128):
        return prepare_arg[cpp_double_complex](data, data_addresses, arg, idx)
    else:
        return 1


cdef class ParamHolder:

    cdef:
        vector.vector[void*] data
        vector.vector[void*] data_addresses
        object kernel_args
        readonly intptr_t ptr

    def __init__(self, kernel_args):
        if len(kernel_args) == 0:
            self.ptr = 0
            return

        cdef size_t n_args = len(kernel_args)
        cdef size_t i
        cdef int not_prepared
        self.data = vector.vector[voidptr](n_args, nullptr)
        self.data_addresses = vector.vector[voidptr](n_args)
        for i, arg in enumerate(kernel_args):
            if isinstance(arg, Buffer):
                # we need the address of where the actual buffer address is stored
                self.data_addresses[i] = <void*><intptr_t>(arg._ptr.getPtr())
                continue
            elif isinstance(arg, int):
                # Here's the dilemma: We want to have a fast path to pass in Python
                # integers as pointer addresses, but one could also (mistakenly) pass
                # it with the intention of passing a scalar integer. It's a mistake
                # bacause a Python int is ambiguous (arbitrary width). Our judgement
                # call here is to treat it as a pointer address, without any warning!
                prepare_arg[intptr_t](self.data, self.data_addresses, arg, i)
                continue
            elif isinstance(arg, float):
                prepare_arg[double](self.data, self.data_addresses, arg, i)
                continue
            elif isinstance(arg, complex):
                prepare_arg[cpp_double_complex](self.data, self.data_addresses, arg, i)
                continue
            elif isinstance(arg, bool):
                prepare_arg[cpp_bool](self.data, self.data_addresses, arg, i)
                continue

            not_prepared = prepare_numpy_arg(self.data, self.data_addresses, arg, i)
            if not_prepared != 0:
                not_prepared = prepare_ctypes_arg(self.data, self.data_addresses, arg, i)
            if not_prepared != 0:
                # TODO: support ctypes/numpy struct
                raise TypeError

        self.kernel_args = kernel_args
        self.ptr = <intptr_t>self.data_addresses.data()

    def __dealloc__(self):
        for data in self.data:
            if data:
                PyMem_Free(data)
