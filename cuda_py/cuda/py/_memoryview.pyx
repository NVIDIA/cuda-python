# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

cimport cython

from ._dlpack cimport *

import functools
from typing import Any, Optional

import numpy


@cython.dataclasses.dataclass
cdef class GPUMemoryView:

    # TODO: switch to use Cython's cdef typing?
    ptr: int = None
    shape: tuple = None
    strides: tuple = None  # in counts, not bytes
    dtype: numpy.dtype = None
    device_id: int = None  # -1 for CPU
    device_accessible: bool = None
    readonly: bool = None
    obj: Any = None

    # TODO: implement __repr__ to avoid printing obj's content


cdef class _GPUMemoryViewProxy:

    cdef:
        object obj
        bint has_dlpack

    def __init__(self, obj):
        if hasattr(obj, "__dlpack__") and hasattr(obj, "__dlpack_device__"):
            has_dlpack = True
        elif hasattr(obj, "__cuda_array_interface__"):
            has_dlpack = False
        else:
            raise RuntimeError(
                "the input object does not support any data exchange protocol")
        self.obj = obj
        self.has_dlpack = has_dlpack

    cpdef GPUMemoryView view(self, stream_ptr=None):
        if self.has_dlpack:
            return view_as_dlpack(self.obj, stream_ptr)
        else:
            # TODO: Support CAI
            raise NotImplementedError("TODO")


cdef GPUMemoryView view_as_dlpack(obj, stream_ptr):
    cdef int dldevice, device_id, i
    cdef bint device_accessible, versioned, is_readonly
    dldevice, device_id = obj.__dlpack_device__()
    if dldevice == _kDLCPU:
        device_accessible = False
        assert device_id == 0
        stream_ptr = None
    elif dldevice == _kDLCUDA:
        device_accessible = True
        stream_ptr = -1
    elif dldevice == _kDLCUDAHost:
        device_accessible = True
        assert device_id == 0
        stream_ptr = None
    elif dldevice == _kDLCUDAManaged:
        device_accessible = True
        stream_ptr = -1
    else:
        raise BufferError("device not supported")

    cdef object capsule
    try:
        capsule = obj.__dlpack__(
            stream=stream_ptr,
            max_version=(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION))
        versioned = True
    except TypeError:
        capsule = obj.__dlpack__(
            stream=stream_ptr)
        versioned = False

    cdef void* data = NULL
    if versioned and cpython.PyCapsule_IsValid(
            capsule, DLPACK_VERSIONED_TENSOR_UNUSED_NAME):
        data = cpython.PyCapsule_GetPointer(
            capsule, DLPACK_VERSIONED_TENSOR_UNUSED_NAME)
    elif not versioned and cpython.PyCapsule_IsValid(
            capsule, DLPACK_TENSOR_UNUSED_NAME):
        data = cpython.PyCapsule_GetPointer(
            capsule, DLPACK_TENSOR_UNUSED_NAME)
    else:
        assert False

    cdef DLManagedTensor* dlm_tensor
    cdef DLManagedTensorVersioned* dlm_tensor_ver
    cdef DLTensor* dl_tensor
    if versioned:
        dlm_tensor_ver = <DLManagedTensorVersioned*>data
        dl_tensor = &dlm_tensor_ver.dl_tensor
        is_readonly = bool((dlm_tensor_ver.flags & DLPACK_FLAG_BITMASK_READ_ONLY) != 0)
    else:
        dlm_tensor = <DLManagedTensor*>data
        dl_tensor = &dlm_tensor.dl_tensor
        is_readonly = False

    cdef GPUMemoryView buf = GPUMemoryView()
    buf.ptr = <intptr_t>(dl_tensor.data)
    buf.shape = tuple(int(dl_tensor.shape[i]) for i in range(dl_tensor.ndim))
    if dl_tensor.strides:
        buf.strides = tuple(
            int(dl_tensor.strides[i]) for i in range(dl_tensor.ndim))
    else:
        # C-order
        buf.strides = None
    buf.dtype = dtype_dlpack_to_numpy(&dl_tensor.dtype)
    buf.device_id = device_id
    buf.device_accessible = device_accessible
    buf.readonly = is_readonly
    buf.obj = obj

    cdef const char* used_name = (
        DLPACK_VERSIONED_TENSOR_USED_NAME if versioned else DLPACK_TENSOR_USED_NAME)
    cpython.PyCapsule_SetName(capsule, used_name)

    return buf


cdef object dtype_dlpack_to_numpy(DLDataType* dtype):
    cdef int bits = dtype.bits
    if dtype.lanes != 1:
        # TODO: return a NumPy structured dtype?
        raise NotImplementedError(
            f'vector dtypes (lanes={dtype.lanes}) is not supported')
    if dtype.code == kDLUInt:
        if bits == 8:
            np_dtype = numpy.uint8
        elif bits == 16:
            np_dtype = numpy.uint16
        elif bits == 32:
            np_dtype = numpy.uint32
        elif bits == 64:
            np_dtype = numpy.uint64
        else:
            raise TypeError('uint{} is not supported.'.format(bits))
    elif dtype.code == kDLInt:
        if bits == 8:
            np_dtype = numpy.int8
        elif bits == 16:
            np_dtype = numpy.int16
        elif bits == 32:
            np_dtype = numpy.int32
        elif bits == 64:
            np_dtype = numpy.int64
        else:
            raise TypeError('int{} is not supported.'.format(bits))
    elif dtype.code == kDLFloat:
        if bits == 16:
            np_dtype = numpy.float16
        elif bits == 32:
            np_dtype = numpy.float32
        elif bits == 64:
            np_dtype = numpy.float64
        else:
            raise TypeError('float{} is not supported.'.format(bits))
    elif dtype.code == kDLComplex:
        # TODO(leofang): support complex32
        if bits == 64:
            np_dtype = numpy.complex64
        elif bits == 128:
            np_dtype = numpy.complex128
        else:
            raise TypeError('complex{} is not supported.'.format(bits))
    elif dtype.code == kDLBool:
        if bits == 8:
            np_dtype = numpy.bool_
        else:
            raise TypeError(f'{bits}-bit bool is not supported')
    elif dtype.code == kDLBfloat:
        # TODO(leofang): use ml_dtype.bfloat16?
        raise NotImplementedError('bfloat is not supported yet')
    else:
        raise TypeError('Unsupported dtype. dtype code: {}'.format(dtype.code))

    return np_dtype


def viewable(tuple arg_indices):
    def wrapped_func_with_indices(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            args = list(args)
            cdef int idx
            for idx in arg_indices:
                args[idx] = _GPUMemoryViewProxy(args[idx])
            func(*args, **kwargs)
        return wrapped_func
    return wrapped_func_with_indices
