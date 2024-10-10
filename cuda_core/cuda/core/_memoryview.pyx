# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

cimport cython

from ._dlpack cimport *

import functools
from typing import Any, Optional

from cuda import cuda
import numpy

from cuda.core._utils import handle_return


# TODO(leofang): support NumPy structured dtypes


@cython.dataclasses.dataclass
cdef class StridedMemoryView:

    # TODO: switch to use Cython's cdef typing?
    ptr: int = None
    shape: tuple = None
    strides: tuple = None  # in counts, not bytes
    dtype: numpy.dtype = None
    device_id: int = None  # -1 for CPU
    device_accessible: bool = None
    readonly: bool = None
    obj: Any = None

    def __init__(self, obj=None, stream_ptr=None):
        if obj is not None:
            # populate self's attributes
            if check_has_dlpack(obj):
                view_as_dlpack(obj, stream_ptr, self)
            else:
                view_as_cai(obj, stream_ptr, self)
        else:
            # default construct
            pass

    def __repr__(self):
        return (f"StridedMemoryView(ptr={self.ptr},\n"
              + f"                  shape={self.shape},\n"
              + f"                  strides={self.strides},\n"
              + f"                  dtype={get_simple_repr(self.dtype)},\n"
              + f"                  device_id={self.device_id},\n"
              + f"                  device_accessible={self.device_accessible},\n"
              + f"                  readonly={self.readonly},\n"
              + f"                  obj={get_simple_repr(self.obj)})")


cdef str get_simple_repr(obj):
    # TODO: better handling in np.dtype objects
    cdef object obj_class
    cdef str obj_repr
    if isinstance(obj, type):
        obj_class = obj
    else:
        obj_class = obj.__class__
    if obj_class.__module__ in (None, "builtins"):
        obj_repr = obj_class.__name__
    else:
        obj_repr = f"{obj_class.__module__}.{obj_class.__name__}"
    return obj_repr


cdef bint check_has_dlpack(obj) except*:
    cdef bint has_dlpack
    if hasattr(obj, "__dlpack__") and hasattr(obj, "__dlpack_device__"):
        has_dlpack = True
    elif hasattr(obj, "__cuda_array_interface__"):
        has_dlpack = False
    else:
        raise RuntimeError(
            "the input object does not support any data exchange protocol")
    return has_dlpack


cdef class _StridedMemoryViewProxy:

    cdef:
        object obj
        bint has_dlpack

    def __init__(self, obj):
        self.obj = obj
        self.has_dlpack = check_has_dlpack(obj)

    cpdef StridedMemoryView view(self, stream_ptr=None):
        if self.has_dlpack:
            return view_as_dlpack(self.obj, stream_ptr)
        else:
            return view_as_cai(self.obj, stream_ptr)


cdef StridedMemoryView view_as_dlpack(obj, stream_ptr, view=None):
    cdef int dldevice, device_id, i
    cdef bint device_accessible, versioned, is_readonly
    dldevice, device_id = obj.__dlpack_device__()
    if dldevice == _kDLCPU:
        device_accessible = False
        assert device_id == 0
        if stream_ptr is None:
            raise BufferError("stream=None is ambiguous with view()")
        elif stream_ptr == -1:
            stream_ptr = None
    elif dldevice == _kDLCUDA:
        device_accessible = True
        # no need to check other stream values, it's a pass-through
        if stream_ptr is None:
            raise BufferError("stream=None is ambiguous with view()")
    elif dldevice == _kDLCUDAHost:
        device_accessible = True
        assert device_id == 0
        # just do a pass-through without any checks, as pinned memory can be
        # accessed on both host and device
    elif dldevice == _kDLCUDAManaged:
        device_accessible = True
        # just do a pass-through without any checks, as managed memory can be
        # accessed on both host and device
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

    cdef StridedMemoryView buf = StridedMemoryView() if view is None else view
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

    # We want the dtype object not just the type object
    return numpy.dtype(np_dtype)


cdef StridedMemoryView view_as_cai(obj, stream_ptr, view=None):
    cdef dict cai_data = obj.__cuda_array_interface__
    if cai_data["version"] < 3:
        raise BufferError("only CUDA Array Interface v3 or above is supported")
    if cai_data.get("mask") is not None:
        raise BufferError("mask is not supported")
    if stream_ptr is None:
        raise BufferError("stream=None is ambiguous with view()")

    cdef StridedMemoryView buf = StridedMemoryView() if view is None else view
    buf.obj = obj
    buf.ptr, buf.readonly = cai_data["data"]
    buf.shape = cai_data["shape"]
    # TODO: this only works for built-in numeric types
    buf.dtype = numpy.dtype(cai_data["typestr"])
    buf.strides = cai_data.get("strides")
    if buf.strides is not None:
        # convert to counts
        buf.strides = tuple(s // buf.dtype.itemsize for s in buf.strides)
    buf.device_accessible = True
    buf.device_id = handle_return(
        cuda.cuPointerGetAttribute(
            cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
            buf.ptr))

    cdef intptr_t producer_s, consumer_s
    stream = cai_data.get("stream")
    if stream is not None:
        producer_s = <intptr_t>(stream)
        consumer_s = <intptr_t>(stream_ptr)
        assert producer_s > 0
        # establish stream order
        if producer_s != consumer_s:
            e = handle_return(cuda.cuEventCreate(
                cuda.CUevent_flags.CU_EVENT_DISABLE_TIMING))
            handle_return(cuda.cuEventRecord(e, producer_s))
            handle_return(cuda.cuStreamWaitEvent(consumer_s, e, 0))
            handle_return(cuda.cuEventDestroy(e))

    return buf


def viewable(tuple arg_indices):
    def wrapped_func_with_indices(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            args = list(args)
            cdef int idx
            for idx in arg_indices:
                args[idx] = _StridedMemoryViewProxy(args[idx])
            return func(*args, **kwargs)
        return wrapped_func
    return wrapped_func_with_indices
