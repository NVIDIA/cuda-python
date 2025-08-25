# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

cimport cython

from ._dlpack cimport *

import functools
from typing import Any, Optional

import numpy

from cuda.core.experimental._utils.cuda_utils import handle_return, driver
from cuda.core.experimental._utils cimport cuda_utils


# TODO(leofang): support NumPy structured dtypes


cdef class StridedMemoryView:
    """A dataclass holding metadata of a strided dense array/tensor.

    A :obj:`StridedMemoryView` instance can be created in two ways:

      1. Using the :obj:`args_viewable_as_strided_memory` decorator (recommended)
      2. Explicit construction, see below

    This object supports both DLPack (up to v1.0) and CUDA Array Interface
    (CAI) v3. When wrapping an arbitrary object it will try the DLPack protocol
    first, then the CAI protocol. A :obj:`BufferError` is raised if neither is
    supported.

    Since either way would take a consumer stream, for DLPack it is passed to
    ``obj.__dlpack__()`` as-is (except for :obj:`None`, see below); for CAI, a
    stream order will be established between the consumer stream and the
    producer stream (from ``obj.__cuda_array_interface__()["stream"]``), as if
    ``cudaStreamWaitEvent`` is called by this method.

    To opt-out of the stream ordering operation in either DLPack or CAI,
    please pass ``stream_ptr=-1``. Note that this deviates (on purpose)
    from the semantics of ``obj.__dlpack__(stream=None, ...)`` since ``cuda.core``
    does not encourage using the (legacy) default/null stream, but is
    consistent with the CAI's semantics. For DLPack, ``stream=-1`` will be
    internally passed to ``obj.__dlpack__()`` instead.

    Attributes
    ----------
    ptr : int
        Pointer to the tensor buffer (as a Python `int`).
    shape : tuple
        Shape of the tensor.
    strides : Optional[tuple]
        Strides of the tensor (in **counts**, not bytes).
    dtype: numpy.dtype
        Data type of the tensor.
    device_id : int
        The device ID for where the tensor is located. It is -1 for CPU tensors
        (meaning those only accessible from the host).
    is_device_accessible : bool
        Whether the tensor data can be accessed on the GPU.
    readonly: bool
        Whether the tensor data can be modified in place.
    exporting_obj : Any
        A reference to the original tensor object that is being viewed.

    Parameters
    ----------
    obj : Any
        Any objects that supports either DLPack (up to v1.0) or CUDA Array
        Interface (v3).
    stream_ptr: int
        The pointer address (as Python `int`) to the **consumer** stream.
        Stream ordering will be properly established unless ``-1`` is passed.
    """
    cdef readonly:
        intptr_t ptr
        int device_id
        bint is_device_accessible
        bint readonly
        object exporting_obj

    # If using dlpack, this is a strong reference to the result of
    # obj.__dlpack__() so we can lazily create shape and strides from
    # it later.  If using CAI, this is a reference to the source
    # `__cuda_array_interface__` object.
    cdef object metadata

    # The tensor object if has obj has __dlpack__, otherwise must be NULL
    cdef DLTensor *dl_tensor

    # Memoized properties
    cdef tuple _shape
    cdef tuple _strides
    cdef bint _strides_init  # Has the strides tuple been init'ed?
    cdef object _dtype

    def __init__(self, obj=None, stream_ptr=None):
        if obj is not None:
            # populate self's attributes
            if check_has_dlpack(obj):
                view_as_dlpack(obj, stream_ptr, self)
            else:
                view_as_cai(obj, stream_ptr, self)
        else:
            pass

    @property
    def shape(self) -> tuple[int]:
        if self._shape is None and self.exporting_obj is not None:
            if self.dl_tensor != NULL:
                self._shape = cuda_utils.carray_int64_t_to_tuple(
                    self.dl_tensor.shape,
                    self.dl_tensor.ndim
                )
            else:
                self._shape = self.metadata["shape"]
        else:
            self._shape = ()
        return self._shape

    @property
    def strides(self) -> Optional[tuple[int]]:
        cdef int itemsize
        if self._strides_init is False:
            if self.exporting_obj is not None:
                if self.dl_tensor != NULL:
                    if self.dl_tensor.strides:
                        self._strides = cuda_utils.carray_int64_t_to_tuple(
                            self.dl_tensor.strides,
                            self.dl_tensor.ndim
                        )
                else:
                    strides = self.metadata.get("strides")
                    if strides is not None:
                        itemsize = self.dtype.itemsize
                        self._strides = cpython.PyTuple_New(len(strides))
                        for i in range(len(strides)):
                            cpython.PyTuple_SET_ITEM(
                                self._strides, i, strides[i] // itemsize
                            )
            self._strides_init = True
        return self._strides

    @property
    def dtype(self) -> Optional[numpy.dtype]:
        if self._dtype is None:
            if self.exporting_obj is not None:
                if self.dl_tensor != NULL:
                    self._dtype = dtype_dlpack_to_numpy(&self.dl_tensor.dtype)
                else:
                    # TODO: this only works for built-in numeric types
                    self._dtype = numpy.dtype(self.metadata["typestr"])
        return self._dtype

    def __repr__(self):
        return (f"StridedMemoryView(ptr={self.ptr},\n"
              + f"                  shape={self.shape},\n"
              + f"                  strides={self.strides},\n"
              + f"                  dtype={get_simple_repr(self.dtype)},\n"
              + f"                  device_id={self.device_id},\n"
              + f"                  is_device_accessible={self.is_device_accessible},\n"
              + f"                  readonly={self.readonly},\n"
              + f"                  exporting_obj={get_simple_repr(self.exporting_obj)})")


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
    cdef bint is_device_accessible, is_readonly
    is_device_accessible = False
    dldevice, device_id = obj.__dlpack_device__()
    if dldevice == _kDLCPU:
        assert device_id == 0
        device_id = -1
        if stream_ptr is None:
            raise BufferError("stream=None is ambiguous with view()")
        elif stream_ptr == -1:
            stream_ptr = None
    elif dldevice == _kDLCUDA:
        assert device_id >= 0
        is_device_accessible = True
        # no need to check other stream values, it's a pass-through
        if stream_ptr is None:
            raise BufferError("stream=None is ambiguous with view()")
    elif dldevice in (_kDLCUDAHost, _kDLCUDAManaged):
        is_device_accessible = True
        # just do a pass-through without any checks, as pinned/managed memory can be
        # accessed on both host and device
    else:
        raise BufferError("device not supported")

    cdef object capsule
    try:
        capsule = obj.__dlpack__(
            stream=int(stream_ptr) if stream_ptr else None,
            max_version=(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION))
    except TypeError:
        capsule = obj.__dlpack__(
            stream=int(stream_ptr) if stream_ptr else None)

    cdef void* data = NULL
    cdef DLTensor* dl_tensor
    cdef DLManagedTensorVersioned* dlm_tensor_ver
    cdef DLManagedTensor* dlm_tensor
    cdef const char *used_name
    if cpython.PyCapsule_IsValid(
            capsule, DLPACK_VERSIONED_TENSOR_UNUSED_NAME):
        data = cpython.PyCapsule_GetPointer(
            capsule, DLPACK_VERSIONED_TENSOR_UNUSED_NAME)
        dlm_tensor_ver = <DLManagedTensorVersioned*>data
        dl_tensor = &dlm_tensor_ver.dl_tensor
        is_readonly = bool((dlm_tensor_ver.flags & DLPACK_FLAG_BITMASK_READ_ONLY) != 0)
        used_name = DLPACK_VERSIONED_TENSOR_USED_NAME
    elif cpython.PyCapsule_IsValid(
            capsule, DLPACK_TENSOR_UNUSED_NAME):
        data = cpython.PyCapsule_GetPointer(
            capsule, DLPACK_TENSOR_UNUSED_NAME)
        dlm_tensor = <DLManagedTensor*>data
        dl_tensor = &dlm_tensor.dl_tensor
        is_readonly = False
        used_name = DLPACK_TENSOR_USED_NAME
    else:
        assert False

    cpython.PyCapsule_SetName(capsule, used_name)

    cdef StridedMemoryView buf = StridedMemoryView() if view is None else view
    buf.dl_tensor = dl_tensor
    buf.metadata = capsule
    buf.ptr = <intptr_t>(dl_tensor.data)
    buf.device_id = device_id
    buf.is_device_accessible = is_device_accessible
    buf.readonly = is_readonly
    buf.exporting_obj = obj

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


# Also generate for Python so we can test this code path
cpdef StridedMemoryView view_as_cai(obj, stream_ptr, view=None):
    cdef dict cai_data = obj.__cuda_array_interface__
    if cai_data["version"] < 3:
        raise BufferError("only CUDA Array Interface v3 or above is supported")
    if cai_data.get("mask") is not None:
        raise BufferError("mask is not supported")
    if stream_ptr is None:
        raise BufferError("stream=None is ambiguous with view()")

    cdef StridedMemoryView buf = StridedMemoryView() if view is None else view
    buf.exporting_obj = obj
    buf.metadata = cai_data
    buf.dl_tensor = NULL
    buf.ptr, buf.readonly = cai_data["data"]
    buf.is_device_accessible = True
    buf.device_id = handle_return(
        driver.cuPointerGetAttribute(
            driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
            buf.ptr))

    cdef intptr_t producer_s, consumer_s
    stream_ptr = int(stream_ptr)
    if stream_ptr != -1:
        stream = cai_data.get("stream")
        if stream is not None:
            producer_s = <intptr_t>(stream)
            consumer_s = <intptr_t>(stream_ptr)
            assert producer_s > 0
            # establish stream order
            if producer_s != consumer_s:
                e = handle_return(driver.cuEventCreate(
                    driver.CUevent_flags.CU_EVENT_DISABLE_TIMING))
                handle_return(driver.cuEventRecord(e, producer_s))
                handle_return(driver.cuStreamWaitEvent(consumer_s, e, 0))
                handle_return(driver.cuEventDestroy(e))

    return buf


def args_viewable_as_strided_memory(tuple arg_indices):
    """
    Decorator to create proxy objects to :obj:`StridedMemoryView` for the
    specified positional arguments.

    This allows array/tensor attributes to be accessed inside the function
    implementation, while keeping the function body array-library-agnostic (if
    desired).

    Inside the decorated function, the specified arguments become instances
    of an (undocumented) proxy type, regardless of its original source. A
    :obj:`StridedMemoryView` instance can be obtained by passing the (consumer)
    stream pointer (as a Python `int`) to the proxies's ``view()`` method. For
    example:

    .. code-block:: python

        @args_viewable_as_strided_memory((1,))
        def my_func(arg0, arg1, arg2, stream: Stream):
            # arg1 can be any object supporting DLPack or CUDA Array Interface
            view = arg1.view(stream.handle)
            assert isinstance(view, StridedMemoryView)
            ...

    Parameters
    ----------
    arg_indices : tuple
        The indices of the target positional arguments.
    """
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
