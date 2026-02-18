# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ._dlpack cimport *
from libc.stdint cimport intptr_t
from cuda.core._layout cimport _StridedLayout
from cuda.core._stream import Stream

import functools
import warnings

import numpy

from cuda.bindings cimport cydriver
from cuda.core._resource_handles cimport (
    EventHandle,
    create_event_handle_noctx,
    as_cu,
)

from cuda.core._utils.cuda_utils import handle_return, driver
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN


from cuda.core._memory import Buffer


try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None

# TODO(leofang): support NumPy structured dtypes


cdef class StridedMemoryView:
    """A class holding metadata of a strided dense array/tensor.

    A :obj:`StridedMemoryView` instance can be created in three ways:

      1. Using the :obj:`args_viewable_as_strided_memory` decorator (recommended)
      2. Explicit construction relying on DLPack or CUDA Array Interface, see below.
      3. From :obj:`~_memory.Buffer` and shape and size tuples (see
         :meth:`from_buffer` classmethod)

    ``StridedMemoryView(obj, stream_ptr)`` can be used to create a view from
    objects supporting either DLPack (up to v1.0) or CUDA Array Interface
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

    Parameters
    ----------
    obj : Any
        Any objects that supports either DLPack (up to v1.0) or CUDA Array
        Interface (v3).
    stream_ptr: int
        The pointer address (as Python `int`) to the **consumer** stream.
        Stream ordering will be properly established unless ``-1`` is passed.


    Attributes
    -----------
    ptr : int
        Pointer to the tensor buffer (as a Python `int`).
    device_id : int
        The device ID for where the tensor is located. It is -1 for CPU tensors
        (meaning those only accessible from the host).
    is_device_accessible : bool
        Whether the tensor data can be accessed on the GPU.
    readonly: bool
        Whether the tensor data can be modified in place.
    exporting_obj : Any
        A reference to the original tensor object that is being viewed.
        If the view is created with :meth:`from_buffer`,
        it will be the Buffer instance passed to the method.

    Notes
    -----
    The :attr:`dtype` property supports narrow data types (e.g., ``bfloat16``) when the optional
    `ml_dtypes <https://github.com/jax-ml/ml_dtypes>`_ package is installed. This enables
    interoperability with libraries that use narrow dtype tensors, such as PyTorch with
    ``torch.bfloat16`` or CuPy with ``"bfloat16"`` dtype. If ``ml_dtypes`` is not available
    and such a tensor is encountered, a :obj:`NotImplementedError` will be raised.

    Currently supported narrow data types:
    - ``bfloat16``

    """
    cdef readonly:
        intptr_t ptr
        int device_id
        bint is_device_accessible
        bint readonly
        object exporting_obj

    cdef:
        # If using dlpack, this is a strong reference to the result of
        # obj.__dlpack__() so we can lazily create shape and strides from
        # it later.  If using CAI, this is a reference to the source
        # `__cuda_array_interface__` object.
        object metadata

        # The tensor object if has obj has __dlpack__, otherwise must be NULL
        DLTensor *dl_tensor

        # Memoized properties
        # Either lazily inferred from dl_tensor/metadata,
        # or explicitly provided if created with from_buffer().
        _StridedLayout _layout
        # Either exporting_obj if it is a Buffer, otherwise a Buffer instance
        # with owner set to the exporting object.
        object _buffer
        # Either lazily inferred from dl_tensor/metadata,
        # or explicitly provided if created with from_buffer().
        # In the latter case, it can be None.
        object _dtype

    def __init__(self, obj: object = None, stream_ptr: int | None = None) -> None:
        cdef str clsname = self.__class__.__name__
        if obj is not None:
            # populate self's attributes
            if check_has_dlpack(obj):
                warnings.warn(
                    f"Constructing a {clsname} directly from a DLPack-supporting object is deprecated; "
                    "Use `StridedMemoryView.from_dlpack` or `StridedMemoryView.from_any_interface` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                view_as_dlpack(obj, stream_ptr, self)
            else:
                warnings.warn(
                    f"Constructing a {clsname} directly from a CUDA-array-interface-supporting object is deprecated; "
                    "Use `StridedMemoryView.from_cuda_array_interface` or `StridedMemoryView.from_any_interface` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                view_as_cai(obj, stream_ptr, self)
        else:
            warnings.warn(
                f"Constructing an empty {clsname} is deprecated; "
                "use one of the classmethods `from_dlpack`, `from_cuda_array_interface` or `from_any_interface` "
                "to construct a StridedMemoryView from an object",
                DeprecationWarning,
                stacklevel=2,
            )

    @classmethod
    def from_dlpack(cls, obj: object, stream_ptr: int | None=None) -> StridedMemoryView:
        cdef StridedMemoryView buf = StridedMemoryView.__new__(cls)
        view_as_dlpack(obj, stream_ptr, buf)
        return buf

    @classmethod
    def from_cuda_array_interface(cls, obj: object, stream_ptr: int | None=None) -> StridedMemoryView:
        cdef StridedMemoryView buf = StridedMemoryView.__new__(cls)
        view_as_cai(obj, stream_ptr, buf)
        return buf

    @classmethod
    def from_array_interface(cls, obj: object) -> StridedMemoryView:
        cdef StridedMemoryView buf = StridedMemoryView.__new__(cls)
        view_as_array_interface(obj, buf)
        return buf

    @classmethod
    def from_any_interface(cls, obj: object, stream_ptr: int | None = None) -> StridedMemoryView:
        if check_has_dlpack(obj):
            return cls.from_dlpack(obj, stream_ptr)
        return cls.from_cuda_array_interface(obj, stream_ptr)

    @classmethod
    def from_buffer(
        cls,
        buffer : Buffer,
        shape : tuple[int, ...],
        strides : tuple[int, ...] | None = None,
        *,
        itemsize : int | None = None,
        dtype : numpy.dtype | None = None,
        is_readonly : bool = False
    ) -> StridedMemoryView:
        """
        Creates a :obj:`StridedMemoryView` instance from a :obj:`~_memory.Buffer` and shape and strides tuples.
        The Buffer can be either allocation coming from a :obj:`MemoryResource` or an external allocation
        wrapped in a :obj:`~_memory.Buffer` object with ``Buffer.from_handle(ptr, size, owner=...)``.

        .. caution::
            When creating a :obj:`StridedMemoryView` from a :obj:`~_memory.Buffer`,
            no synchronization is performed. It is the user's responsibility to ensure
            the data in ``buffer`` is properly synchronized when consuming the view.

        Parameters
        ----------
        buffer : :obj:`~_memory.Buffer`
            The buffer to create the view from.
        shape : :obj:`tuple`
            The layout describing the shape, strides and itemsize of the elements in
            the buffer.
        strides : :obj:`tuple`
            The layout describing the shape, strides and itemsize of the elements in
            the buffer.
        dtype : :obj:`numpy.dtype`
            Optional dtype.
            If specified, the dtype's itemsize must match the layout's itemsize.
        is_readonly : bool, optional
            Whether the mark the view as readonly.
        """
        cdef StridedMemoryView view = StridedMemoryView.__new__(cls)
        if itemsize is None and dtype is None:
            raise ValueError("Either itemsize or dtype must be specified")
        if itemsize is not None and dtype is not None and itemsize != dtype.itemsize:
            raise ValueError(
                f"itemsize ({itemsize}) does not match dtype.itemsize ({dtype.itemsize})"
            )
        # (itemsize is None XOR dtype is None) OR they are equal
        view_buffer_strided(
            view,
            buffer,
            _StridedLayout(shape=shape, strides=strides, itemsize=getattr(dtype, "itemsize", itemsize)),
            dtype,
            is_readonly,
        )
        return view

    def __dealloc__(self):
        if self.dl_tensor == NULL:
            return

        if cpython.PyCapsule_IsValid(
                self.metadata, DLPACK_VERSIONED_TENSOR_USED_NAME):
            data = cpython.PyCapsule_GetPointer(
                self.metadata, DLPACK_VERSIONED_TENSOR_USED_NAME)
            dlm_tensor_ver = <DLManagedTensorVersioned*>data
            dlm_tensor_ver.deleter(dlm_tensor_ver)
        elif cpython.PyCapsule_IsValid(
                self.metadata, DLPACK_TENSOR_USED_NAME):
            data = cpython.PyCapsule_GetPointer(
                self.metadata, DLPACK_TENSOR_USED_NAME)
            dlm_tensor = <DLManagedTensor*>data
            dlm_tensor.deleter(dlm_tensor)

    def view(
        self, layout : _StridedLayout | None = None, dtype : numpy.dtype | None = None
    ) -> StridedMemoryView:
        """
        Creates a new view with adjusted layout and dtype.
        Same as calling :meth:`from_buffer` with the current buffer.
        """
        cdef StridedMemoryView view = StridedMemoryView.__new__(self.__class__)
        if layout is None and dtype is None:
            return self
        if layout is None:
            layout = self.get_layout()
        if dtype is None:
            dtype = self.get_dtype()
        view_buffer_strided(view, self.get_buffer(), layout, dtype, self.readonly)
        return view

    def copy_from(
        self, other : StridedMemoryView, stream : Stream,
        allocator = None,
        blocking : bool | None = None,
    ):
        """
        Copies the data from the other view into this view.

        The copy can be performed between following memory spaces:
        host-to-device, device-to-host, device-to-device (on the same device).

        Parameters
        ----------
        other : StridedMemoryView
            The view to copy data from.
        stream : Stream | None, optional
            The stream to schedule the copy on.
        allocator : MemoryResource | None, optional
            If temporary buffers are needed, the specified memory resources
            will be used to allocate the memory. If not specified, default
            resources will be used.
        blocking : bool | None, optional
            Whether the call should block until the copy is complete.
                * ``True``: the ``stream`` is synchronized with the host at the end of the call,
                  blocking until the copy is complete.
                * ``False``: if possible, the call returns immediately once the copy is scheduled.
                  However, in some cases of host-to-device or device-to-host copies, the call may
                  still synchronize with the host if necessary.
                * ``None`` (default):
                    * for device-to-device, it defaults to ``False`` (non-blocking),
                    * for host-to-device or device-to-host, it defaults to ``True`` (blocking).
        """
        raise NotImplementedError("Sorry, not supported: copy_from")

    def copy_to(
        self, other : StridedMemoryView, stream : Stream | None = None,
        allocator = None,
        blocking : bool | None = None,
    ):
        """
        Copies the data from this view into the ``other`` view.

        For details, see :meth:`copy_from`.
        """
        raise NotImplementedError("Sorry, not supported: copy_to")

    @property
    def _layout(self) -> _StridedLayout:
        """
        The layout of the tensor. For StridedMemoryView created from DLPack or CAI,
        the layout is inferred from the tensor object's metadata.
        """
        return self.get_layout()

    @property
    def size(self) -> int:
        return self.get_layout().get_volume()

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Shape of the tensor.
        """
        return self.get_layout().get_shape_tuple()

    @property
    def strides(self) -> tuple[int, ...] | None:
        """
        Strides of the tensor (in **counts**, not bytes).
        """
        return self.get_layout().get_strides_tuple()

    @property
    def dtype(self) -> numpy.dtype | None:
        """
        Data type of the tensor.

        Supports standard NumPy dtypes as well as narrow data types (e.g., ``bfloat16``)
        when the optional `ml_dtypes <https://github.com/jax-ml/ml_dtypes>`_ package is
        installed. If ``ml_dtypes`` is not available and such a tensor is encountered,
        a :obj:`NotImplementedError` will be raised.
        """
        return self.get_dtype()

    def __repr__(self):
        return (f"StridedMemoryView(ptr={self.ptr},\n"
              + f"                  shape={self.shape},\n"
              + f"                  strides={self.strides},\n"
              + f"                  itemsize={self._layout.itemsize},\n"
              + f"                  dtype={get_simple_repr(self.dtype)},\n"
              + f"                  device_id={self.device_id},\n"
              + f"                  is_device_accessible={self.is_device_accessible},\n"
              + f"                  readonly={self.readonly},\n"
              + f"                  exporting_obj={get_simple_repr(self.exporting_obj)})")

    cdef inline _StridedLayout get_layout(self):
        if self._layout is None:
            if self.dl_tensor:
                self._layout = layout_from_dlpack(self.dl_tensor)
            elif self.metadata is not None:
                self._layout = layout_from_cai(self.metadata)
            else:
                raise ValueError("Cannot infer layout from the exporting object")
        return self._layout

    cdef inline object get_buffer(self):
        """
        Returns Buffer instance with the underlying data.
        If the SMV was created from a Buffer, it will return the same Buffer instance.
        Otherwise, it will create a new instance with owner set to the exporting object.
        """
        if self._buffer is None:
            if isinstance(self.exporting_obj, Buffer):
                self._buffer = self.exporting_obj
            else:
                self._buffer = Buffer.from_handle(self.ptr, 0, owner=self.exporting_obj)
        return self._buffer

    cdef inline object get_dtype(self):
        if self._dtype is None:
            if self.dl_tensor != NULL:
                self._dtype = dtype_dlpack_to_numpy(&self.dl_tensor.dtype)
            elif self.metadata is not None:
                self._dtype = _typestr2dtype(self.metadata["typestr"])
        return self._dtype


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
    cdef readonly:
        object obj
        bint has_dlpack

    def __init__(self, obj):
        self.obj = obj
        self.has_dlpack = check_has_dlpack(obj)

    cpdef StridedMemoryView view(self, stream_ptr=None):
        if self.has_dlpack:
            return StridedMemoryView.from_dlpack(self.obj, stream_ptr)
        else:
            return StridedMemoryView.from_cuda_array_interface(self.obj, stream_ptr)


cdef StridedMemoryView view_as_dlpack(obj, stream_ptr, view=None):
    cdef int dldevice, device_id
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


@functools.lru_cache
def _typestr2dtype(str typestr):
    return numpy.dtype(typestr)


@functools.lru_cache
def _typestr2itemsize(str typestr):
    return _typestr2dtype(typestr).itemsize


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
        if bfloat16 is not None:
            np_dtype = numpy.dtype("bfloat16")
        else:
            raise NotImplementedError(
                'Support for bfloat16 within cuda-core requires `ml_dtypes`'
                'to be installed.'
            )
    else:
        raise TypeError('Unsupported dtype. dtype code: {}'.format(dtype.code))

    # We want the dtype object not just the type object
    return numpy.dtype(np_dtype)


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
    if buf.ptr != 0:
        buf.device_id = handle_return(
            driver.cuPointerGetAttribute(
                driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                buf.ptr))
    else:
        buf.device_id = handle_return(driver.cuCtxGetDevice())

    cdef intptr_t producer_s, consumer_s
    cdef EventHandle h_event
    stream_ptr = int(stream_ptr)
    if stream_ptr != -1:
        stream = cai_data.get("stream")
        if stream is not None:
            producer_s = <intptr_t>(stream)
            consumer_s = <intptr_t>(stream_ptr)
            assert producer_s > 0
            # establish stream order
            if producer_s != consumer_s:
                with nogil:
                    h_event = create_event_handle_noctx(cydriver.CUevent_flags.CU_EVENT_DISABLE_TIMING)
                    HANDLE_RETURN(cydriver.cuEventRecord(
                        as_cu(h_event), <cydriver.CUstream>producer_s))
                    HANDLE_RETURN(cydriver.cuStreamWaitEvent(
                        <cydriver.CUstream>consumer_s, as_cu(h_event), 0))

    return buf


cpdef StridedMemoryView view_as_array_interface(obj, view=None):
    cdef dict data = obj.__array_interface__
    if data["version"] < 3:
        raise BufferError("only NumPy Array Interface v3 or above is supported")
    if data.get("mask") is not None:
        raise BufferError("mask is not supported")

    cdef StridedMemoryView buf = StridedMemoryView() if view is None else view
    buf.exporting_obj = obj
    buf.metadata = data
    buf.dl_tensor = NULL
    buf.ptr, buf.readonly = data["data"]
    buf.is_device_accessible = False
    buf.device_id = handle_return(driver.cuCtxGetDevice())
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


cdef inline _StridedLayout layout_from_dlpack(DLTensor* dl_tensor):
    cdef _StridedLayout layout = _StridedLayout.__new__(_StridedLayout)
    cdef int nbits = dl_tensor.dtype.bits * dl_tensor.dtype.lanes
    cdef int itemsize = nbits >> 3
    if (itemsize << 3) != nbits:
        raise ValueError("dl_tensor.dtype.bits must be a multiple of 8")
    layout.init_from_ptr(dl_tensor.ndim, dl_tensor.shape, dl_tensor.strides, itemsize)
    return layout


cdef _StridedLayout layout_from_cai(object metadata):
    cdef _StridedLayout layout = _StridedLayout.__new__(_StridedLayout)
    cdef object shape = metadata["shape"]
    cdef object strides = metadata.get("strides")
    cdef int itemsize = _typestr2itemsize(metadata["typestr"])
    layout.init_from_tuple(shape, strides, itemsize, True)
    return layout


cdef inline intptr_t get_data_ptr(object buffer, _StridedLayout layout) except? 0:
    return <intptr_t>(int(buffer.handle)) + layout.get_slice_offset_in_bytes()


cdef inline int view_buffer_strided(
    StridedMemoryView view,
    object buffer,
    _StridedLayout layout,
    object dtype,
    bint is_readonly,
) except -1:
    if dtype is not None:
        dtype = numpy.dtype(dtype)
        if dtype.itemsize != layout.itemsize:
            raise ValueError(
                f"The dtype's itemsize ({dtype.itemsize}) does not match the layout's "
                f"itemsize ({layout.itemsize})."
            )
    # Check the layout's offset range [min_offset, max_offset] fits
    # within the [0, buffer.size - 1] range.
    # The required_size_in_bytes fails if min_offset < 0.
    # NB. For external memory, both positive and negative offsets can be valid,
    # but for a proper check we'd need to know both size and data offset,
    # while neither is reported by the packages.
    cdef bint is_allocated = buffer.memory_resource is not None
    if is_allocated and buffer.size < layout.get_required_size_in_bytes():
        raise ValueError(
            f"Buffer size is too small for the layout. "
            f"Expected at least {layout.get_required_size_in_bytes()} bytes, "
            f"got {buffer.size} bytes."
        )
    # set the public attributes
    view.ptr = get_data_ptr(buffer, layout)
    view.device_id = buffer.device_id
    view.is_device_accessible = buffer.is_device_accessible
    view.readonly = is_readonly
    view.exporting_obj = view._buffer = buffer
    # no dlpack/cai metadata
    view.dl_tensor = NULL
    view.metadata = None
    # we get the layout from the caller
    view._layout = layout
    view._dtype = dtype
    return 0
