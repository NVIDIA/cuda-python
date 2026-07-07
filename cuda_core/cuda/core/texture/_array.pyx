# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

cimport cpython
from libc.stdint cimport intptr_t
from libc.string cimport memset

from cuda.bindings cimport cydriver
from cuda.core._memory._buffer cimport Buffer
from cuda.core._resource_handles cimport (
    OpaqueArrayHandle,
    as_cu,
    as_intptr,
    create_array_handle,
    create_array_handle_owning,
    create_array_handle_ref,
    get_last_error,
)
from cuda.core._stream cimport Stream, Stream_accept
from cuda.core._utils.cuda_utils cimport (
    HANDLE_RETURN,
    _get_current_device_id,
)

import numpy

from dataclasses import dataclass

from cuda.core._utils.cuda_utils import check_or_create_options
from cuda.core.typing import ArrayFormatType


# Bridge between the public ArrayFormatType StrEnum and the driver
# CUarray_format integer values. OpaqueArray stores the driver int internally
# (see ._format), so all conversions funnel through these two maps.
_ARRAYFORMAT_TO_CU = {
    ArrayFormatType.UINT8:   int(cydriver.CU_AD_FORMAT_UNSIGNED_INT8),
    ArrayFormatType.UINT16:  int(cydriver.CU_AD_FORMAT_UNSIGNED_INT16),
    ArrayFormatType.UINT32:  int(cydriver.CU_AD_FORMAT_UNSIGNED_INT32),
    ArrayFormatType.INT8:    int(cydriver.CU_AD_FORMAT_SIGNED_INT8),
    ArrayFormatType.INT16:   int(cydriver.CU_AD_FORMAT_SIGNED_INT16),
    ArrayFormatType.INT32:   int(cydriver.CU_AD_FORMAT_SIGNED_INT32),
    ArrayFormatType.FLOAT16: int(cydriver.CU_AD_FORMAT_HALF),
    ArrayFormatType.FLOAT32: int(cydriver.CU_AD_FORMAT_FLOAT),
}
_CU_TO_ARRAYFORMAT = {cu: fmt for fmt, cu in _ARRAYFORMAT_TO_CU.items()}


# Every ArrayFormatType value is spelled as a NumPy dtype name, so the eight
# formats map 1:1 to NumPy dtypes. This lets callers pass a dtype object (or
# anything numpy.dtype() accepts) instead of the enum, matching the precedent
# set by TensorMapDescriptorOptions.data_type.
_NUMPY_DTYPE_TO_ARRAYFORMAT = {
    numpy.dtype(fmt.value): fmt for fmt in ArrayFormatType
}


# Bytes per element (single channel), keyed by the driver CUarray_format int.
_FORMAT_ELEM_SIZE = {
    _ARRAYFORMAT_TO_CU[ArrayFormatType.UINT8]:   1,
    _ARRAYFORMAT_TO_CU[ArrayFormatType.INT8]:    1,
    _ARRAYFORMAT_TO_CU[ArrayFormatType.UINT16]:  2,
    _ARRAYFORMAT_TO_CU[ArrayFormatType.INT16]:   2,
    _ARRAYFORMAT_TO_CU[ArrayFormatType.FLOAT16]: 2,
    _ARRAYFORMAT_TO_CU[ArrayFormatType.UINT32]:  4,
    _ARRAYFORMAT_TO_CU[ArrayFormatType.INT32]:   4,
    _ARRAYFORMAT_TO_CU[ArrayFormatType.FLOAT32]: 4,
}


def _normalize_array_format(format):
    """Coerce ``format`` to an :class:`ArrayFormatType`.

    Accepts, in order of preference:

    * an :class:`ArrayFormatType`;
    * a plain ``str`` naming one of its values (e.g. ``"float32"``);
    * a NumPy dtype object (or anything ``numpy.dtype()`` accepts, such as
      ``numpy.float32``) whose canonical dtype maps 1:1 to one of the eight
      supported formats.

    Raises :class:`ValueError` on anything else."""
    if isinstance(format, ArrayFormatType):
        return format
    if isinstance(format, str):
        try:
            return ArrayFormatType(format)
        except ValueError as e:
            valid = ", ".join(repr(f.value) for f in ArrayFormatType)
            raise ValueError(
                f"format must be an ArrayFormatType or one of {{{valid}}}, got {format!r}"
            ) from e
    # Fall back to interpreting ``format`` as a NumPy dtype (dtype object,
    # scalar type, etc.). Unknown dtypes are reported against the supported set.
    try:
        dt = numpy.dtype(format)
    except TypeError as e:
        raise ValueError(
            f"format must be an ArrayFormatType, str, or NumPy dtype, got {format!r}"
        ) from e
    try:
        return _NUMPY_DTYPE_TO_ARRAYFORMAT[dt]
    except KeyError as e:
        valid = ", ".join(repr(f.value) for f in ArrayFormatType)
        raise ValueError(
            f"NumPy dtype {dt!r} has no ArrayFormatType equivalent; "
            f"supported formats: {{{valid}}}"
        ) from e


def _validate_format_channels(format, num_channels):
    """Validate the ``(format, num_channels)`` pair shared by the array,
    mipmap, and texture factories. Returns the normalized
    :class:`ArrayFormatType`. Raises on an invalid combination."""
    fmt = _normalize_array_format(format)
    if isinstance(num_channels, bool) or num_channels not in (1, 2, 4):
        raise ValueError(f"num_channels must be 1, 2, or 4, got {num_channels!r}")
    return fmt


def _validate_array_shape(shape):
    """Coerce ``shape`` to a tuple of ints and validate rank (1-3) and that
    every extent is >= 1. Returns the normalized tuple."""
    try:
        shape_t = tuple(int(s) for s in shape)
    except TypeError as e:
        raise TypeError(f"shape must be a tuple of ints, got {type(shape).__name__}") from e
    if not 1 <= len(shape_t) <= 3:
        raise ValueError(f"shape rank must be 1, 2, or 3, got {len(shape_t)}")
    for i, dim in enumerate(shape_t):
        if dim < 1:
            raise ValueError(f"shape[{i}] must be >= 1, got {dim}")
    return shape_t


@dataclass
class OpaqueArrayOptions:
    """Options for :meth:`cuda.core.Device.create_opaque_array`.

    Attributes
    ----------
    shape : tuple of int
        ``(width,)``, ``(width, height)``, or ``(width, height, depth)`` in
        elements.
    format : ArrayFormatType, str, or numpy.dtype
        Element format. Accepts an :class:`~cuda.core.typing.ArrayFormatType`,
        a plain string (e.g. ``"float32"``), or a NumPy dtype object.
    num_channels : int
        Channels per element. Must be 1, 2, or 4.
    is_surface_load_store : bool
        If True, allocate with ``CUDA_ARRAY3D_SURFACE_LDST`` so the array can be
        bound as a :class:`~cuda.core.texture.SurfaceObject` for kernel-side
        writes. Default False.

    .. versionadded:: 1.1.0
    """

    shape: tuple[int, ...]
    format: object
    num_channels: int
    is_surface_load_store: bool = False

    def __post_init__(self):
        self.format = _validate_format_channels(self.format, self.num_channels)
        self.shape = _validate_array_shape(self.shape)


cdef void _fill_array_endpoint(
    cydriver.CUDA_MEMCPY3D* p, OpaqueArray arr, bint is_src
) noexcept:
    """Populate the src or dst array fields of a CUDA_MEMCPY3D struct."""
    if is_src:
        p.srcMemoryType = cydriver.CU_MEMORYTYPE_ARRAY
        p.srcArray = as_cu(arr._handle)
        p.srcXInBytes = 0
        p.srcY = 0
        p.srcZ = 0
    else:
        p.dstMemoryType = cydriver.CU_MEMORYTYPE_ARRAY
        p.dstArray = as_cu(arr._handle)
        p.dstXInBytes = 0
        p.dstY = 0
        p.dstZ = 0


cdef int _fill_host_endpoint(
    cydriver.CUDA_MEMCPY3D* p,
    object obj,
    bint is_src,
    size_t width_bytes,
    size_t height,
    size_t required,
    cpython.Py_buffer* pybuf_out,
) except -1:
    """Populate src/dst host fields from a buffer-protocol ``obj``.

    Acquires a Py_buffer view; the caller is responsible for releasing it
    (this function always returns with the view held when it returns 1).
    """
    cdef int flags = cpython.PyBUF_SIMPLE
    if not is_src:
        flags |= cpython.PyBUF_WRITABLE
    if cpython.PyObject_GetBuffer(obj, pybuf_out, flags) != 0:
        raise TypeError(
            f"Source/destination must be a Buffer or a contiguous "
            f"buffer-protocol object, got {type(obj).__name__}"
        )
    if <size_t>pybuf_out.len < required:
        cpython.PyBuffer_Release(pybuf_out)
        raise ValueError(
            f"Host buffer has {pybuf_out.len} bytes, smaller than the array "
            f"extent ({required} bytes)"
        )
    if is_src:
        p.srcMemoryType = cydriver.CU_MEMORYTYPE_HOST
        p.srcHost = pybuf_out.buf
        p.srcPitch = width_bytes
        p.srcHeight = height
        p.srcXInBytes = 0
        p.srcY = 0
        p.srcZ = 0
    else:
        p.dstMemoryType = cydriver.CU_MEMORYTYPE_HOST
        p.dstHost = pybuf_out.buf
        p.dstPitch = width_bytes
        p.dstHeight = height
        p.dstXInBytes = 0
        p.dstY = 0
        p.dstZ = 0
    return 1


cdef int _fill_linear_endpoint(
    cydriver.CUDA_MEMCPY3D* p,
    object obj,
    bint is_src,
    size_t width_bytes,
    size_t height,
    size_t depth,
    cpython.Py_buffer* pybuf_out,
) except -1:
    """Populate the src or dst linear fields. Returns 1 if pybuf_out was
    filled (caller must release it), 0 otherwise.
    """
    cdef intptr_t ptr
    cdef size_t required = width_bytes * height * depth
    if isinstance(obj, Buffer):
        if <size_t>(<Buffer>obj).size < required:
            raise ValueError(
                f"Buffer size ({(<Buffer>obj).size} bytes) is smaller than "
                f"the array extent ({required} bytes)"
            )
        ptr = int((<Buffer>obj).handle)
        if is_src:
            p.srcMemoryType = cydriver.CU_MEMORYTYPE_DEVICE
            p.srcDevice = <cydriver.CUdeviceptr>ptr
            p.srcPitch = width_bytes
            p.srcHeight = height
            p.srcXInBytes = 0
            p.srcY = 0
            p.srcZ = 0
        else:
            p.dstMemoryType = cydriver.CU_MEMORYTYPE_DEVICE
            p.dstDevice = <cydriver.CUdeviceptr>ptr
            p.dstPitch = width_bytes
            p.dstHeight = height
            p.dstXInBytes = 0
            p.dstY = 0
            p.dstZ = 0
        return 0
    return _fill_host_endpoint(
        p, obj, is_src, width_bytes, height, required, pybuf_out
    )


cdef _copy3d(OpaqueArray arr, object other, Stream stream, bint to_array):
    """Issue a full-array async 3D memcpy between ``arr`` and ``other``.

    Direction is determined by ``to_array``: True copies *into* arr, False
    copies *out of* arr. ``stream`` must already be a concrete :class:`Stream`
    (callers coerce via :func:`Stream_accept`).
    """
    cdef cydriver.CUDA_MEMCPY3D params
    cdef cpython.Py_buffer pybuf
    cdef int got_buffer = 0
    cdef intptr_t stream_handle
    cdef cydriver.CUstream c_stream

    memset(&params, 0, sizeof(params))
    width_bytes, height, depth = arr._extent_bytes()
    params.WidthInBytes = <size_t>width_bytes
    params.Height = <size_t>height
    params.Depth = <size_t>depth

    try:
        if to_array:
            got_buffer = _fill_linear_endpoint(
                &params, other, True, width_bytes, height, depth, &pybuf
            )
            _fill_array_endpoint(&params, arr, False)
        else:
            _fill_array_endpoint(&params, arr, True)
            got_buffer = _fill_linear_endpoint(
                &params, other, False, width_bytes, height, depth, &pybuf
            )

        stream_handle = int((<Stream>stream).handle)
        c_stream = <cydriver.CUstream><void*>stream_handle
        with nogil:
            HANDLE_RETURN(cydriver.cuMemcpy3DAsync(&params, c_stream))
    finally:
        if got_buffer:
            cpython.PyBuffer_Release(&pybuf)


cdef class OpaqueArray:
    """An opaque, hardware-laid-out GPU allocation for texture/surface access.

    Distinct from :class:`Buffer`: a ``CUarray`` has no exposed device pointer
    and can only be accessed from kernels through a :class:`TextureObject` or
    :class:`SurfaceObject`. Its memory layout is chosen by the driver for 2D/3D
    spatial locality.

    **Copy-only interop.** Because the layout is opaque and there is no linear
    device pointer, a ``OpaqueArray`` cannot expose ``__cuda_array_interface__`` /
    DLPack and cannot be shared zero-copy with NumPy, CuPy, numba-cuda, or
    PyTorch. Moving data in or out is therefore always a copy: use
    :meth:`copy_from` / :meth:`copy_to` against a linear :class:`Buffer` or a
    host buffer-protocol object. There is no allocation helper — allocate the
    linear :class:`Buffer` yourself (e.g. ``mr.allocate(arr.size_bytes,
    stream=s)``) and copy.

    Construct via :meth:`cuda.core.Device.create_opaque_array`. Only plain
    1D/2D/3D allocations are supported in this initial version; layered/cubemap/
    sparse variants will follow once their shape semantics are settled.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "OpaqueArray cannot be instantiated directly. "
            "Use Device.create_opaque_array()."
        )

    @classmethod
    def _from_handle(cls, intptr_t handle, bint owning, *, device_id=None):
        """Wrap an externally-allocated ``CUarray``.

        Intended for graphics interop (``cuGraphicsSubResourceGetMappedArray``)
        where the array is owned by the graphics API. With ``owning=False`` the
        underlying ``CUarray`` is never destroyed by this object. Shape, format,
        and channel count are queried from the driver.
        """
        cdef cydriver.CUarray raw = <cydriver.CUarray><void*>handle
        cdef OpaqueArrayHandle h
        if owning:
            h = create_array_handle_owning(raw)
        else:
            h = create_array_handle_ref(raw)
        cdef int dev = _get_current_device_id() if device_id is None else int(device_id)
        return _array_from_handle(h, dev)

    @property
    def handle(self):
        """The underlying ``CUarray`` as an integer."""
        return as_intptr(self._handle)

    @property
    def shape(self):
        """Allocation shape, in elements."""
        return self._shape

    @property
    def format(self):
        """The element :class:`~cuda.core.typing.ArrayFormatType`."""
        return _CU_TO_ARRAYFORMAT[self._format]

    @property
    def num_channels(self):
        """Channels per element (1, 2, or 4)."""
        return self._num_channels

    @property
    def element_bytes(self):
        """Bytes per element (format size * channels).

        .. versionadded:: 1.1.0
        """
        return _FORMAT_ELEM_SIZE[self._format] * self._num_channels

    @property
    def device(self):
        """The :class:`Device` this array was allocated on."""
        from cuda.core._device import Device
        return Device(self._device_id)

    @property
    def is_surface_load_store(self):
        """True if this array was created with ``CUDA_ARRAY3D_SURFACE_LDST``
        and can be bound as a :class:`SurfaceObject`."""
        return self._surface_load_store

    def _extent_bytes(self):
        """Return (width_bytes, height, depth) for cuMemcpy3D, with height/depth
        normalized to >=1 for lower-rank arrays."""
        cdef int rank = len(self._shape)
        cdef size_t w = <size_t>self._shape[0] * <size_t>(
            _FORMAT_ELEM_SIZE[self._format] * self._num_channels
        )
        cdef size_t h = <size_t>(self._shape[1] if rank >= 2 else 1)
        cdef size_t d = <size_t>(self._shape[2] if rank >= 3 else 1)
        return w, h, d

    def copy_from(self, src, *, stream) -> None:
        """Copy a full-array's worth of data into this array.

        Parameters
        ----------
        src : Buffer or buffer-protocol object
            Source data. Must contain at least ``self.size_bytes`` bytes
            of contiguous data.
        stream : Stream or GraphBuilder
            Stream to issue the copy on. A :class:`~cuda.core.graph.GraphBuilder`
            is accepted so the copy can be captured into a graph.
        """
        _copy3d(self, src, Stream_accept(stream), to_array=True)

    def copy_to(self, dst, *, stream):
        """Copy a full-array's worth of data out of this array.

        Parameters
        ----------
        dst : Buffer or writable buffer-protocol object
            Destination. Must have at least ``self.size_bytes`` bytes of
            writable, contiguous space.
        stream : Stream or GraphBuilder
            Stream to issue the copy on. A :class:`~cuda.core.graph.GraphBuilder`
            is accepted so the copy can be captured into a graph.

        Returns
        -------
        The ``dst`` object, for parity with :meth:`Buffer.copy_to`.
        """
        _copy3d(self, dst, Stream_accept(stream), to_array=False)
        return dst

    @property
    def size_bytes(self):
        """Total bytes of array storage (``prod(shape) * element_bytes``)."""
        cdef size_t n = 1
        for s in self._shape:
            n *= <size_t>s
        return n * <size_t>(_FORMAT_ELEM_SIZE[self._format] * self._num_channels)

    cpdef close(self):
        """Release this object's reference to the underlying ``CUarray``.

        Destruction (``cuArrayDestroy``) happens via the handle's deleter when
        the last reference is dropped; for a non-owning handle (graphics interop
        or a mipmap-level view) nothing is destroyed. Idempotent: a second call
        (or destruction after ``close()``) is a no-op.
        """
        self._handle.reset()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __repr__(self):
        return (
            f"OpaqueArray(shape={self._shape}, "
            f"format={_CU_TO_ARRAYFORMAT[self._format].name}, "
            f"num_channels={self._num_channels})"
        )


cdef OpaqueArray _array_from_handle(OpaqueArrayHandle h, int device_id):
    """Wrap an existing OpaqueArrayHandle as a OpaqueArray, querying the driver for the
    array's shape/format/channels/surface-flag metadata.

    Any owning/non-owning semantics and parent (mipmap) dependency are already
    captured structurally inside ``h``'s C++ box.
    """
    if not h:
        HANDLE_RETURN(get_last_error())

    cdef OpaqueArray self = OpaqueArray.__new__(OpaqueArray)
    self._handle = h
    self._device_id = device_id

    cdef cydriver.CUDA_ARRAY3D_DESCRIPTOR desc
    cdef cydriver.CUarray raw = as_cu(h)
    with nogil:
        HANDLE_RETURN(cydriver.cuArray3DGetDescriptor(&desc, raw))

    if desc.Depth > 0:
        self._shape = (int(desc.Width), int(desc.Height), int(desc.Depth))
    elif desc.Height > 0:
        self._shape = (int(desc.Width), int(desc.Height))
    else:
        self._shape = (int(desc.Width),)
    self._format = desc.Format
    self._num_channels = desc.NumChannels
    self._surface_load_store = bool(desc.Flags & cydriver.CUDA_ARRAY3D_SURFACE_LDST)
    return self


def _create_opaque_array(options):
    """Allocate a new :class:`OpaqueArray` on the current device.

    Backs :meth:`cuda.core.Device.create_opaque_array`. ``options`` is an
    :class:`OpaqueArrayOptions` (or a mapping accepted by it); it is validated
    at construction, so ``shape`` is already a normalized tuple and ``format``
    an :class:`~cuda.core.typing.ArrayFormatType`.
    """
    cdef object opts = check_or_create_options(
        OpaqueArrayOptions, options, "Opaque array options"
    )
    shape_t = opts.shape

    cdef cydriver.CUarray_format c_format = <cydriver.CUarray_format>_ARRAYFORMAT_TO_CU[opts.format]
    cdef cydriver.CUDA_ARRAY3D_DESCRIPTOR desc3d
    cdef int rank = len(shape_t)
    cdef unsigned int flags = (
        cydriver.CUDA_ARRAY3D_SURFACE_LDST if opts.is_surface_load_store else 0
    )

    # cuArray3DCreate handles 1D/2D/3D uniformly (Height/Depth 0 sentinels),
    # so a single descriptor + create_array_handle covers every shape.
    memset(&desc3d, 0, sizeof(desc3d))
    desc3d.Width = <size_t>shape_t[0]
    desc3d.Height = <size_t>(shape_t[1] if rank >= 2 else 0)
    desc3d.Depth = <size_t>(shape_t[2] if rank >= 3 else 0)
    desc3d.Format = c_format
    desc3d.NumChannels = <unsigned int>opts.num_channels
    desc3d.Flags = flags

    cdef OpaqueArrayHandle h = create_array_handle(desc3d)
    if not h:
        HANDLE_RETURN(get_last_error())

    cdef OpaqueArray self = OpaqueArray.__new__(OpaqueArray)
    self._handle = h
    self._shape = shape_t
    self._format = c_format
    self._num_channels = opts.num_channels
    self._surface_load_store = bool(opts.is_surface_load_store)
    self._device_id = _get_current_device_id()
    return self
