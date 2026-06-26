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
    CUDAArrayHandle,
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

from enum import IntEnum


class ArrayFormat(IntEnum):
    """Element format for a :class:`CUDAArray` allocation.

    Mirrors ``CUarray_format`` from the CUDA driver API.
    """
    UINT8   = cydriver.CU_AD_FORMAT_UNSIGNED_INT8
    UINT16  = cydriver.CU_AD_FORMAT_UNSIGNED_INT16
    UINT32  = cydriver.CU_AD_FORMAT_UNSIGNED_INT32
    INT8    = cydriver.CU_AD_FORMAT_SIGNED_INT8
    INT16   = cydriver.CU_AD_FORMAT_SIGNED_INT16
    INT32   = cydriver.CU_AD_FORMAT_SIGNED_INT32
    FLOAT16 = cydriver.CU_AD_FORMAT_HALF
    FLOAT32 = cydriver.CU_AD_FORMAT_FLOAT


# Bytes per element (single channel) for each format.
_FORMAT_ELEM_SIZE = {
    int(ArrayFormat.UINT8):   1,
    int(ArrayFormat.INT8):    1,
    int(ArrayFormat.UINT16):  2,
    int(ArrayFormat.INT16):   2,
    int(ArrayFormat.FLOAT16): 2,
    int(ArrayFormat.UINT32):  4,
    int(ArrayFormat.INT32):   4,
    int(ArrayFormat.FLOAT32): 4,
}


def _validate_format_channels(format, num_channels):
    """Validate the ``(format, num_channels)`` pair shared by the array,
    mipmap, and texture factories. Raises on an invalid combination."""
    if not isinstance(format, ArrayFormat):
        raise TypeError(f"format must be an ArrayFormat, got {type(format).__name__}")
    if isinstance(num_channels, bool) or num_channels not in (1, 2, 4):
        raise ValueError(f"num_channels must be 1, 2, or 4, got {num_channels!r}")


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


cdef void _fill_array_endpoint(
    cydriver.CUDA_MEMCPY3D* p, CUDAArray arr, bint is_src
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


cdef _copy3d(CUDAArray arr, object other, Stream stream, bint to_array):
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


cdef class CUDAArray:
    """An opaque, hardware-laid-out GPU allocation for texture/surface access.

    Distinct from :class:`Buffer`: a ``CUarray`` has no exposed device pointer
    and can only be accessed from kernels through a :class:`TextureObject` or
    :class:`SurfaceObject`. Its memory layout is chosen by the driver for 2D/3D
    spatial locality.

    **Copy-only interop.** Because the layout is opaque and there is no linear
    device pointer, a ``CUDAArray`` cannot expose ``__cuda_array_interface__`` /
    DLPack and cannot be shared zero-copy with NumPy, CuPy, numba-cuda, or
    PyTorch. Moving data in or out is therefore always a copy: use
    :meth:`copy_from` / :meth:`copy_to` against a linear :class:`Buffer` or a
    host buffer-protocol object. There is no allocation helper — allocate the
    linear :class:`Buffer` yourself (e.g. ``mr.allocate(arr.size_bytes,
    stream=s)``) and copy.

    Construct via :meth:`from_descriptor`. Only plain 1D/2D/3D allocations are
    supported in this initial version; layered/cubemap/sparse variants will
    follow once their shape semantics are settled.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "CUDAArray cannot be instantiated directly. Use CUDAArray.from_descriptor()."
        )

    @classmethod
    def from_descriptor(cls, *, shape, format, num_channels, is_surface_load_store=False):
        """Allocate a new CUDA array.

        Parameters
        ----------
        shape : tuple of int
            ``(width,)``, ``(width, height)``, or ``(width, height, depth)``
            in elements.
        format : ArrayFormat
            Element format.
        num_channels : int
            Channels per element. Must be 1, 2, or 4.
        is_surface_load_store : bool
            If True, allocate with ``CUDA_ARRAY3D_SURFACE_LDST`` so the array
            can be bound as a :class:`SurfaceObject` for kernel-side writes.
            Default False.

        Returns
        -------
        CUDAArray
        """
        _validate_format_channels(format, num_channels)
        shape_t = _validate_array_shape(shape)

        cdef cydriver.CUarray_format c_format = <cydriver.CUarray_format><int>format
        cdef cydriver.CUDA_ARRAY3D_DESCRIPTOR desc3d
        cdef int rank = len(shape_t)
        cdef unsigned int flags = (
            cydriver.CUDA_ARRAY3D_SURFACE_LDST if is_surface_load_store else 0
        )

        # cuArray3DCreate handles 1D/2D/3D uniformly (Height/Depth 0 sentinels),
        # so a single descriptor + create_array_handle covers every shape.
        memset(&desc3d, 0, sizeof(desc3d))
        desc3d.Width = <size_t>shape_t[0]
        desc3d.Height = <size_t>(shape_t[1] if rank >= 2 else 0)
        desc3d.Depth = <size_t>(shape_t[2] if rank >= 3 else 0)
        desc3d.Format = c_format
        desc3d.NumChannels = <unsigned int>num_channels
        desc3d.Flags = flags

        cdef CUDAArrayHandle h = create_array_handle(desc3d)
        if not h:
            HANDLE_RETURN(get_last_error())

        cdef CUDAArray self = cls.__new__(cls)
        self._handle = h
        self._shape = shape_t
        self._format = c_format
        self._num_channels = num_channels
        self._surface_load_store = bool(is_surface_load_store)
        self._device_id = _get_current_device_id()
        return self

    @classmethod
    def _from_handle(cls, intptr_t handle, bint owning, *, device_id=None):
        """Wrap an externally-allocated ``CUarray``.

        Intended for graphics interop (``cuGraphicsSubResourceGetMappedArray``)
        where the array is owned by the graphics API. With ``owning=False`` the
        underlying ``CUarray`` is never destroyed by this object. Shape, format,
        and channel count are queried from the driver.
        """
        cdef cydriver.CUarray raw = <cydriver.CUarray><void*>handle
        cdef CUDAArrayHandle h
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
        """The element :class:`ArrayFormat`."""
        return ArrayFormat(self._format)

    @property
    def num_channels(self):
        """Channels per element (1, 2, or 4)."""
        return self._num_channels

    @property
    def element_size(self):
        """Bytes per element (format size * channels)."""
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
        """Total bytes of array storage (``prod(shape) * element_size``)."""
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
            f"CUDAArray(shape={self._shape}, "
            f"format={ArrayFormat(self._format).name}, "
            f"num_channels={self._num_channels})"
        )


cdef CUDAArray _array_from_handle(CUDAArrayHandle h, int device_id):
    """Wrap an existing CUDAArrayHandle as a CUDAArray, querying the driver for the
    array's shape/format/channels/surface-flag metadata.

    Any owning/non-owning semantics and parent (mipmap) dependency are already
    captured structurally inside ``h``'s C++ box.
    """
    if not h:
        HANDLE_RETURN(get_last_error())

    cdef CUDAArray self = CUDAArray.__new__(CUDAArray)
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
