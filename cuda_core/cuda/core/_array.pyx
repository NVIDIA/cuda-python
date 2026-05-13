# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from libc.stdint cimport intptr_t
from libc.string cimport memset

from cuda.bindings cimport cydriver
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN

import enum


class ArrayFormat(enum.IntEnum):
    """Element format for a :class:`Array` allocation.

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


cdef inline intptr_t _get_current_context_ptr() except? 0:
    cdef cydriver.CUcontext ctx
    with nogil:
        HANDLE_RETURN(cydriver.cuCtxGetCurrent(&ctx))
    if ctx == NULL:
        raise RuntimeError("Array allocation requires an active CUDA context")
    return <intptr_t>ctx


cdef inline int _get_current_device_id() except -1:
    cdef cydriver.CUdevice dev
    with nogil:
        HANDLE_RETURN(cydriver.cuCtxGetDevice(&dev))
    return <int>dev


cdef class Array:
    """An opaque, hardware-laid-out GPU allocation for texture/surface access.

    Distinct from :class:`Buffer`: a ``CUarray`` has no exposed device pointer
    and can only be accessed from kernels through a :class:`TextureObject` or
    :class:`SurfaceObject`. Its memory layout is chosen by the driver for 2D/3D
    spatial locality.

    Construct via :meth:`from_descriptor`. Only plain 1D/2D/3D allocations are
    supported in this initial version; layered/cubemap/sparse variants will
    follow once their shape semantics are settled.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Array cannot be instantiated directly. Use Array.from_descriptor()."
        )

    @classmethod
    def from_descriptor(cls, *, shape, format, num_channels, surface_load_store=False):
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
        surface_load_store : bool
            If True, allocate with ``CUDA_ARRAY3D_SURFACE_LDST`` so the array
            can be bound as a :class:`SurfaceObject` for kernel-side writes.
            Default False.

        Returns
        -------
        Array
        """
        if not isinstance(format, ArrayFormat):
            raise TypeError(f"format must be an ArrayFormat, got {type(format)}")
        if num_channels not in (1, 2, 4):
            raise ValueError(f"num_channels must be 1, 2, or 4, got {num_channels}")

        try:
            shape_t = tuple(int(s) for s in shape)
        except TypeError as e:
            raise TypeError(f"shape must be a tuple of ints, got {type(shape)}") from e
        if not 1 <= len(shape_t) <= 3:
            raise ValueError(f"shape rank must be 1, 2, or 3, got {len(shape_t)}")
        for i, dim in enumerate(shape_t):
            if dim < 1:
                raise ValueError(f"shape[{i}] must be >= 1, got {dim}")

        cdef Array self = cls.__new__(cls)
        self._owning = True
        self._shape = shape_t
        self._format = int(format)
        self._num_channels = num_channels
        self._context = _get_current_context_ptr()
        self._device_id = _get_current_device_id()

        cdef cydriver.CUarray_format c_format = <cydriver.CUarray_format><int>format
        cdef cydriver.CUDA_ARRAY3D_DESCRIPTOR desc3d
        cdef cydriver.CUDA_ARRAY_DESCRIPTOR desc2d
        cdef int rank = len(shape_t)
        cdef unsigned int flags = (
            cydriver.CUDA_ARRAY3D_SURFACE_LDST if surface_load_store else 0
        )

        # cuArrayCreate (2D path) does not accept flags; use the 3D descriptor
        # whenever any flag is set or shape is 3D.
        if rank == 3 or flags != 0:
            memset(&desc3d, 0, sizeof(desc3d))
            desc3d.Width = <size_t>shape_t[0]
            desc3d.Height = <size_t>(shape_t[1] if rank >= 2 else 0)
            desc3d.Depth = <size_t>(shape_t[2] if rank >= 3 else 0)
            desc3d.Format = c_format
            desc3d.NumChannels = <unsigned int>num_channels
            desc3d.Flags = flags
            with nogil:
                HANDLE_RETURN(cydriver.cuArray3DCreate(&self._handle, &desc3d))
        else:
            memset(&desc2d, 0, sizeof(desc2d))
            desc2d.Width = <size_t>shape_t[0]
            desc2d.Height = <size_t>(shape_t[1] if rank == 2 else 0)
            desc2d.Format = c_format
            desc2d.NumChannels = <unsigned int>num_channels
            with nogil:
                HANDLE_RETURN(cydriver.cuArrayCreate(&self._handle, &desc2d))

        return self

    @classmethod
    def _from_handle(cls, intptr_t handle, bint owning, *, device_id=None):
        """Wrap an externally-allocated ``CUarray``.

        Intended for graphics interop (``cuGraphicsSubResourceGetMappedArray``)
        where the array is owned by the graphics API. With ``owning=False``,
        :meth:`close` and ``__dealloc__`` will not free the handle. Shape,
        format, and channel count are queried from the driver.
        """
        cdef Array self = cls.__new__(cls)
        self._handle = <cydriver.CUarray><void*>handle
        self._owning = owning
        self._context = _get_current_context_ptr()
        self._device_id = _get_current_device_id() if device_id is None else int(device_id)

        cdef cydriver.CUDA_ARRAY3D_DESCRIPTOR desc
        with nogil:
            HANDLE_RETURN(cydriver.cuArray3DGetDescriptor(&desc, self._handle))

        if desc.Depth > 0:
            self._shape = (int(desc.Width), int(desc.Height), int(desc.Depth))
        elif desc.Height > 0:
            self._shape = (int(desc.Width), int(desc.Height))
        else:
            self._shape = (int(desc.Width),)
        self._format = <int>desc.Format
        self._num_channels = desc.NumChannels
        return self

    @property
    def handle(self):
        """The underlying ``CUarray`` as an integer."""
        return <intptr_t>self._handle

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

    cpdef close(self):
        """Destroy the underlying ``CUarray`` if owned by this object."""
        if self._handle != NULL and self._owning:
            HANDLE_RETURN(cydriver.cuArrayDestroy(self._handle))
        self._handle = NULL

    def __dealloc__(self):
        # Cython destructors cannot raise; any cuArrayDestroy error here is
        # silently dropped. Callers needing visibility should use close().
        if self._handle != NULL and self._owning:
            cydriver.cuArrayDestroy(self._handle)
            self._handle = NULL

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __repr__(self):
        return (
            f"Array(shape={self._shape}, "
            f"format={ArrayFormat(self._format).name}, "
            f"num_channels={self._num_channels})"
        )
