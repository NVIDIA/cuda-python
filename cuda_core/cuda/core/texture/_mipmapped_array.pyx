# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from libc.string cimport memset

from cuda.bindings cimport cydriver
from cuda.core.texture._array cimport _array_from_handle
from cuda.core.texture._array import ArrayFormat, _validate_array_shape, _validate_format_channels
from cuda.core._resource_handles cimport (
    CUDAArrayHandle,
    MipmappedArrayHandle,
    as_intptr,
    create_array_level_handle,
    create_mipmapped_array_handle,
    get_last_error,
)
from cuda.core._utils.cuda_utils cimport (
    HANDLE_RETURN,
    _get_current_device_id,
)


cdef class MipmappedArray:
    """A mipmapped CUDA array for texture/surface access across levels.

    Wraps ``CUmipmappedArray``. Each mip level is a distinct, hardware-laid-out
    allocation accessible only via a :class:`TextureObject` (or by retrieving
    the level's :class:`CUDAArray` and binding it as a :class:`SurfaceObject`).
    Destroying the :class:`MipmappedArray` destroys all level arrays
    implicitly, so the :class:`CUDAArray` instances returned by :meth:`get_level`
    are non-owning and hold a strong reference back to their parent.

    Construct via :meth:`from_descriptor`.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "MipmappedArray cannot be instantiated directly. "
            "Use MipmappedArray.from_descriptor()."
        )

    @classmethod
    def from_descriptor(
        cls, *, shape, format, num_channels, num_levels, is_surface_load_store=False
    ):
        """Allocate a new mipmapped CUDA array.

        Parameters
        ----------
        shape : tuple of int
            ``(width,)``, ``(width, height)``, or ``(width, height, depth)``
            in elements, for the base (level 0) mip.
        format : ArrayFormat
            Element format.
        num_channels : int
            Channels per element. Must be 1, 2, or 4.
        num_levels : int
            Number of mip levels to allocate; must be >= 1. The driver caps
            this at the log2 of the largest dimension; passing a larger value
            yields a driver error.
        is_surface_load_store : bool
            If True, allocate with ``CUDA_ARRAY3D_SURFACE_LDST`` so individual
            levels (obtained via :meth:`get_level`) can be bound as
            :class:`SurfaceObject` for kernel-side writes. Default False.

        Returns
        -------
        MipmappedArray
        """
        _validate_format_channels(format, num_channels)
        shape_t = _validate_array_shape(shape)

        levels = int(num_levels)
        if levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {levels}")

        cdef cydriver.CUarray_format c_format = <cydriver.CUarray_format><int>format
        cdef cydriver.CUDA_ARRAY3D_DESCRIPTOR desc3d
        cdef int rank = len(shape_t)
        cdef unsigned int flags = (
            cydriver.CUDA_ARRAY3D_SURFACE_LDST if is_surface_load_store else 0
        )
        cdef unsigned int c_levels = <unsigned int>levels

        # Mipmap creation uses the 3D descriptor regardless of rank; lower-rank
        # shapes use Height=0/Depth=0 sentinels, matching cuArray3DCreate.
        memset(&desc3d, 0, sizeof(desc3d))
        desc3d.Width = <size_t>shape_t[0]
        desc3d.Height = <size_t>(shape_t[1] if rank >= 2 else 0)
        desc3d.Depth = <size_t>(shape_t[2] if rank >= 3 else 0)
        desc3d.Format = c_format
        desc3d.NumChannels = <unsigned int>num_channels
        desc3d.Flags = flags

        cdef MipmappedArrayHandle h = create_mipmapped_array_handle(desc3d, c_levels)
        if not h:
            HANDLE_RETURN(get_last_error())

        cdef MipmappedArray self = cls.__new__(cls)
        self._handle = h
        self._shape = shape_t
        self._format = c_format
        self._num_channels = num_channels
        self._num_levels = <unsigned int>levels
        self._surface_load_store = bool(is_surface_load_store)
        self._device_id = _get_current_device_id()
        return self

    def get_level(self, level):
        """Return a non-owning :class:`CUDAArray` view of the given mip level.

        Parameters
        ----------
        level : int
            Mip level index in ``[0, num_levels)``.

        Returns
        -------
        CUDAArray
            A non-owning :class:`CUDAArray` wrapping the level's ``CUarray``.
            The :class:`MipmappedArray` is kept alive for the lifetime of the
            returned :class:`CUDAArray`; the underlying storage is released only
            when this :class:`MipmappedArray` is destroyed.
        """
        lvl = int(level)
        if lvl < 0:
            raise ValueError(f"level must be >= 0, got {lvl}")
        if lvl >= <int>self._num_levels:
            raise ValueError(
                f"level ({lvl}) must be < num_levels ({self._num_levels})"
            )

        cdef CUDAArrayHandle h_level = create_array_level_handle(self._handle, <unsigned int>lvl)
        if not h_level:
            HANDLE_RETURN(get_last_error())
        # The returned CUDAArray is non-owning; its C++ box embeds this mipmap's
        # handle, so the parent's storage structurally outlives the level view
        # (no Python parent reference needed).
        return _array_from_handle(h_level, self._device_id)

    @property
    def handle(self):
        """The underlying ``CUmipmappedArray`` as an integer."""
        return as_intptr(self._handle)

    @property
    def shape(self):
        """Base-level (level 0) allocation shape, in elements."""
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
    def num_levels(self):
        """Number of mip levels."""
        return int(self._num_levels)

    @property
    def is_surface_load_store(self):
        """True if this mipmap (and each of its levels) was created with
        ``CUDA_ARRAY3D_SURFACE_LDST`` and can back a :class:`SurfaceObject`."""
        return self._surface_load_store

    @property
    def device(self):
        """The :class:`Device` this mipmap was allocated on."""
        from cuda.core._device import Device
        return Device(self._device_id)

    cpdef close(self):
        """Release this object's reference to the underlying ``CUmipmappedArray``.

        Destruction (``cuMipmappedArrayDestroy``) happens via the handle's
        deleter when the last reference is dropped. A level :class:`CUDAArray`
        from :meth:`get_level` holds its own reference to this mipmap's storage,
        so it stays valid until both it and this object are released. Idempotent.
        """
        self._handle.reset()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __repr__(self):
        return (
            f"MipmappedArray(shape={self._shape}, "
            f"format={ArrayFormat(self._format).name}, "
            f"num_channels={self._num_channels}, "
            f"num_levels={self._num_levels})"
        )
