# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from libc.stdint cimport intptr_t
from libc.string cimport memset

from cuda.bindings cimport cydriver
from cuda.core._array cimport Array
from cuda.core._texture import ResourceDescriptor
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN


cdef inline intptr_t _get_current_context_ptr() except? 0:
    cdef cydriver.CUcontext ctx
    with nogil:
        HANDLE_RETURN(cydriver.cuCtxGetCurrent(&ctx))
    if ctx == NULL:
        raise RuntimeError("SurfaceObject requires an active CUDA context")
    return <intptr_t>ctx


cdef inline int _get_current_device_id() except -1:
    cdef cydriver.CUdevice dev
    with nogil:
        HANDLE_RETURN(cydriver.cuCtxGetDevice(&dev))
    return <int>dev


cdef class SurfaceObject:
    """A bindless surface handle for kernel-side typed load/store.

    Wraps ``cuSurfObjectCreate``. Unlike a :class:`TextureObject`, a surface
    has no sampling state (no filtering, no addressing modes, no normalization);
    kernels read and write through it using integer pixel coordinates.

    The backing :class:`Array` must have been created with
    ``surface_load_store=True`` and is kept alive for the lifetime of this
    object to prevent dangling handles.

    Construct via :meth:`from_array` or :meth:`from_descriptor`. Passes to
    kernels as a 64-bit handle (via the ``handle`` property).
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "SurfaceObject cannot be instantiated directly. "
            "Use SurfaceObject.from_array() or SurfaceObject.from_descriptor()."
        )

    @classmethod
    def from_array(cls, array):
        """Create a surface object directly from an :class:`Array`.

        The array must have been created with ``surface_load_store=True``.
        """
        if not isinstance(array, Array):
            raise TypeError(f"array must be an Array, got {type(array).__name__}")
        return cls.from_descriptor(ResourceDescriptor.from_array(array))

    @classmethod
    def from_descriptor(cls, resource_desc):
        """Create a surface object from a :class:`ResourceDescriptor`.

        Parameters
        ----------
        resource_desc : ResourceDescriptor
            Must wrap an :class:`Array` allocated with
            ``surface_load_store=True``. Linear/pitch2d resources are not
            valid surface backings.
        """
        if not isinstance(resource_desc, ResourceDescriptor):
            raise TypeError(
                f"resource_desc must be a ResourceDescriptor, got "
                f"{type(resource_desc).__name__}"
            )
        if resource_desc.kind != "array":
            raise ValueError(
                f"SurfaceObject requires an array-backed ResourceDescriptor, "
                f"got kind={resource_desc.kind!r}"
            )

        cdef Array arr = <Array>resource_desc.source
        if not arr.surface_load_store:
            raise ValueError(
                "Array must be created with surface_load_store=True to be "
                "bound as a SurfaceObject"
            )

        cdef cydriver.CUDA_RESOURCE_DESC res_desc
        memset(&res_desc, 0, sizeof(res_desc))
        res_desc.resType = cydriver.CU_RESOURCE_TYPE_ARRAY
        res_desc.res.array.hArray = arr._handle

        cdef SurfaceObject self = cls.__new__(cls)
        self._source_ref = resource_desc
        self._context = _get_current_context_ptr()
        self._device_id = _get_current_device_id()

        with nogil:
            HANDLE_RETURN(
                cydriver.cuSurfObjectCreate(&self._handle, &res_desc)
            )
        return self

    @property
    def handle(self):
        """The underlying ``CUsurfObject`` as an integer (64-bit kernel arg)."""
        return <intptr_t>self._handle

    @property
    def resource(self):
        """The :class:`ResourceDescriptor` this surface was built from."""
        return self._source_ref

    @property
    def device(self):
        from cuda.core._device import Device
        return Device(self._device_id)

    cpdef close(self):
        """Destroy the underlying ``CUsurfObject``."""
        if self._handle != 0:
            HANDLE_RETURN(cydriver.cuSurfObjectDestroy(self._handle))
        self._handle = 0
        self._source_ref = None

    def __dealloc__(self):
        # Cython destructors cannot raise; any cuSurfObjectDestroy error is
        # silently dropped. Callers needing visibility should use close().
        if self._handle != 0:
            cydriver.cuSurfObjectDestroy(self._handle)
            self._handle = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __repr__(self):
        return f"SurfaceObject(handle=0x{<intptr_t>self._handle:x})"
