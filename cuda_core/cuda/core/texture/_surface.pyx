# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from libc.string cimport memset

from cuda.bindings cimport cydriver
from cuda.core.texture._array cimport OpaqueArray
from cuda.core._resource_handles cimport (
    SurfObjectHandle,
    as_cu,
    as_intptr,
    create_surf_object_handle,
    get_last_error,
)
from cuda.core.texture._texture import ResourceDescriptor
from cuda.core._utils.cuda_utils cimport (
    HANDLE_RETURN,
    _get_current_device_id,
)


cdef class SurfaceObject:
    """A bindless surface handle for kernel-side typed load/store.

    Wraps ``cuSurfObjectCreate``. Unlike a :class:`TextureObject`, a surface
    has no sampling state (no filtering, no addressing modes, no normalization);
    kernels read and write through it using integer pixel coordinates.

    The backing :class:`OpaqueArray` must have been created with
    ``is_surface_load_store=True`` and is kept alive for the lifetime of this
    object to prevent dangling handles.

    Construct via :meth:`cuda.core.Device.create_surface_object`. Passes to
    kernels as a 64-bit handle (via the ``handle`` property).
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "SurfaceObject cannot be instantiated directly. "
            "Use Device.create_surface_object()."
        )

    @property
    def handle(self):
        """The underlying ``CUsurfObject`` as an integer (64-bit kernel arg)."""
        return as_intptr(self._handle)

    @property
    def resource(self):
        """The :class:`ResourceDescriptor` this surface was built from."""
        return self._source_ref

    @property
    def device(self):
        from cuda.core._device import Device
        return Device(self._device_id)

    cpdef close(self):
        """Release this object's reference to the underlying ``CUsurfObject``.

        Destruction (``cuSurfObjectDestroy``) and release of the backing array
        happen via the handle's deleter when the last reference is dropped.
        Idempotent.
        """
        self._handle.reset()
        self._source_ref = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __repr__(self):
        return f"SurfaceObject(handle=0x{as_intptr(self._handle):x})"


def _create_surface_object(resource):
    """Create a :class:`SurfaceObject` on the current device.

    Backs :meth:`cuda.core.Device.create_surface_object`. ``resource`` must be a
    :class:`ResourceDescriptor` wrapping an :class:`OpaqueArray` allocated with
    ``is_surface_load_store=True``; linear/pitch2d resources are not valid
    surface backings.
    """
    if not isinstance(resource, ResourceDescriptor):
        raise TypeError(
            f"resource must be a ResourceDescriptor, got "
            f"{type(resource).__name__}"
        )
    if resource.kind != "array":
        raise ValueError(
            f"SurfaceObject requires an array-backed ResourceDescriptor, "
            f"got kind={resource.kind!r}"
        )

    cdef OpaqueArray arr = <OpaqueArray>resource.source
    if not arr.is_surface_load_store:
        raise ValueError(
            "OpaqueArray must be created with is_surface_load_store=True to be "
            "bound as a SurfaceObject"
        )

    cdef cydriver.CUDA_RESOURCE_DESC res_desc
    memset(&res_desc, 0, sizeof(res_desc))
    res_desc.resType = cydriver.CU_RESOURCE_TYPE_ARRAY
    res_desc.res.array.hArray = as_cu(arr._handle)

    cdef SurfObjectHandle h = create_surf_object_handle(res_desc, arr._handle)
    if not h:
        HANDLE_RETURN(get_last_error())

    cdef SurfaceObject self = SurfaceObject.__new__(SurfaceObject)
    self._handle = h
    self._source_ref = resource
    self._device_id = _get_current_device_id()
    return self
