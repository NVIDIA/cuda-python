# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cuda.bindings cimport cydriver

from cuda.core._memory._memory_pool cimport _MemPool, MP_init_create_pool, MP_init_current_pool
from cuda.core._utils.cuda_utils cimport (
    HANDLE_RETURN,
    check_or_create_options,
)
from cuda.core._utils.cuda_utils import CUDAError

from dataclasses import dataclass
import threading
import warnings

__all__ = ['ManagedMemoryResource', 'ManagedMemoryResourceOptions']


@dataclass
cdef class ManagedMemoryResourceOptions:
    """Customizable :obj:`~_memory.ManagedMemoryResource` options.

    Attributes
    ----------
    preferred_location : int | None, optional
        A location identifier (device ordinal or NUMA node ID) whose
        meaning depends on ``preferred_location_type``.
        (Default to ``None``)

    preferred_location_type : ``"device"`` | ``"host"`` | ``"host_numa"`` | None, optional
        Controls how ``preferred_location`` is interpreted.

        When set to ``None`` (the default), legacy behavior is used:
        ``preferred_location`` is interpreted as a device ordinal,
        ``-1`` for host, or ``None`` for no preference.

        When set explicitly, the type determines both the kind of
        preferred location and the valid values for
        ``preferred_location``:

        - ``"device"``: prefer a specific GPU. ``preferred_location``
          must be a device ordinal (``>= 0``).
        - ``"host"``: prefer host memory (OS-managed NUMA placement).
          ``preferred_location`` must be ``None``.
        - ``"host_numa"``: prefer a specific host NUMA node.
          ``preferred_location`` must be a NUMA node ID (``>= 0``),
          or ``None`` to derive the NUMA node from the current CUDA
          device's ``host_numa_id`` attribute (requires an active
          CUDA context).

        (Default to ``None``)
    """
    preferred_location: int | None = None
    preferred_location_type: str | None = None


cdef class ManagedMemoryResource(_MemPool):
    """
    A managed memory resource managing a stream-ordered memory pool.

    Managed memory is accessible from both the host and device, with automatic
    migration between them as needed.

    Parameters
    ----------
    options : ManagedMemoryResourceOptions
        Memory resource creation options.

        If set to `None`, the memory resource uses the driver's current
        stream-ordered memory pool. If no memory pool is set as current,
        the driver's default memory pool is used.

        If not set to `None`, a new memory pool is created, which is owned by
        the memory resource.

        When using an existing (current or default) memory pool, the returned
        managed memory resource does not own the pool (`is_handle_owned` is
        `False`), and closing the resource has no effect.

    Notes
    -----
    IPC (Inter-Process Communication) is not currently supported for managed
    memory pools.
    """

    def __init__(self, options=None):
        _MMR_init(self, options)

    @property
    def device_id(self) -> int:
        """The preferred device ordinal, or -1 if the preferred location is not a device."""
        if self._pref_loc_type == "device":
            return self._pref_loc_id
        return -1

    @property
    def preferred_location(self) -> tuple | None:
        """The preferred location for managed memory allocations.

        Returns ``None`` if no preferred location is set (driver decides),
        or a tuple ``(type, id)`` where *type* is one of ``"device"``,
        ``"host"``, or ``"host_numa"``, and *id* is the device ordinal,
        ``None`` (for ``"host"``), or the NUMA node ID, respectively.
        """
        if self._pref_loc_type is None:
            return None
        if self._pref_loc_type == "host":
            return ("host", None)
        return (self._pref_loc_type, self._pref_loc_id)

    @property
    def is_device_accessible(self) -> bool:
        """Return True. This memory resource provides device-accessible buffers."""
        return True

    @property
    def is_host_accessible(self) -> bool:
        """Return True. This memory resource provides host-accessible buffers."""
        return True


IF CUDA_CORE_BUILD_MAJOR >= 13:
    cdef tuple _VALID_LOCATION_TYPES = ("device", "host", "host_numa")


    cdef _resolve_preferred_location(ManagedMemoryResourceOptions opts):
        """Resolve preferred location options into driver and stored values.

        Returns a 4-tuple:
            (CUmemLocationType, loc_id, pref_loc_type_str, pref_loc_id)
        """
        cdef object pref_loc = opts.preferred_location if opts is not None else None
        cdef object pref_type = opts.preferred_location_type if opts is not None else None

        if pref_type is not None and pref_type not in _VALID_LOCATION_TYPES:
            raise ValueError(
                f"preferred_location_type must be one of {_VALID_LOCATION_TYPES!r} "
                f"or None, got {pref_type!r}"
            )

        if pref_type is None:
            # Legacy behavior
            if pref_loc is None:
                return (
                    cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_NONE,
                    -1, None, -1,
                )
            if pref_loc == -1:
                return (
                    cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST,
                    -1, "host", -1,
                )
            if pref_loc < 0:
                raise ValueError(
                    f"preferred_location must be a device ordinal (>= 0), -1 for "
                    f"host, or None for no preference, got {pref_loc}"
                )
            return (
                cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE,
                pref_loc, "device", pref_loc,
            )

        if pref_type == "device":
            if pref_loc is None or pref_loc < 0:
                raise ValueError(
                    f"preferred_location must be a device ordinal (>= 0) when "
                    f"preferred_location_type is 'device', got {pref_loc!r}"
                )
            return (
                cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE,
                pref_loc, "device", pref_loc,
            )

        if pref_type == "host":
            if pref_loc is not None:
                raise ValueError(
                    f"preferred_location must be None when "
                    f"preferred_location_type is 'host', got {pref_loc!r}"
                )
            return (
                cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST,
                -1, "host", -1,
            )

        # pref_type == "host_numa"
        if pref_loc is None:
            from .._device import Device
            dev = Device()
            numa_id = dev.properties.host_numa_id
            if numa_id < 0:
                raise RuntimeError(
                    "Cannot determine host NUMA ID for the current CUDA device. "
                    "The system may not support NUMA, or no CUDA context is "
                    "active. Set preferred_location to an explicit NUMA node ID "
                    "or call Device.set_current() first."
                )
            return (
                cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA,
                numa_id, "host_numa", numa_id,
            )
        if pref_loc < 0:
            raise ValueError(
                f"preferred_location must be a NUMA node ID (>= 0) or None "
                f"when preferred_location_type is 'host_numa', got {pref_loc}"
            )
        return (
            cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA,
            pref_loc, "host_numa", pref_loc,
        )


cdef inline _MMR_init(ManagedMemoryResource self, options):
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        cdef ManagedMemoryResourceOptions opts = check_or_create_options(
            ManagedMemoryResourceOptions, options, "ManagedMemoryResource options",
            keep_none=True
        )
        cdef cydriver.CUmemLocationType loc_type
        cdef int loc_id

        loc_type, loc_id, self._pref_loc_type, self._pref_loc_id = (
            _resolve_preferred_location(opts)
        )

        if opts is None:
            try:
                MP_init_current_pool(
                    self,
                    loc_type,
                    loc_id,
                    cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_MANAGED,
                )
            except CUDAError as e:
                if "CUDA_ERROR_NOT_SUPPORTED" in str(e):
                    from .._device import Device
                    if not Device().properties.concurrent_managed_access:
                        raise RuntimeError(
                            "The default memory pool on this device does not support "
                            "managed allocations (concurrent managed access is not "
                            "available). Use "
                            "ManagedMemoryResource(options=ManagedMemoryResourceOptions(...)) "
                            "to create a dedicated managed pool."
                        ) from e
                raise
        else:
            MP_init_create_pool(
                self,
                loc_type,
                loc_id,
                cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_MANAGED,
                False,
                0,
            )

        _check_concurrent_managed_access()
    ELSE:
        raise RuntimeError("ManagedMemoryResource requires CUDA 13.0 or later")


cdef bint _concurrent_access_warned = False
cdef object _concurrent_access_lock = threading.Lock()


cdef inline _check_concurrent_managed_access():
    """Warn once if the platform lacks concurrent managed memory access."""
    global _concurrent_access_warned
    if _concurrent_access_warned:
        return

    cdef int c_concurrent = 0
    with _concurrent_access_lock:
        if _concurrent_access_warned:
            return

        # concurrent_managed_access is a system-level attribute for sm_60 and
        # later, so any device will do.
        with nogil:
            HANDLE_RETURN(cydriver.cuDeviceGetAttribute(
                &c_concurrent,
                cydriver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
                0))
        if not c_concurrent:
            warnings.warn(
                "This platform does not support concurrent managed memory access "
                "(Device.properties.concurrent_managed_access is False). Host access to any managed "
                "allocation is forbidden while any GPU kernel is in flight, even "
                "if the kernel does not touch that allocation. Failing to "
                "synchronize before host access will cause a segfault. "
                "See: https://docs.nvidia.com/cuda/cuda-c-programming-guide/"
                "index.html#gpu-exclusive-access-to-managed-memory",
                UserWarning,
                stacklevel=3
            )

        _concurrent_access_warned = True


def reset_concurrent_access_warning():
    """Reset the concurrent access warning flag for testing purposes."""
    global _concurrent_access_warned
    _concurrent_access_warned = False
