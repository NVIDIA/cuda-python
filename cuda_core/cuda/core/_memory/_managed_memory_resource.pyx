# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cuda.bindings cimport cydriver
from cuda.core._memory._memory_pool cimport _MemPool, _MemPoolOptions
from cuda.core._utils.cuda_utils cimport (
    check_or_create_options,
)

from dataclasses import dataclass

__all__ = ['ManagedMemoryResource', 'ManagedMemoryResourceOptions']


@dataclass
cdef class ManagedMemoryResourceOptions:
    """Customizable :obj:`~_memory.ManagedMemoryResource` options.

    Attributes
    ----------
    preferred_location : int | None, optional
        The preferred device location for the managed memory.
        Use a device ID (0, 1, 2, ...) for device preference, -1 for CPU/host,
        or None to let the driver decide.
        (Default to None)
    """
    preferred_location: int | None = None


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
        cdef ManagedMemoryResourceOptions opts = check_or_create_options(
            ManagedMemoryResourceOptions, options, "ManagedMemoryResource options",
            keep_none=True
        )
        cdef _MemPoolOptions opts_base = _MemPoolOptions()

        cdef int device_id = -1
        cdef object preferred_location = None
        if opts:
            preferred_location = opts.preferred_location
            if preferred_location is not None:
                device_id = preferred_location
            opts_base._use_current = False

        opts_base._ipc_enabled = False  # IPC not supported for managed memory pools

        IF CUDA_CORE_BUILD_MAJOR >= 13:
            # Set location based on preferred_location
            if preferred_location is None:
                # Let the driver decide
                opts_base._location = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_NONE
            elif device_id == -1:
                # CPU/host preference
                opts_base._location = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST
            else:
                # Device preference
                opts_base._location = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE

            opts_base._type = cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_MANAGED
        ELSE:
            raise RuntimeError("ManagedMemoryResource requires CUDA 13.0 or later")

        super().__init__(device_id, opts_base)

    @property
    def is_device_accessible(self) -> bool:
        """Return True. This memory resource provides device-accessible buffers."""
        return True

    @property
    def is_host_accessible(self) -> bool:
        """Return True. This memory resource provides host-accessible buffers."""
        return True
