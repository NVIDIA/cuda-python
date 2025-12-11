# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cuda.bindings cimport cydriver
from cuda.core.experimental._memory._memory_pool cimport _MemPool, _MemPoolOptions
from cuda.core.experimental._memory cimport _ipc
from cuda.core.experimental._memory._ipc cimport IPCAllocationHandle
from cuda.core.experimental._utils.cuda_utils cimport (
    check_or_create_options,
)

from dataclasses import dataclass
from typing import Optional
import uuid

__all__ = ['ManagedMemoryResource', 'ManagedMemoryResourceOptions']


@dataclass
cdef class ManagedMemoryResourceOptions:
    """Customizable :obj:`~_memory.ManagedMemoryResource` options.

    Attributes
    ----------
    preferred_location : int, optional
        The preferred device location for the managed memory.
        Use a device ID (0, 1, 2, ...) for device preference, or -1 for CPU/host.
        (Default to -1 for CPU/host)

    max_size : int, optional
        Maximum pool size. When set to 0, defaults to a system-dependent value.
        (Default to 0)
    """
    preferred_location : int = -1
    max_size : int = 0


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

        cdef int device_id = -1  # Default: CPU/host preference
        if opts:
            device_id = opts.preferred_location
            opts_base._max_size = opts.max_size
            opts_base._use_current = False

        opts_base._ipc_enabled = False  # IPC not supported for managed memory pools

        # Set location based on preferred_location
        if device_id == -1:
            # CPU/host preference
            opts_base._location = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST
        else:
            # Device preference
            opts_base._location = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE

        opts_base._type = cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_MANAGED

        super().__init__(device_id, opts_base)

    def __reduce__(self):
        return ManagedMemoryResource.from_registry, (self.uuid,)

    @staticmethod
    def from_registry(uuid: uuid.UUID) -> ManagedMemoryResource:  # no-cython-lint
        """
        Obtain a registered mapped memory resource.

        Raises
        ------
        RuntimeError
            If no mapped memory resource is found in the registry.
        """
        return <ManagedMemoryResource>(_ipc.MP_from_registry(uuid))

    def register(self, uuid: uuid.UUID) -> ManagedMemoryResource:  # no-cython-lint
        """
        Register a mapped memory resource.

        Returns
        -------
        The registered mapped memory resource. If one was previously registered
        with the given key, it is returned.
        """
        return <ManagedMemoryResource>(_ipc.MP_register(self, uuid))

    @classmethod
    def from_allocation_handle(
        cls, alloc_handle: int | IPCAllocationHandle
    ) -> ManagedMemoryResource:
        """Create a managed memory resource from an allocation handle.

        Construct a new `ManagedMemoryResource` instance that imports a memory
        pool from a shareable handle. The memory pool is marked as owned.

        Parameters
        ----------
        alloc_handle : int | IPCAllocationHandle
            The shareable handle of the managed memory resource to import. If an
            integer is supplied, it must represent a valid platform-specific
            handle. It is the caller's responsibility to close that handle.

        Returns
        -------
            A new managed memory resource instance with the imported handle.
        """
        cdef ManagedMemoryResource mr = <ManagedMemoryResource>(
            _ipc.MP_from_allocation_handle(cls, alloc_handle))
        return mr

    def get_allocation_handle(self) -> IPCAllocationHandle:
        """Export the memory pool handle to be shared (requires IPC).

        The handle can be used to share the memory pool with other processes.
        The handle is cached in this `MemoryResource` and owned by it.

        Returns
        -------
            The shareable handle for the memory pool.

        Raises
        ------
        RuntimeError
            IPC is not currently supported for managed memory pools.
        """
        raise RuntimeError("IPC is not currently supported for managed memory pools")

    @property
    def is_device_accessible(self) -> bool:
        """Return True. This memory resource provides device-accessible buffers."""
        return True

    @property
    def is_host_accessible(self) -> bool:
        """Return True. This memory resource provides host-accessible buffers."""
        return True

    @property
    def is_ipc_enabled(self) -> bool:
        """Whether this memory resource has IPC enabled."""
        return self._ipc_data is not None

    @property
    def is_mapped(self) -> bool:
        """
        Whether this is a mapping of an IPC-enabled memory resource from
        another process.  If True, allocation is not permitted.
        """
        return self._ipc_data is not None and self._ipc_data._is_mapped

    @property
    def uuid(self) -> Optional[uuid.UUID]:
        """
        A universally unique identifier for this memory resource. Meaningful
        only for IPC-enabled memory resources.
        """
        return getattr(self._ipc_data, 'uuid', None)


def _deep_reduce_managed_memory_resource(mr):
    raise RuntimeError("IPC is not currently supported for managed memory pools")


# Multiprocessing support disabled until IPC is supported for managed memory pools
# multiprocessing.reduction.register(ManagedMemoryResource, _deep_reduce_managed_memory_resource)
