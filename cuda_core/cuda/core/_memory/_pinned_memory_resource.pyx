# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cuda.bindings cimport cydriver
from cuda.core._memory._memory_pool cimport _MemPool, MP_init_create_pool, MP_init_current_pool
from cuda.core._memory cimport _ipc
from cuda.core._memory._ipc cimport IPCAllocationHandle
from cuda.core._utils.cuda_utils cimport (
    check_or_create_options,
    HANDLE_RETURN,
)

from dataclasses import dataclass
import multiprocessing
import platform  # no-cython-lint
import uuid

from cuda.core._utils.cuda_utils import check_multiprocessing_start_method

__all__ = ['PinnedMemoryResource', 'PinnedMemoryResourceOptions']


@dataclass
cdef class PinnedMemoryResourceOptions:
    """Customizable :obj:`~_memory.PinnedMemoryResource` options.

    Attributes
    ----------
    ipc_enabled : bool, optional
        Specifies whether to create an IPC-enabled memory pool. When set to
        True, the memory pool and its allocations can be shared with other
        processes. (Default to False)

    max_size : int, optional
        Maximum pool size. When set to 0, defaults to a system-dependent value.
        (Default to 0)

    numa_id : int or None, optional
        Host NUMA node ID for pool placement. When set to None (the default),
        the behavior depends on ``ipc_enabled``:

        - ``ipc_enabled=False``: OS-managed placement (location type HOST).
        - ``ipc_enabled=True``: automatically derived from the current CUDA
          device's ``host_numa_id`` attribute, requiring an active CUDA
          context.

        When set to a non-negative integer, that NUMA node is used explicitly
        regardless of ``ipc_enabled`` (location type HOST_NUMA).
    """
    ipc_enabled : bool = False
    max_size : int = 0
    numa_id : int | None = None


cdef class PinnedMemoryResource(_MemPool):
    """
    A host-pinned memory resource managing a stream-ordered memory pool.

    Parameters
    ----------
    options : PinnedMemoryResourceOptions
        Memory resource creation options.

        If set to `None`, the memory resource uses the driver's current
        stream-ordered memory pool. If no memory
        pool is set as current, the driver's default memory pool
        is used.

        If not set to `None`, a new memory pool is created, which is owned by
        the memory resource.

        When using an existing (current or default) memory pool, the returned
        host-pinned memory resource does not own the pool (`is_handle_owned` is
        `False`), and closing the resource has no effect.

    Notes
    -----
    To create an IPC-Enabled memory resource (MR) that is capable of sharing
    allocations between processes, specify ``ipc_enabled=True`` in the initializer
    option. When IPC is enabled and ``numa_id`` is not specified, the NUMA node
    is automatically derived from the current CUDA device's ``host_numa_id``
    attribute, which requires an active CUDA context. If ``numa_id`` is
    explicitly set, that value is used regardless of ``ipc_enabled``.

    See :class:`DeviceMemoryResource` for more details on IPC usage patterns.
    """

    def __init__(self, options=None):
        _PMR_init(self, options)

    def __reduce__(self):
        return PinnedMemoryResource.from_registry, (self.uuid,)

    @staticmethod
    def from_registry(uuid: uuid.UUID) -> PinnedMemoryResource:  # no-cython-lint
        """
        Obtain a registered mapped memory resource.

        Raises
        ------
        RuntimeError
            If no mapped memory resource is found in the registry.
        """
        return <PinnedMemoryResource>(_ipc.MP_from_registry(uuid))

    def register(self, uuid: uuid.UUID) -> PinnedMemoryResource:  # no-cython-lint
        """
        Register a mapped memory resource.

        Returns
        -------
        The registered mapped memory resource. If one was previously registered
        with the given key, it is returned.
        """
        return <PinnedMemoryResource>(_ipc.MP_register(self, uuid))

    @classmethod
    def from_allocation_handle(
        cls, alloc_handle: int | IPCAllocationHandle
    ) -> PinnedMemoryResource:
        """Create a host-pinned memory resource from an allocation handle.

        Construct a new `PinnedMemoryResource` instance that imports a memory
        pool from a shareable handle. The memory pool is marked as owned.

        Parameters
        ----------
        alloc_handle : int | IPCAllocationHandle
            The shareable handle of the host-pinned memory resource to import. If an
            integer is supplied, it must represent a valid platform-specific
            handle. It is the caller's responsibility to close that handle.

        Returns
        -------
            A new host-pinned memory resource instance with the imported handle.
        """
        # cuMemPoolImportFromShareableHandle requires CUDA to be initialized, but in
        # a child process CUDA may not be initialized yet. For DeviceMemoryResource,
        # this is not a concern because most likely when retrieving the device_id the
        # user would have already initialized CUDA. But since PinnedMemoryResource is
        # not device-specific it is unlikelt the case.
        HANDLE_RETURN(cydriver.cuInit(0))

        cdef PinnedMemoryResource mr = <PinnedMemoryResource>(
            _ipc.MP_from_allocation_handle(cls, alloc_handle))
        return mr

    def get_allocation_handle(self) -> IPCAllocationHandle:
        """Export the memory pool handle to be shared (requires IPC).

        The handle can be used to share the memory pool with other processes.
        The handle is cached in this `MemoryResource` and owned by it.

        Returns
        -------
            The shareable handle for the memory pool.
        """
        if not self.is_ipc_enabled:
            raise RuntimeError("Memory resource is not IPC-enabled")
        return self._ipc_data._alloc_handle

    @property
    def device_id(self) -> int:
        """Return -1. Pinned memory is host memory and is not associated with a specific device."""
        return -1

    @property
    def numa_id(self) -> int:
        """The host NUMA node ID used for pool placement, or -1 for OS-managed placement."""
        return self._numa_id

    @property
    def is_device_accessible(self) -> bool:
        """Return True. This memory resource provides device-accessible buffers."""
        return True

    @property
    def is_host_accessible(self) -> bool:
        """Return True. This memory resource provides host-accessible buffers."""
        return True


cdef inline _PMR_init(PinnedMemoryResource self, options):
    from .._device import Device

    cdef PinnedMemoryResourceOptions opts = check_or_create_options(
        PinnedMemoryResourceOptions, options, "PinnedMemoryResource options",
        keep_none=True
    )
    cdef bint ipc_enabled = False
    cdef size_t max_size = 0
    cdef cydriver.CUmemLocationType loc_type
    cdef int numa_id = -1

    if opts is not None:
        ipc_enabled = opts.ipc_enabled
        if ipc_enabled and not _ipc.is_supported():
            raise RuntimeError(f"IPC is not available on {platform.system()}")
        max_size = opts.max_size

        if opts.numa_id is not None:
            numa_id = opts.numa_id
            if numa_id < 0:
                raise ValueError(f"numa_id must be >= 0, got {numa_id}")
        elif ipc_enabled:
            dev = Device()
            numa_id = dev.properties.host_numa_id
            if numa_id < 0:
                raise RuntimeError(
                    "Cannot determine host NUMA ID for IPC-enabled pinned "
                    "memory pool. The system may not support NUMA, or no "
                    "CUDA context is active. Set numa_id explicitly or "
                    "call Device.set_current() first.")

    if numa_id >= 0:
        loc_type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
    else:
        loc_type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST

    self._numa_id = numa_id

    if opts is None:
        MP_init_current_pool(
            self,
            loc_type,
            numa_id,
            cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED,
        )
    else:
        MP_init_create_pool(
            self,
            loc_type,
            numa_id,
            cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED,
            ipc_enabled,
            max_size,
        )


def _deep_reduce_pinned_memory_resource(mr):
    check_multiprocessing_start_method()
    alloc_handle = mr.get_allocation_handle()
    return mr.from_allocation_handle, (alloc_handle,)


multiprocessing.reduction.register(PinnedMemoryResource, _deep_reduce_pinned_memory_resource)
