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
import os
import platform  # no-cython-lint
import subprocess
import threading
import uuid
import warnings

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
    """
    ipc_enabled : bool = False
    max_size : int = 0


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
    option. When IPC is enabled, the location type is automatically set to
    CU_MEM_LOCATION_TYPE_HOST_NUMA instead of CU_MEM_LOCATION_TYPE_HOST,
    with location ID 0.

    Note: IPC support for pinned memory requires a single NUMA node. A warning
    is issued if multiple NUMA nodes are detected.

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
    def is_device_accessible(self) -> bool:
        """Return True. This memory resource provides device-accessible buffers."""
        return True

    @property
    def is_host_accessible(self) -> bool:
        """Return True. This memory resource provides host-accessible buffers."""
        return True


cdef inline _PMR_init(PinnedMemoryResource self, options):
    cdef PinnedMemoryResourceOptions opts = check_or_create_options(
        PinnedMemoryResourceOptions, options, "PinnedMemoryResource options",
        keep_none=True
    )
    cdef bint ipc_enabled = False
    cdef size_t max_size = 0
    cdef cydriver.CUmemLocationType loc_type
    cdef int location_id

    if opts is not None:
        ipc_enabled = opts.ipc_enabled
        if ipc_enabled and not _ipc.is_supported():
            raise RuntimeError(f"IPC is not available on {platform.system()}")
        if ipc_enabled:
            _check_numa_nodes()
        max_size = opts.max_size

    if ipc_enabled:
        loc_type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
        location_id = 0
    else:
        loc_type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST
        location_id = -1

    if opts is None:
        MP_init_current_pool(
            self,
            loc_type,
            location_id,
            cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED,
        )
    else:
        MP_init_create_pool(
            self,
            loc_type,
            location_id,
            cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED,
            ipc_enabled,
            max_size,
        )


def _deep_reduce_pinned_memory_resource(mr):
    check_multiprocessing_start_method()
    alloc_handle = mr.get_allocation_handle()
    return mr.from_allocation_handle, (alloc_handle,)


multiprocessing.reduction.register(PinnedMemoryResource, _deep_reduce_pinned_memory_resource)


cdef bint _numa_warning_shown = False
cdef object _numa_lock = threading.Lock()


cdef inline _check_numa_nodes():
    """Check if system has multiple NUMA nodes and warn if so."""
    global _numa_warning_shown
    if _numa_warning_shown:
        return

    with _numa_lock:
        if _numa_warning_shown:
            return

        if platform.system() != "Linux":
            _numa_warning_shown = True
            return

        numa_count = None

        # Try /sys filesystem first (most reliable and doesn't require external tools)
        try:
            node_path = "/sys/devices/system/node"
            if os.path.exists(node_path):
                nodes = [d for d in os.listdir(node_path) if d.startswith("node") and d[4:].isdigit()]
                numa_count = len(nodes)
        except (OSError, PermissionError):
            pass

        # Fallback to lscpu if /sys check didn't work
        if numa_count is None:
            try:
                result = subprocess.run(
                    ["lscpu"],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                for line in result.stdout.splitlines():
                    if line.startswith("NUMA node(s):"):
                        numa_count = int(line.split(":")[1].strip())
                        break
            except (subprocess.SubprocessError, ValueError, FileNotFoundError):
                pass

        if numa_count is not None and numa_count > 1:
            warnings.warn(
                f"System has {numa_count} NUMA nodes. IPC-enabled pinned memory "
                f"uses location ID 0, which may not work correctly with multiple "
                f"NUMA nodes.",
                UserWarning,
                stacklevel=3
            )

        _numa_warning_shown = True
