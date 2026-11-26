# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cuda.bindings cimport cydriver
from cuda.core._memory._buffer cimport Buffer
from cuda.core._memory._memory_pool cimport (
    _MemPool,
    _MP_allocate,
    MP_check_open,
    MP_init_create_pool,
    MP_init_current_pool,
    MP_set_access,
)
from cuda.core._memory cimport _ipc
from cuda.core._memory._ipc cimport IPCAllocationHandle
from cuda.core._memory._location cimport cumemlocation_from_id
from cuda.core._stream cimport Stream, Stream_accept
from cuda.core._utils.cuda_utils cimport (
    check_or_create_options,
    HANDLE_RETURN,
)
from libcpp.vector cimport vector

from dataclasses import dataclass
import multiprocessing
import platform  # no-cython-lint
import uuid

from cuda.core._utils.cuda_utils import check_multiprocessing_start_method

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuda.core.graph import GraphBuilder

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
    The device associated with ``stream`` must support host memory pools. If
    ``numa_id`` is set or derived for IPC, it must support host NUMA memory pools.
    You can query these capabilities through
    ``Device.properties.host_memory_pools_supported`` and
    ``Device.properties.host_numa_memory_pools_supported``. If the required pool
    is unsupported and stream-ordered allocation is not needed, use
    :class:`LegacyPinnedMemoryResource`.

    The driver makes a host memory pool accessible only from the host by
    default, so the resource grants read and write access to the pool to every
    device that supports this kind of host memory pool (host NUMA memory pools
    if the pool is placed on a NUMA node), as ``cudaMallocHost`` does. Access
    is per process, so a resource imported with :meth:`from_allocation_handle`,
    including by unpickling, makes the same grant in the importing process.
    Without options, the grant applies to the driver's current pool, which the
    whole process shares. Construction without options does not affect a CUDA
    graph capture in progress on any thread.

    To create an IPC-Enabled memory resource (MR) that is capable of sharing
    allocations between processes, specify ``ipc_enabled=True`` in the initializer
    option. When IPC is enabled and ``numa_id`` is not specified, the NUMA node
    is automatically derived from the current CUDA device's ``host_numa_id``
    attribute, which requires an active CUDA context. If ``numa_id`` is
    explicitly set, that value is used regardless of ``ipc_enabled``.

    See :class:`DeviceMemoryResource` for more details on IPC usage patterns.
    """

    def __init__(self, options: PinnedMemoryResourceOptions | None = None) -> None:
        _PMR_init(self, options)

    def allocate(self, size_t size, *, stream: Stream | GraphBuilder) -> Buffer:
        """Allocate a host-pinned buffer asynchronously on the supplied stream.

        Parameters
        ----------
        size : int
            The size of the buffer to allocate, in bytes.
        stream : :obj:`~_stream.Stream` | :obj:`~graph.GraphBuilder`
            Keyword-only. The stream on which to perform the allocation
            asynchronously. Must be passed explicitly; pass
            ``device.default_stream`` to use the default stream.

        Returns
        -------
        Buffer
            The allocated buffer object, which is accessible from the host.

        Raises
        ------
        RuntimeError
            If the stream's device does not support the requested host memory
            pool. Use :class:`LegacyPinnedMemoryResource` when stream-ordered
            allocation is not required.
        """
        MP_check_open(self)
        if self.is_mapped:
            raise TypeError("Cannot allocate from a mapped IPC-enabled memory resource")
        cdef Stream s = Stream_accept(stream)
        device = s.device
        if not _host_pool_supported(device, self._numa_id >= 0):
            raise RuntimeError(
                f"CUDA device {device.device_id} does not support the requested "
                "host memory pool for PinnedMemoryResource. Use "
                "LegacyPinnedMemoryResource if memory-pool features are not required."
            )
        return _MP_allocate(self, size, s)

    def __reduce__(self) -> tuple[object, ...]:
        MP_check_open(self)
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

        Raises
        ------
        RuntimeError
            If the memory resource is not IPC-enabled.
        ValueError
            If the memory resource already has a different UUID.
        """
        return <PinnedMemoryResource>(_ipc.MP_register(self, uuid))

    @classmethod
    def from_allocation_handle(
        cls, alloc_handle: int | IPCAllocationHandle, numa_id: int | None = None
    ) -> PinnedMemoryResource:
        """Create a host-pinned memory resource from an allocation handle.

        Construct a new `PinnedMemoryResource` instance that imports a memory
        pool from a shareable handle. The memory pool is marked as owned. Every
        device in this process that supports pools at the pool's host location
        is granted read and write access to the imported pool.

        Parameters
        ----------
        alloc_handle : int | IPCAllocationHandle
            The shareable handle of the host-pinned memory resource to import. If an
            integer is supplied, it must represent a valid platform-specific
            handle. It is the caller's responsibility to close that handle.
        numa_id : int | None, optional
            The host NUMA node the exporting process placed the pool on, that
            is, the exporting resource's :attr:`numa_id`. Node IDs are the same
            in every process on the host. The pool's node is read from the
            driver when possible (a CUDA 13 build of ``cuda.core`` and a 13.2
            or later driver); otherwise this value is the imported resource's
            :attr:`numa_id`. Unpickling a resource passes the exporter's node
            automatically if it was known. If the handle's resource is already
            registered in this process with an unknown node, this value becomes
            its node. If the node comes from neither the driver nor this
            argument, the import succeeds but :attr:`numa_id` raises
            ``RuntimeError``. Must be >= 0 if given.

        Returns
        -------
            A new host-pinned memory resource instance with the imported handle.

        Raises
        ------
        TypeError
            If the handle refers to a device memory pool, which must be
            imported with :meth:`DeviceMemoryResource.from_allocation_handle`.
            The pool's kind is checked with a CUDA 13 build of ``cuda.core``
            and a 13.2 or later driver. Otherwise it is checked only when the
            handle's memory resource is already registered in this process.
        ValueError
            If ``numa_id`` is negative, or differs from the node of the pool,
            as reported by the driver or by the resource already registered in
            this process for the handle.
        """
        if numa_id is not None and numa_id < 0:
            raise ValueError(f"numa_id must be >= 0, got {numa_id}")

        # cuMemPoolImportFromShareableHandle requires CUDA to be initialized, but in
        # a child process CUDA may not be initialized yet. For DeviceMemoryResource,
        # this is not a concern because most likely when retrieving the device_id the
        # user would have already initialized CUDA. But since PinnedMemoryResource is
        # not device-specific it is unlikelt the case.
        HANDLE_RETURN(cydriver.cuInit(0))

        cdef PinnedMemoryResource mr = <PinnedMemoryResource>(
            _ipc.MP_from_allocation_handle(cls, alloc_handle, _PMR_init_imported, numa_id))
        # The import hook does not run for a resource that is already registered,
        # whether found at the start or registered by a concurrent import.
        _PMR_apply_numa_id(mr, numa_id)
        return mr

    @property
    def allocation_handle(self) -> IPCAllocationHandle:
        """Shareable handle for this memory pool (requires IPC).

        The handle can be used to share the memory pool with other processes.
        The handle is cached in this `MemoryResource` and owned by it.
        """
        MP_check_open(self)
        if not self.is_ipc_enabled:
            raise RuntimeError("Memory resource is not IPC-enabled")
        return self._ipc_data._alloc_handle

    @property
    def device_id(self) -> int:
        """Return -1. Pinned memory is host memory and is not associated with a specific device."""
        return -1

    @property
    def numa_id(self) -> int:
        """The host NUMA node ID used for pool placement, or -1 for OS-managed placement.

        For a resource imported with :meth:`from_allocation_handle`, the node
        comes from the driver if the pool's location can be queried (a CUDA 13
        build of ``cuda.core`` and a 13.2 or later driver), otherwise from the
        ``numa_id`` argument of :meth:`from_allocation_handle`, which unpickling
        passes automatically if the exporter's node was known.

        Raises
        ------
        RuntimeError
            If the resource was imported and none of these sources gave the
            pool's node.
        """
        if not self._numa_id_known:
            raise RuntimeError(
                "The NUMA node of this imported memory pool is not known in this "
                "process. Pass numa_id to PinnedMemoryResource.from_allocation_handle "
                "to name the node the exporting process placed the pool on.")
        return self._numa_id

    @property
    def is_device_accessible(self) -> bool:
        """Return True. Every device that supports this kind of host memory pool can read and write its buffers."""
        return True

    @property
    def is_host_accessible(self) -> bool:
        """Return True. This memory resource provides host-accessible buffers."""
        return True


cdef inline bint _host_pool_supported(device, bint host_numa):
    return (
        device.properties.host_numa_memory_pools_supported
        if host_numa
        else device.properties.host_memory_pools_supported
    )


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
    self._numa_id_known = True

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

    _PMR_grant_device_access(self, numa_id >= 0, False)


cdef inline int _PMR_grant_device_access(
        PinnedMemoryResource self, bint host_numa, bint imported) except? -1:
    # A pool on a host location is accessible only from the host by default
    # (see cuMemPoolCreate), and access is per process, so both the creating
    # and the importing process grant read/write access to every device that
    # supports this kind of host pool, as cudaMallocHost memory is. Other
    # devices cannot use the pool.
    from .._device import Device

    cdef cydriver.CUmemAccessDesc desc
    cdef vector[cydriver.CUmemAccessDesc] access
    desc.flags = cydriver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    for dev in Device.get_all_devices():
        if _host_pool_supported(dev, host_numa):
            desc.location = cumemlocation_from_id(
                cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE, dev.device_id)
            access.push_back(desc)
    if not access.empty():
        MP_set_access(self, access.data(), access.size(), imported)
    return 0


cdef inline int _PMR_apply_numa_id(PinnedMemoryResource self, object numa_id) except? -1:
    # A node the caller passed must agree with a node that is known. A resource
    # whose node is unknown, such as one registered by an earlier import that
    # was given no node, takes the caller's node.
    if numa_id is None:
        return 0
    if not self._numa_id_known:
        self._numa_id = numa_id
        self._numa_id_known = True
    elif self._numa_id != numa_id:
        raise ValueError(
            f"numa_id {numa_id} does not match the NUMA node of the memory pool, "
            f"which is {self._numa_id}")
    return 0


cdef int _PMR_init_imported(_MemPool mr, object numa_id) except? -1:
    cdef PinnedMemoryResource self = <PinnedMemoryResource>mr
    cdef cydriver.CUmemLocation loc
    # IPC-enabled pinned pools created by cuda.core are always placed on a host
    # NUMA node. Without a location query (a CUDA 12 build or a driver before
    # 13.2), the pool's kind is not checked and the node is the one the caller
    # passed, if any. Otherwise it is unknown and the numa_id property raises.
    cdef bint host_numa = True
    if _ipc.MP_check_import_location(self, True, &loc):
        host_numa = loc.type == cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
        self._numa_id = loc.id if host_numa else -1
        self._numa_id_known = True
        _PMR_apply_numa_id(self, numa_id)
    elif numa_id is not None:
        self._numa_id = numa_id
        self._numa_id_known = True
    return _PMR_grant_device_access(self, host_numa, True)


def _deep_reduce_pinned_memory_resource(mr: object) -> tuple[object, ...]:
    check_multiprocessing_start_method()
    cdef PinnedMemoryResource pmr = <PinnedMemoryResource>mr
    alloc_handle = pmr.allocation_handle
    # NUMA node IDs are the same in every process on the host, so the node needs
    # no translation as a device ordinal does. An unknown node is not sent, so
    # that the importer does not mistake a placeholder for the pool's node.
    return pmr.from_allocation_handle, (alloc_handle, pmr._numa_id if pmr._numa_id_known else None)


multiprocessing.reduction.register(PinnedMemoryResource, _deep_reduce_pinned_memory_resource)
