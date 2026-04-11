# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cuda.bindings cimport cydriver
from cuda.core._memory._memory_pool cimport (
    _MemPool, MP_init_create_pool, MP_raise_release_threshold,
)
from cuda.core._memory cimport _ipc
from cuda.core._memory._ipc cimport IPCAllocationHandle
from cuda.core._resource_handles cimport (
    as_cu,
    get_device_mempool,
)
from cuda.core._utils.cuda_utils cimport (
    check_or_create_options,
    HANDLE_RETURN,
)
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from dataclasses import dataclass
import multiprocessing
import platform  # no-cython-lint
import uuid

from ._peer_access_utils import plan_peer_access_update
from cuda.core._utils.cuda_utils import check_multiprocessing_start_method

__all__ = ['DeviceMemoryResource', 'DeviceMemoryResourceOptions']


@dataclass
cdef class DeviceMemoryResourceOptions:
    """Customizable :obj:`~_memory.DeviceMemoryResource` options.

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


cdef class DeviceMemoryResource(_MemPool):
    """
    A device memory resource managing a stream-ordered memory pool.

    Parameters
    ----------
    device_id : Device | int
        Device or Device ordinal for which a memory resource is constructed.

    options : DeviceMemoryResourceOptions
        Memory resource creation options.

        If set to `None`, the memory resource uses the driver's current
        stream-ordered memory pool for the specified `device_id`. If no memory
        pool is set as current, the driver's default memory pool for the device
        is used.

        If not set to `None`, a new memory pool is created, which is owned by
        the memory resource.

        When using an existing (current or default) memory pool, the returned
        device memory resource does not own the pool (`is_handle_owned` is
        `False`), and closing the resource has no effect.

    Notes
    -----
    To create an IPC-Enabled memory resource (MR) that is capable of sharing
    allocations between processes, specify ``ipc_enabled=True`` in the initializer
    option. Sharing an allocation is a two-step procedure that involves
    mapping a memory resource and then mapping buffers owned by that resource.
    These steps can be accomplished in several ways.

    An IPC-enabled memory resource can allocate memory buffers but cannot
    receive shared buffers. Mapping an MR to another process creates a "mapped
    memory resource" (MMR). An MMR cannot allocate memory buffers and can only
    receive shared buffers. MRs and MMRs are both of type
    :class:`DeviceMemoryResource` and can be distinguished via
    :attr:`DeviceMemoryResource.is_mapped`.

    An MR is shared via an allocation handle obtained by calling
    :meth:`DeviceMemoryResource.get_allocation_handle`. The allocation handle
    has a platform-specific interpretation; however, memory IPC is currently
    only supported for Linux, and in that case allocation handles are file
    descriptors. After sending an allocation handle to another process, it can
    be used to create an MMR by invoking
    :meth:`DeviceMemoryResource.from_allocation_handle`.

    Buffers can be shared as serializable descriptors obtained by calling
    :meth:`Buffer.get_ipc_descriptor`. In a receiving process, a shared buffer is
    created by invoking :meth:`Buffer.from_ipc_descriptor` with an MMR and
    buffer descriptor, where the MMR corresponds to the MR that created the
    described buffer.

    To help manage the association between memory resources and buffers, a
    registry is provided. Every MR has a unique identifier (UUID). MMRs can be
    registered by calling :meth:`DeviceMemoryResource.register` with the UUID
    of the corresponding MR. Registered MMRs can be looked up via
    :meth:`DeviceMemoryResource.from_registry`. When registering MMRs in this
    way, the use of buffer descriptors can be avoided. Instead, buffer objects
    can themselves be serialized and transferred directly. Serialization embeds
    the UUID, which is used to locate the correct MMR during reconstruction.

    IPC-enabled memory resources interoperate with the :mod:`multiprocessing`
    module to provide a simplified interface. This approach can avoid direct
    use of allocation handles, buffer descriptors, MMRs, and the registry. When
    using :mod:`multiprocessing` to spawn processes or send objects through
    communication channels such as :class:`multiprocessing.Queue`,
    :class:`multiprocessing.Pipe`, or :class:`multiprocessing.Connection`,
    :class:`Buffer` objects may be sent directly, and in such cases the process
    for creating MMRs and mapping buffers will be handled automatically.

    For greater efficiency when transferring many buffers, one may also send
    MRs and buffers separately. When an MR is sent via :mod:`multiprocessing`,
    an MMR is created and registered in the receiving process. Subsequently,
    buffers may be serialized and transferred using ordinary :mod:`pickle`
    methods.  The reconstruction procedure uses the registry to find the
    associated MMR.
    """

    def __cinit__(self, *args, **kwargs):
        self._dev_id = cydriver.CU_DEVICE_INVALID
        self._peer_accessible_by = None

    def __init__(self, device_id: Device | int, options=None):
        _DMR_init(self, device_id, options)

    def __reduce__(self):
        return DeviceMemoryResource.from_registry, (self.uuid,)

    @staticmethod
    def from_registry(uuid: uuid.UUID) -> DeviceMemoryResource:  # no-cython-lint
        """
        Obtain a registered mapped memory resource.

        Raises
        ------
        RuntimeError
            If no mapped memory resource is found in the registry.
        """
        return <DeviceMemoryResource>(_ipc.MP_from_registry(uuid))

    def register(self, uuid: uuid.UUID) -> DeviceMemoryResource:  # no-cython-lint
        """
        Register a mapped memory resource.

        Returns
        -------
        The registered mapped memory resource. If one was previously registered
        with the given key, it is returned.
        """
        return <DeviceMemoryResource>(_ipc.MP_register(self, uuid))

    @classmethod
    def from_allocation_handle(
        cls, device_id: Device | int, alloc_handle: int | IPCAllocationHandle
    ) -> DeviceMemoryResource:
        """Create a device memory resource from an allocation handle.

        Construct a new `DeviceMemoryResource` instance that imports a memory
        pool from a shareable handle. The memory pool is marked as owned, and
        the resource is associated with the specified `device_id`.

        Parameters
        ----------
        device_id : int | Device
            The ID of the device or a Device object for which the memory
            resource is created.

        alloc_handle : int | IPCAllocationHandle
            The shareable handle of the device memory resource to import. If an
            integer is supplied, it must represent a valid platform-specific
            handle. It is the caller's responsibility to close that handle.

        Returns
        -------
            A new device memory resource instance with the imported handle.
        """
        cdef DeviceMemoryResource mr = <DeviceMemoryResource>(
            _ipc.MP_from_allocation_handle(cls, alloc_handle))
        from .._device import Device
        mr._dev_id = Device(device_id).device_id
        mr._peer_accessible_by = ()
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
        """The associated device ordinal."""
        return self._dev_id

    @property
    def peer_accessible_by(self):
        """
        Get or set the devices that can access allocations from this memory
        pool. Access can be modified at any time and affects all allocations
        from this memory pool.

        Returns a tuple of sorted device IDs that currently have peer access to
        allocations from this memory pool.

        When setting, accepts a sequence of Device objects or device IDs.
        Setting to an empty sequence revokes all peer access.

        For non-owned pools (the default or current device pool), the state
        is always queried from the driver to reflect changes made by other
        wrappers or direct driver calls.

        Examples
        --------
        >>> dmr = DeviceMemoryResource(0)
        >>> dmr.peer_accessible_by = [1]  # Grant access to device 1
        >>> assert dmr.peer_accessible_by == (1,)
        >>> dmr.peer_accessible_by = []  # Revoke access
        """
        if not self._mempool_owned:
            _DMR_query_peer_access(self)
        return self._peer_accessible_by

    @peer_accessible_by.setter
    def peer_accessible_by(self, devices):
        _DMR_set_peer_accessible_by(self, devices)

    @property
    def is_device_accessible(self) -> bool:
        """Return True. This memory resource provides device-accessible buffers."""
        return True

    @property
    def is_host_accessible(self) -> bool:
        """Return False. This memory resource does not provide host-accessible buffers."""
        return False


cdef inline _DMR_query_peer_access(DeviceMemoryResource self):
    """Query the driver for the actual peer access state of this pool."""
    cdef int total
    cdef cydriver.CUmemAccess_flags flags
    cdef cydriver.CUmemLocation location
    cdef list peers = []

    with nogil:
        HANDLE_RETURN(cydriver.cuDeviceGetCount(&total))

    location.type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    for dev_id in range(total):
        if dev_id == self._dev_id:
            continue
        location.id = dev_id
        with nogil:
            HANDLE_RETURN(cydriver.cuMemPoolGetAccess(&flags, as_cu(self._h_pool), &location))
        if flags == cydriver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE:
            peers.append(dev_id)

    self._peer_accessible_by = tuple(sorted(peers))


cdef inline _DMR_set_peer_accessible_by(DeviceMemoryResource self, devices):
    from .._device import Device

    this_dev = Device(self._dev_id)
    cdef object resolve_device_id = lambda dev: Device(dev).device_id
    cdef object plan
    cdef tuple target_ids
    cdef tuple to_add
    cdef tuple to_rm
    if not self._mempool_owned:
        _DMR_query_peer_access(self)
    plan = plan_peer_access_update(
        owner_device_id=self._dev_id,
        current_peer_ids=self._peer_accessible_by,
        requested_devices=devices,
        resolve_device_id=resolve_device_id,
        can_access_peer=this_dev.can_access_peer,
    )
    target_ids = plan.target_ids
    to_add = plan.to_add
    to_rm = plan.to_remove
    cdef size_t count = len(to_add) + len(to_rm)
    cdef cydriver.CUmemAccessDesc* access_desc = NULL
    cdef size_t i = 0

    if count > 0:
        access_desc = <cydriver.CUmemAccessDesc*>PyMem_Malloc(count * sizeof(cydriver.CUmemAccessDesc))
        if access_desc == NULL:
            raise MemoryError("Failed to allocate memory for access descriptors")

        try:
            for dev_id in to_add:
                access_desc[i].flags = cydriver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                access_desc[i].location.type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
                access_desc[i].location.id = dev_id
                i += 1

            for dev_id in to_rm:
                access_desc[i].flags = cydriver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_NONE
                access_desc[i].location.type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
                access_desc[i].location.id = dev_id
                i += 1

            with nogil:
                HANDLE_RETURN(cydriver.cuMemPoolSetAccess(as_cu(self._h_pool), access_desc, count))
        finally:
            if access_desc != NULL:
                PyMem_Free(access_desc)

        self._peer_accessible_by = tuple(target_ids)


cdef inline _DMR_init(DeviceMemoryResource self, device_id, options):
    from .._device import Device
    cdef int dev_id = Device(device_id).device_id
    cdef DeviceMemoryResourceOptions opts = check_or_create_options(
        DeviceMemoryResourceOptions, options, "DeviceMemoryResource options",
        keep_none=True
    )
    cdef bint ipc_enabled = False
    cdef size_t max_size = 0

    self._dev_id = dev_id

    if opts is not None:
        ipc_enabled = opts.ipc_enabled
        if ipc_enabled and not _ipc.is_supported():
            raise RuntimeError(f"IPC is not available on {platform.system()}")
        max_size = opts.max_size

    if opts is None:
        self._h_pool = get_device_mempool(dev_id)
        self._mempool_owned = False
        MP_raise_release_threshold(self)
    else:
        self._peer_accessible_by = ()
        MP_init_create_pool(
            self,
            cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE,
            dev_id,
            cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED,
            ipc_enabled,
            max_size,
        )


# Note: this is referenced in instructions to debug nvbug 5698116.
cpdef DMR_mempool_get_access(DeviceMemoryResource dmr, int device_id):
    """
    Probes peer access from the given device using cuMemPoolGetAccess.

    Parameters
    ----------
    device_id : int or Device
        The device to query access for.

    Returns
    -------
    str
        Access permissions: "rw" for read-write, "r" for read-only, "" for no access.
    """
    from .._device import Device

    cdef int dev_id = Device(device_id).device_id
    cdef cydriver.CUmemAccess_flags flags
    cdef cydriver.CUmemLocation location

    location.type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    location.id = dev_id

    with nogil:
        HANDLE_RETURN(cydriver.cuMemPoolGetAccess(&flags, as_cu(dmr._h_pool), &location))

    if flags == cydriver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE:
        return "rw"
    elif flags == cydriver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ:
        return "r"
    else:
        return ""


def _deep_reduce_device_memory_resource(mr):
    check_multiprocessing_start_method()
    from .._device import Device
    device = Device(mr.device_id)
    alloc_handle = mr.get_allocation_handle()
    return mr.from_allocation_handle, (device, alloc_handle)


multiprocessing.reduction.register(DeviceMemoryResource, _deep_reduce_device_memory_resource)
