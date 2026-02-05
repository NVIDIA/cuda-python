# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cuda.bindings cimport cydriver
from cuda.core._memory._memory_pool cimport _MemPool, _MemPoolOptions
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
from cuda.core._resource_handles cimport as_cu

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

    def __init__(self, device_id: Device | int, options=None):
        from .._device import Device
        cdef int dev_id = Device(device_id).device_id
        cdef DeviceMemoryResourceOptions opts = check_or_create_options(
            DeviceMemoryResourceOptions, options, "DeviceMemoryResource options",
            keep_none=True
        )
        cdef _MemPoolOptions opts_base = _MemPoolOptions()

        cdef bint ipc_enabled = False
        if opts:
            ipc_enabled = opts.ipc_enabled
            if ipc_enabled and not _ipc.is_supported():
                raise RuntimeError("IPC is not available on {platform.system()}")
            opts_base._max_size = opts.max_size
            opts_base._use_current = False
        opts_base._ipc_enabled = ipc_enabled
        opts_base._location = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        opts_base._type = cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED

        super().__init__(dev_id, opts_base)

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
    def is_device_accessible(self) -> bool:
        """Return True. This memory resource provides device-accessible buffers."""
        return True

    @property
    def is_host_accessible(self) -> bool:
        """Return False. This memory resource does not provide host-accessible buffers."""
        return False


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
