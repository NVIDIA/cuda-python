# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from libc.limits cimport ULLONG_MAX
from libc.stdint cimport uintptr_t, intptr_t
from libc.string cimport memset

from cuda.bindings cimport cydriver
from cuda.core.experimental._memory.buffer cimport Buffer, MemoryResource
from cuda.core.experimental._memory cimport ipc
from cuda.core.experimental._memory.ipc cimport IPCAllocationHandle
from cuda.core.experimental._stream cimport default_stream, Stream as _cyStream
from cuda.core.experimental._utils.cuda_utils cimport (
    _check_driver_error as raise_if_driver_error,
    check_or_create_options,
    HANDLE_RETURN,
)

import cython
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import platform
import weakref

from cuda.core.experimental._stream import Stream
from cuda.core.experimental._utils.cuda_utils import driver

if TYPE_CHECKING:
    from cuda.core.experimental._memory.buffer import DevicePointerT
    from .._device import Device
    import uuid


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
    ipc_enabled : cython.bint = False
    max_size : cython.size_t = 0


# TODO: cythonize this?
class DeviceMemoryResourceAttributes:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("DeviceMemoryResourceAttributes cannot be instantiated directly. Please use MemoryResource APIs.")

    @classmethod
    def _init(cls, mr : DeviceMemoryReference):
        self = DeviceMemoryResourceAttributes.__new__(cls)
        self._mr = mr
        return self

    def mempool_property(property_type: type):
        def decorator(stub):
            attr_enum = getattr(driver.CUmemPool_attribute, f"CU_MEMPOOL_ATTR_{stub.__name__.upper()}")

            def fget(self) -> property_type:
                mr = self._mr()
                if mr is None:
                    raise RuntimeError("DeviceMemoryResource is expired")
                # TODO: this implementation does not allow lowering to Cython + nogil
                err, value = driver.cuMemPoolGetAttribute(mr.handle, attr_enum)
                raise_if_driver_error(err)
                return property_type(value)
            return property(fget=fget, doc=stub.__doc__)
        return decorator

    @mempool_property(bool)
    def reuse_follow_event_dependencies(self):
        """Allow memory to be reused when there are event dependencies between streams."""

    @mempool_property(bool)
    def reuse_allow_opportunistic(self):
        """Allow reuse of completed frees without dependencies."""

    @mempool_property(bool)
    def reuse_allow_internal_dependencies(self):
        """Allow insertion of new stream dependencies for memory reuse."""

    @mempool_property(int)
    def release_threshold(self):
        """Amount of reserved memory to hold before OS release."""

    @mempool_property(int)
    def reserved_mem_current(self):
        """Current amount of backing memory allocated."""

    @mempool_property(int)
    def reserved_mem_high(self):
        """High watermark of backing memory allocated."""

    @mempool_property(int)
    def used_mem_current(self):
        """Current amount of memory in use."""

    @mempool_property(int)
    def used_mem_high(self):
        """High watermark of memory in use."""

    del mempool_property


cdef class DeviceMemoryResource(MemoryResource):
    """
    Create a device memory resource managing a stream-ordered memory pool.

    Parameters
    ----------
    device_id : int | Device
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

    def __cinit__(self):
        self._dev_id = cydriver.CU_DEVICE_INVALID
        self._mempool_handle = NULL
        self._attributes = None
        self._ipc_handle_type = cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_MAX
        self._mempool_owned = False
        self._is_mapped = False
        self._uuid = None
        self._alloc_handle = None

    def __init__(self, device_id: int | Device, options=None):
        cdef int dev_id = getattr(device_id, 'device_id', device_id)
        opts = check_or_create_options(
            DeviceMemoryResourceOptions, options, "DeviceMemoryResource options", keep_none=True
        )

        if opts is None:
            DMR_init_current(self, dev_id)
        else:
            DMR_init_create(self, dev_id, opts)

    def __dealloc__(self):
        DMR_close(self)

    def close(self):
        """Close the device memory resource and destroy the associated memory pool if owned."""
        DMR_close(self)

    def __reduce__(self):
        return DeviceMemoryResource.from_registry, (self.uuid,)

    @staticmethod
    def from_registry(uuid: uuid.UUID) -> DeviceMemoryResource:
        """
        Obtain a registered mapped memory resource.

        Raises
        ------
        RuntimeError
            If no mapped memory resource is found in the registry.
        """
        return ipc.DMR_from_registry(uuid)

    def register(self, uuid: uuid.UUID) -> DeviceMemoryResource:
        """
        Register a mapped memory resource.

        Returns
        -------
        The registered mapped memory resource. If one was previously registered
        with the given key, it is returned.
        """
        return ipc.DMR_register(self, uuid)

    @classmethod
    def from_allocation_handle(
        cls, device_id: int | Device, alloc_handle: int | IPCAllocationHandle
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
            The shareable handle of the device memory resource to import.

        Returns
        -------
            A new device memory resource instance with the imported handle.
        """
        return ipc.DMR_from_allocation_handle(cls, device_id, alloc_handle)

    def get_allocation_handle(self) -> IPCAllocationHandle:
        """Export the memory pool handle to be shared (requires IPC).

        The handle can be used to share the memory pool with other processes.
        The handle is cached in this `MemoryResource` and owned by it.

        Returns
        -------
            The shareable handle for the memory pool.
        """
        return ipc.DMR_get_allocation_handle(self)

    def allocate(self, size_t size, stream: Stream = None) -> Buffer:
        """Allocate a buffer of the requested size.

        Parameters
        ----------
        size : int
            The size of the buffer to allocate, in bytes.
        stream : Stream, optional
            The stream on which to perform the allocation asynchronously.
            If None, an internal stream is used.

        Returns
        -------
        Buffer
            The allocated buffer object, which is accessible on the device that this memory
            resource was created for.
        """
        if self._is_mapped:
            raise TypeError("Cannot allocate from a mapped IPC-enabled memory resource")
        if stream is None:
            stream = default_stream()
        return DMR_allocate(self, size, <_cyStream>stream)

    def deallocate(self, ptr: DevicePointerT, size_t size, stream: Stream = None):
        """Deallocate a buffer previously allocated by this resource.

        Parameters
        ----------
        ptr : :obj:`~_memory.DevicePointerT`
            The pointer or handle to the buffer to deallocate.
        size : int
            The size of the buffer to deallocate, in bytes.
        stream : Stream, optional
            The stream on which to perform the deallocation asynchronously.
            If the buffer is deallocated without an explicit stream, the allocation stream
            is used.
        """
        DMR_deallocate(self, <intptr_t>ptr, size, <_cyStream>stream)

    @property
    def attributes(self) -> DeviceMemoryResourceAttributes:
        if self._attributes is None:
            ref = weakref.ref(self)
            self._attributes = DeviceMemoryResourceAttributes._init(ref)
        return self._attributes

    @property
    def device_id(self) -> int:
        """The associated device ordinal."""
        return self._dev_id

    @property
    def handle(self) -> driver.CUmemoryPool:
        """Handle to the underlying memory pool."""
        return driver.CUmemoryPool(<uintptr_t>(self._mempool_handle))

    @property
    def is_device_accessible(self) -> bool:
        """Return True. This memory resource provides device-accessible buffers."""
        return True

    @property
    def is_handle_owned(self) -> bool:
        """Whether the memory resource handle is owned. If False, ``close`` has no effect."""
        return self._mempool_owned

    @property
    def is_host_accessible(self) -> bool:
        """Return False. This memory resource does not provide host-accessible buffers."""
        return False

    @property
    def is_ipc_enabled(self) -> bool:
        """Whether this memory resource has IPC enabled."""
        return self._ipc_handle_type != cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE

    @property
    def is_mapped(self) -> bool:
        """
        Whether this is a mapping of an IPC-enabled memory resource from
        another process.  If True, allocation is not permitted.
        """
        return self._is_mapped

    @property
    def uuid(self) -> Optional[uuid.UUID]:
        """
        A universally unique identifier for this memory resource. Meaningful
        only for IPC-enabled memory resources.
        """
        return self._uuid


cdef void DMR_init_current(DeviceMemoryResource self, int dev_id):
    # Get the current memory pool.
    cdef cydriver.cuuint64_t current_threshold
    cdef cydriver.cuuint64_t max_threshold = ULLONG_MAX

    self._dev_id = dev_id
    self._ipc_handle_type = cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
    self._mempool_owned = False

    with nogil:
        HANDLE_RETURN(cydriver.cuDeviceGetMemPool(&(self._mempool_handle), dev_id))

        # Set a higher release threshold to improve performance when there are no active allocations.
        # By default, the release threshold is 0, which means memory is immediately released back
        # to the OS when there are no active suballocations, causing performance issues.
        # Check current release threshold
        HANDLE_RETURN(cydriver.cuMemPoolGetAttribute(
            self._mempool_handle, cydriver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &current_threshold)
        )

        # If threshold is 0 (default), set it to maximum to retain memory in the pool
        if current_threshold == 0:
            HANDLE_RETURN(cydriver.cuMemPoolSetAttribute(
                self._mempool_handle,
                cydriver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &max_threshold
            ))


cdef void DMR_init_create(DeviceMemoryResource self, int dev_id, DeviceMemoryResourceOptions opts):
    # Create a new memory pool.
    cdef cydriver.CUmemPoolProps properties

    if opts.ipc_enabled and ipc.IPC_HANDLE_TYPE == cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE:
        raise RuntimeError("IPC is not available on {platform.system()}")

    memset(&properties, 0, sizeof(cydriver.CUmemPoolProps))
    properties.allocType = cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    properties.handleTypes = ipc.IPC_HANDLE_TYPE if opts.ipc_enabled else cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
    properties.location.id = dev_id
    properties.location.type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    properties.maxSize = opts.max_size
    properties.win32SecurityAttributes = NULL
    properties.usage = 0

    self._dev_id = dev_id
    self._ipc_handle_type = properties.handleTypes
    self._mempool_owned = True

    with nogil:
        HANDLE_RETURN(cydriver.cuMemPoolCreate(&(self._mempool_handle), &properties))
        # TODO: should we also set the threshold here?

    if opts.ipc_enabled:
        self.get_allocation_handle()  # enables Buffer.get_ipc_descriptor, sets uuid


cdef Buffer DMR_allocate(DeviceMemoryResource self, size_t size, _cyStream stream):
    cdef cydriver.CUstream s = stream._handle
    cdef cydriver.CUdeviceptr devptr
    with nogil:
        HANDLE_RETURN(cydriver.cuMemAllocFromPoolAsync(&devptr, size, self._mempool_handle, s))
    cdef Buffer buf = Buffer.__new__(Buffer)
    buf._ptr = <intptr_t>(devptr)
    buf._ptr_obj = None
    buf._size = size
    buf._mr = self
    buf._alloc_stream = stream
    return buf


cdef void DMR_deallocate(DeviceMemoryResource self, intptr_t ptr, size_t size, _cyStream stream) noexcept:
    cdef cydriver.CUstream s = stream._handle
    cdef cydriver.CUdeviceptr devptr = <cydriver.CUdeviceptr>ptr
    with nogil:
        HANDLE_RETURN(cydriver.cuMemFreeAsync(devptr, s))


cdef DMR_close(DeviceMemoryResource self):
    if self._mempool_handle == NULL:
        return

    try:
        if self._mempool_owned:
            with nogil:
                HANDLE_RETURN(cydriver.cuMemPoolDestroy(self._mempool_handle))
    finally:
        self._dev_id = cydriver.CU_DEVICE_INVALID
        self._mempool_handle = NULL
        self._attributes = None
        self._ipc_handle_type = cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_MAX
        self._mempool_owned = False
        self._is_mapped = False
        self._uuid = None
        self._alloc_handle = None

