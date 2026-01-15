# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from libc.limits cimport ULLONG_MAX
from libc.stdint cimport uintptr_t
from libc.string cimport memset
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from cuda.bindings cimport cydriver
from cuda.core._memory._buffer cimport Buffer, Buffer_from_deviceptr_handle, MemoryResource
from cuda.core._memory cimport _ipc
from cuda.core._stream cimport default_stream, Stream_accept, Stream
from cuda.core._resource_handles cimport (
    MemoryPoolHandle,
    DevicePtrHandle,
    create_mempool_handle,
    create_mempool_handle_ref,
    get_device_mempool,
    deviceptr_alloc_from_pool,
    as_cu,
    as_py,
)

from cuda.core._utils.cuda_utils cimport (
    HANDLE_RETURN,
)

from typing import TYPE_CHECKING
import platform  # no-cython-lint
import weakref

from cuda.core._utils.cuda_utils import driver

if TYPE_CHECKING:
    from cuda.core._memory.buffer import DevicePointerT
    from .._device import Device


cdef class _MemPoolOptions:

    def __cinit__(self):
        self._ipc_enabled = False
        self._max_size = 0
        self._location = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_INVALID
        self._type = cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_INVALID
        self._use_current = True


cdef class _MemPoolAttributes:
    cdef:
        object _mr_weakref

    def __init__(self, *args, **kwargs):
        raise RuntimeError("_MemPoolAttributes cannot be instantiated directly. Please use MemoryResource APIs.")

    @classmethod
    def _init(cls, mr):
        cdef _MemPoolAttributes self = _MemPoolAttributes.__new__(cls)
        self._mr_weakref = mr
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(%s)" % ", ".join(
            f"{attr}={getattr(self, attr)}" for attr in dir(self)
                                            if not attr.startswith("_")
        )

    cdef int _getattribute(self, cydriver.CUmemPool_attribute attr_enum, void* value) except?-1:
        cdef _MemPool mr = <_MemPool>(self._mr_weakref())
        if mr is None:
            raise RuntimeError("_MemPool is expired")
        cdef cydriver.CUmemoryPool pool_handle = as_cu(mr._h_pool)
        with nogil:
            HANDLE_RETURN(cydriver.cuMemPoolGetAttribute(pool_handle, attr_enum, value))
        return 0

    @property
    def reuse_follow_event_dependencies(self):
        """Allow memory to be reused when there are event dependencies between streams."""
        cdef int value
        self._getattribute(cydriver.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES, &value)
        return bool(value)

    @property
    def reuse_allow_opportunistic(self):
        """Allow reuse of completed frees without dependencies."""
        cdef int value
        self._getattribute(cydriver.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC, &value)
        return bool(value)

    @property
    def reuse_allow_internal_dependencies(self):
        """Allow insertion of new stream dependencies for memory reuse."""
        cdef int value
        self._getattribute(cydriver.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES, &value)
        return bool(value)

    @property
    def release_threshold(self):
        """Amount of reserved memory to hold before OS release."""
        cdef cydriver.cuuint64_t value
        self._getattribute(cydriver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &value)
        return int(value)

    @property
    def reserved_mem_current(self):
        """Current amount of backing memory allocated."""
        cdef cydriver.cuuint64_t value
        self._getattribute(cydriver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT, &value)
        return int(value)

    @property
    def reserved_mem_high(self):
        """High watermark of backing memory allocated."""
        cdef cydriver.cuuint64_t value
        self._getattribute(cydriver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH, &value)
        return int(value)

    @property
    def used_mem_current(self):
        """Current amount of memory in use."""
        cdef cydriver.cuuint64_t value
        self._getattribute(cydriver.CUmemPool_attribute.CU_MEMPOOL_ATTR_USED_MEM_CURRENT, &value)
        return int(value)

    @property
    def used_mem_high(self):
        """High watermark of memory in use."""
        cdef cydriver.cuuint64_t value
        self._getattribute(cydriver.CUmemPool_attribute.CU_MEMPOOL_ATTR_USED_MEM_HIGH, &value)
        return int(value)


cdef class _MemPool(MemoryResource):

    def __cinit__(self):
        self._dev_id = cydriver.CU_DEVICE_INVALID
        self._mempool_owned = False
        self._ipc_data = None
        self._attributes = None
        self._peer_accessible_by = ()

    def __init__(self, int device_id, _MemPoolOptions opts):
        if opts._use_current:
            _MP_init_current(self, device_id, opts)
        else:
            _MP_init_create(self, device_id, opts)

    def __dealloc__(self):
        _MP_close(self)

    def close(self):
        """
        Close the device memory resource and destroy the associated memory pool
        if owned.
        """
        _MP_close(self)

    def allocate(self, size_t size, stream: Stream | GraphBuilder | None = None) -> Buffer:
        """Allocate a buffer of the requested size.

        Parameters
        ----------
        size : int
            The size of the buffer to allocate, in bytes.
        stream : :obj:`~_stream.Stream` | :obj:`~_graph.GraphBuilder`, optional
            The stream on which to perform the allocation asynchronously.
            If None, an internal stream is used.

        Returns
        -------
        Buffer
            The allocated buffer object, which is accessible on the device that this memory
            resource was created for.
        """
        if self.is_mapped:
            raise TypeError("Cannot allocate from a mapped IPC-enabled memory resource")
        stream = Stream_accept(stream) if stream is not None else default_stream()
        return _MP_allocate(self, size, <Stream> stream)

    def deallocate(self, ptr: DevicePointerT, size_t size, stream: Stream | GraphBuilder | None = None):
        """Deallocate a buffer previously allocated by this resource.

        Parameters
        ----------
        ptr : :obj:`~_memory.DevicePointerT`
            The pointer or handle to the buffer to deallocate.
        size : int
            The size of the buffer to deallocate, in bytes.
        stream : :obj:`~_stream.Stream` | :obj:`~_graph.GraphBuilder`, optional
            The stream on which to perform the deallocation asynchronously.
            If the buffer is deallocated without an explicit stream, the allocation stream
            is used.
        """
        stream = Stream_accept(stream) if stream is not None else default_stream()
        _MP_deallocate(self, <uintptr_t>ptr, size, <Stream> stream)

    @property
    def attributes(self) -> _MemPoolAttributes:
        """Memory pool attributes."""
        if self._attributes is None:
            ref = weakref.ref(self)
            self._attributes = _MemPoolAttributes._init(ref)
        return self._attributes

    @property
    def device_id(self) -> int:
        """The associated device ordinal."""
        return self._dev_id

    @property
    def handle(self) -> object:
        """Handle to the underlying memory pool."""
        return as_py(self._h_pool)

    @property
    def is_handle_owned(self) -> bool:
        """Whether the memory resource handle is owned. If False, ``close`` has no effect."""
        return self._mempool_owned

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

        Examples
        --------
        >>> dmr = DeviceMemoryResource(0)
        >>> dmr.peer_accessible_by = [1]  # Grant access to device 1
        >>> assert dmr.peer_accessible_by == (1,)
        >>> dmr.peer_accessible_by = []  # Revoke access
        """
        return self._peer_accessible_by

    @peer_accessible_by.setter
    def peer_accessible_by(self, devices):
        """Set which devices can access this memory pool."""
        from .._device import Device

        # Convert all devices to device IDs
        cdef set[int] target_ids = {Device(dev).device_id for dev in devices}
        target_ids.discard(self._dev_id)  # exclude this device from peer access list
        this_dev = Device(self._dev_id)
        cdef list bad = [dev for dev in target_ids if not this_dev.can_access_peer(dev)]
        if bad:
            raise ValueError(f"Device {self._dev_id} cannot access peer(s): {', '.join(map(str, bad))}")
        cdef set[int] cur_ids = set(self._peer_accessible_by)
        cdef set[int] to_add = target_ids - cur_ids
        cdef set[int] to_rm = cur_ids - target_ids
        cdef size_t count = len(to_add) + len(to_rm) # transaction size
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


# _MemPool Implementation
# -----------------------

cdef int _MP_init_current(_MemPool self, int dev_id, _MemPoolOptions opts) except?-1:
    # Get the current memory pool.
    cdef cydriver.cuuint64_t current_threshold
    cdef cydriver.cuuint64_t max_threshold = ULLONG_MAX
    cdef cydriver.CUmemLocation loc
    cdef cydriver.CUmemoryPool pool

    self._dev_id = dev_id
    self._mempool_owned = False

    if opts._type == cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED \
            and opts._location == cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE:
        assert dev_id >= 0
        self._h_pool = get_device_mempool(dev_id)

        # Set a higher release threshold to improve performance when there are
        # no active allocations.  By default, the release threshold is 0, which
        # means memory is immediately released back to the OS when there are no
        # active suballocations, causing performance issues.
        with nogil:
            HANDLE_RETURN(
                cydriver.cuMemPoolGetAttribute(
                    as_cu(self._h_pool),
                    cydriver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &current_threshold
                )
            )
            if current_threshold == 0:
                HANDLE_RETURN(cydriver.cuMemPoolSetAttribute(
                    as_cu(self._h_pool),
                    cydriver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &max_threshold
                ))
    elif opts._type == cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED \
            and opts._location == cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST:
        IF CUDA_CORE_BUILD_MAJOR >= 13:
            assert dev_id == -1
            loc.id = dev_id
            loc.type = opts._location
            with nogil:
                HANDLE_RETURN(cydriver.cuMemGetMemPool(&pool, &loc, opts._type))
            self._h_pool = create_mempool_handle_ref(pool)
        ELSE:
            raise RuntimeError("not supported")
    elif opts._type == cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED \
            and opts._location == cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA:
        IF CUDA_CORE_BUILD_MAJOR >= 13:
            assert dev_id == 0
            loc.id = 0
            loc.type = opts._location
            with nogil:
                HANDLE_RETURN(cydriver.cuMemGetMemPool(&pool, &loc, opts._type))
            self._h_pool = create_mempool_handle_ref(pool)
        ELSE:
            raise RuntimeError("not supported")
    else:
        IF CUDA_CORE_BUILD_MAJOR >= 13:
            if opts._type == cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_MANAGED:
                # Managed memory pools
                loc.id = dev_id
                loc.type = opts._location
                with nogil:
                    HANDLE_RETURN(cydriver.cuMemGetMemPool(&pool, &loc, opts._type))
                self._h_pool = create_mempool_handle_ref(pool)
            else:
                assert False
        ELSE:
            assert False

    return 0


cdef int _MP_init_create(_MemPool self, int dev_id, _MemPoolOptions opts) except?-1:
    cdef cydriver.CUmemPoolProps properties
    memset(&properties, 0, sizeof(cydriver.CUmemPoolProps))

    cdef bint ipc_enabled = opts._ipc_enabled
    properties.allocType = opts._type
    properties.handleTypes = _ipc.IPC_HANDLE_TYPE if ipc_enabled else cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
    properties.location.id = dev_id
    properties.location.type = opts._location
    # managed memory does not support maxSize as of CUDA 13.0
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        if properties.allocType != cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_MANAGED:
            properties.maxSize = opts._max_size
    ELSE:
        properties.maxSize = opts._max_size

    self._dev_id = dev_id
    self._mempool_owned = True

    self._h_pool = create_mempool_handle(properties)

    if ipc_enabled:
        alloc_handle = _ipc.MP_export_mempool(self)
        self._ipc_data = _ipc.IPCDataForMR(alloc_handle, False)

    return 0


# Raise an exception if the given stream is capturing.
# A result of CU_STREAM_CAPTURE_STATUS_INVALIDATED is considered an error.
cdef inline int check_not_capturing(cydriver.CUstream s) except?-1 nogil:
    cdef cydriver.CUstreamCaptureStatus capturing
    HANDLE_RETURN(cydriver.cuStreamIsCapturing(s, &capturing))
    if capturing != cydriver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE:
        raise RuntimeError("_MemPool cannot perform memory operations on "
                           "a capturing stream (consider using GraphMemoryResource).")


cdef inline Buffer _MP_allocate(_MemPool self, size_t size, Stream stream):
    cdef cydriver.CUstream s = as_cu(stream._h_stream)
    cdef DevicePtrHandle h_ptr
    with nogil:
        check_not_capturing(s)
        h_ptr = deviceptr_alloc_from_pool(size, self._h_pool, stream._h_stream)
    if not h_ptr:
        raise RuntimeError("Failed to allocate memory from pool")
    return Buffer_from_deviceptr_handle(h_ptr, size, self, None)


cdef inline void _MP_deallocate(
    _MemPool self, uintptr_t ptr, size_t size, Stream stream
) noexcept nogil:
    cdef cydriver.CUstream s = as_cu(stream._h_stream)
    cdef cydriver.CUdeviceptr devptr = <cydriver.CUdeviceptr>ptr
    cdef cydriver.CUresult r
    with nogil:
        r = cydriver.cuMemFreeAsync(devptr, s)
        if r != cydriver.CUDA_ERROR_INVALID_CONTEXT:
            HANDLE_RETURN(r)


cdef inline _MP_close(_MemPool self):
    if not self._h_pool:
        return

    # This works around nvbug 5698116. When a memory pool handle is recycled
    # the new handle inherits the peer access state of the previous handle.
    if self._peer_accessible_by:
        self.peer_accessible_by = []

    # Reset members in declaration order.
    # The RAII deleter handles nvbug 5698116 workaround (clears peer access)
    # and calls cuMemPoolDestroy if this is an owning handle.
    self._h_pool.reset()
    self._dev_id = cydriver.CU_DEVICE_INVALID
    self._mempool_owned = False
    self._ipc_data = None
    self._attributes = None
    self._peer_accessible_by = ()
