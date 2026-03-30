# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from libc.limits cimport ULLONG_MAX
from libc.stdint cimport uintptr_t
from libc.string cimport memset

from cuda.bindings cimport cydriver
from cuda.core._memory._buffer cimport Buffer, Buffer_from_deviceptr_handle, MemoryResource
from cuda.core._memory cimport _ipc
from cuda.core._stream cimport default_stream, Stream_accept, Stream
from cuda.core._resource_handles cimport (
    MemoryPoolHandle,
    DevicePtrHandle,
    create_mempool_handle,
    create_mempool_handle_ref,
    deviceptr_alloc_from_pool,
    as_cu,
    as_py,
)

from cuda.core._utils.cuda_utils cimport (
    HANDLE_RETURN,
)


cdef class _MemPoolAttributes:
    """Provides access to memory pool attributes."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("_MemPoolAttributes cannot be instantiated directly. Please use MemoryResource APIs.")

    @staticmethod
    cdef _MemPoolAttributes _init(MemoryPoolHandle h_pool):
        cdef _MemPoolAttributes self = _MemPoolAttributes.__new__(_MemPoolAttributes)
        self._h_pool = h_pool
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(%s)" % ", ".join(
            f"{attr}={getattr(self, attr)}" for attr in dir(self)
                                            if not attr.startswith("_")
        )

    cdef int _getattribute(self, cydriver.CUmemPool_attribute attr_enum, void* value) except?-1:
        with nogil:
            HANDLE_RETURN(cydriver.cuMemPoolGetAttribute(as_cu(self._h_pool), attr_enum, value))
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
        # Note: subclasses use MP_init_create_pool or MP_init_current_pool to initialize.
        self._mempool_owned = False
        self._ipc_data = None
        self._attributes = None

    def close(self):
        """
        Close the memory resource and destroy the associated memory pool
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
            self._attributes = _MemPoolAttributes._init(self._h_pool)
        return self._attributes

    @property
    def handle(self) -> object:
        """Handle to the underlying memory pool."""
        return as_py(self._h_pool)

    @property
    def is_handle_owned(self) -> bool:
        """Whether the memory resource handle is owned. If False, ``close`` has no effect."""
        return self._mempool_owned

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
    def uuid(self) -> uuid.UUID | None:
        """
        A universally unique identifier for this memory resource. Meaningful
        only for IPC-enabled memory resources.
        """
        return getattr(self._ipc_data, 'uuid', None)


cdef int MP_init_create_pool(
    _MemPool self,
    cydriver.CUmemLocationType loc_type,
    int loc_id,
    cydriver.CUmemAllocationType alloc_type,
    bint ipc_enabled,
    size_t max_size,
) except? -1:
    """Initialize a _MemPool by creating a new memory pool with the given
    parameters.

    Sets ``_h_pool`` (owning), ``_mempool_owned``, and ``_ipc_data``.
    """
    cdef cydriver.CUmemPoolProps properties
    memset(&properties, 0, sizeof(cydriver.CUmemPoolProps))

    properties.allocType = alloc_type
    properties.handleTypes = (
        _ipc.IPC_HANDLE_TYPE if ipc_enabled
        else cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
    )
    properties.location.id = loc_id
    properties.location.type = loc_type
    properties.maxSize = max_size

    self._mempool_owned = True
    self._h_pool = create_mempool_handle(properties)

    if ipc_enabled:
        alloc_handle = _ipc.MP_export_mempool(self)
        self._ipc_data = _ipc.IPCDataForMR(alloc_handle, False)

    return 0


cdef int MP_init_current_pool(
    _MemPool self,
    cydriver.CUmemLocationType loc_type,
    int loc_id,
    cydriver.CUmemAllocationType alloc_type,
) except? -1:
    """Initialize a _MemPool by getting the driver's current pool for a
    location and allocation type.

    Sets ``_h_pool`` (non-owning) via ``cuMemGetMemPool``.
    Requires CUDA 13+.
    """
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        cdef cydriver.CUmemLocation loc
        cdef cydriver.CUmemoryPool pool
        loc.id = loc_id
        loc.type = loc_type
        with nogil:
            HANDLE_RETURN(cydriver.cuMemGetMemPool(&pool, &loc, alloc_type))
        self._h_pool = create_mempool_handle_ref(pool)
        self._mempool_owned = False
    ELSE:
        raise RuntimeError(
            "Getting the current memory pool requires CUDA 13.0 or later"
        )
    return 0


cdef int MP_raise_release_threshold(_MemPool self) except? -1:
    """Raise the pool's release threshold to ULLONG_MAX if currently zero.

    By default the release threshold is 0, meaning memory is returned to
    the OS as soon as there are no active suballocations.  Setting it to
    ULLONG_MAX avoids repeated OS round-trips.
    """
    cdef cydriver.cuuint64_t current_threshold
    cdef cydriver.cuuint64_t max_threshold = ULLONG_MAX
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

    # Reset members in declaration order.
    # The RAII deleter calls cuMemPoolDestroy if this is an owning handle.
    self._h_pool.reset()
    self._mempool_owned = False
    self._ipc_data = None
    self._attributes = None
