# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from libc.stdint cimport intptr_t

from cuda.bindings cimport cydriver
from cuda.core.experimental._memory._buffer cimport Buffer, MemoryResource
from cuda.core.experimental._resource_handles cimport native
from cuda.core.experimental._stream cimport default_stream, Stream_accept, Stream
from cuda.core.experimental._utils.cuda_utils cimport HANDLE_RETURN

from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuda.core.experimental._memory.buffer import DevicePointerT

__all__ = ['GraphMemoryResource']


cdef class GraphMemoryResourceAttributes:
    cdef:
        int _dev_id

    def __init__(self, *args, **kwargs):
        raise RuntimeError("GraphMemoryResourceAttributes cannot be instantiated directly. Please use MemoryResource APIs.")

    @classmethod
    def _init(cls, device_id: int):
        cdef GraphMemoryResourceAttributes self = GraphMemoryResourceAttributes.__new__(cls)
        self._dev_id = device_id
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(%s)" % ", ".join(
            f"{attr}={getattr(self, attr)}" for attr in dir(self)
                                            if not attr.startswith("_")
        )

    cdef int _getattribute(self, cydriver.CUgraphMem_attribute attr_enum, void* value) except?-1:
        with nogil:
            HANDLE_RETURN(cydriver.cuDeviceGetGraphMemAttribute(self._dev_id, attr_enum, value))
        return 0

    cdef int _setattribute(self, cydriver.CUgraphMem_attribute attr_enum, void* value) except?-1:
        with nogil:
            HANDLE_RETURN(cydriver.cuDeviceSetGraphMemAttribute(self._dev_id, attr_enum, value))
        return 0

    @property
    def reserved_mem_current(self):
        """Current amount of backing memory allocated."""
        cdef cydriver.cuuint64_t value
        self._getattribute(cydriver.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT, &value)
        return int(value)

    @property
    def reserved_mem_high(self):
        """
        High watermark of backing memory allocated. It can be set to zero to
        reset it to the current usage.
        """
        cdef cydriver.cuuint64_t value
        self._getattribute(cydriver.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH, &value)
        return int(value)

    @reserved_mem_high.setter
    def reserved_mem_high(self, value: int):
        if value != 0:
            raise AttributeError(f"Attribute 'reserved_mem_high' may only be set to zero (got {value}).")
        cdef cydriver.cuuint64_t zero = 0
        self._setattribute(cydriver.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH, &zero)

    @property
    def used_mem_current(self):
        """Current amount of memory in use."""
        cdef cydriver.cuuint64_t value
        self._getattribute(cydriver.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT, &value)
        return int(value)

    @property
    def used_mem_high(self):
        """
        High watermark of memory in use. It can be set to zero to reset it to
        the current usage.
        """
        cdef cydriver.cuuint64_t value
        self._getattribute(cydriver.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_USED_MEM_HIGH, &value)
        return int(value)

    @used_mem_high.setter
    def used_mem_high(self, value: int):
        if value != 0:
            raise AttributeError(f"Attribute 'used_mem_high' may only be set to zero (got {value}).")
        cdef cydriver.cuuint64_t zero = 0
        self._setattribute(cydriver.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_USED_MEM_HIGH, &zero)


cdef class cyGraphMemoryResource(MemoryResource):
    def __cinit__(self, int device_id):
        self._dev_id = device_id

    def allocate(self, size_t size, stream: Stream | GraphBuilder | None = None) -> Buffer:
        """
        Allocate a buffer of the requested size. See documentation for :obj:`~_memory.MemoryResource`.
        """
        stream = Stream_accept(stream) if stream is not None else default_stream()
        return GMR_allocate(self, size, <Stream> stream)

    def deallocate(self, ptr: DevicePointerT, size_t size, stream: Stream | GraphBuilder | None = None):
        """
        Deallocate a buffer of the requested size. See documentation for :obj:`~_memory.MemoryResource`.
        """
        stream = Stream_accept(stream) if stream is not None else default_stream()
        return GMR_deallocate(ptr, size, <Stream> stream)

    def close(self):
        """No operation (provided for compatibility)."""
        pass

    def trim(self):
        """Free unused memory that was cached on the specified device for use with graphs back to the OS."""
        with nogil:
             HANDLE_RETURN(cydriver.cuDeviceGraphMemTrim(self._dev_id))

    @property
    def attributes(self) -> GraphMemoryResourceAttributes:
        """Asynchronous allocation attributes related to graphs."""
        return GraphMemoryResourceAttributes._init(self._dev_id)

    @property
    def device_id(self) -> int:
        """The associated device ordinal."""
        return self._dev_id

    @property
    def is_device_accessible(self) -> bool:
        """Return True. This memory resource provides device-accessible buffers."""
        return True

    @property
    def is_host_accessible(self) -> bool:
        """Return False. This memory resource does not provide host-accessible buffers."""
        return False


class GraphMemoryResource(cyGraphMemoryResource):
    """
    A memory resource for memory related to graphs.

    The only supported operations are allocation, deallocation, and a limited
    set of status queries.

    This memory resource should be used when building graphs.  Using this when
    graphs capture is not enabled will result in a runtime error.

    Conversely, allocating memory from a `DeviceMemoryResource` when graph
    capturing is enabled results in a runtime error.

    Parameters
    ----------
    device_id: int | Device
        Device or Device ordinal for which a graph memory resource is obtained.
    """

    def __new__(cls, device_id: int | Device):
        cdef int c_device_id = getattr(device_id, 'device_id', device_id)
        return cls._create(c_device_id)

    @classmethod
    @cache
    def _create(cls, int device_id):
        return cyGraphMemoryResource.__new__(cls, device_id)


# Raise an exception if the given stream is capturing.
# A result of CU_STREAM_CAPTURE_STATUS_INVALIDATED is considered an error.
cdef inline int check_capturing(cydriver.CUstream s) except?-1 nogil:
    cdef cydriver.CUstreamCaptureStatus capturing
    HANDLE_RETURN(cydriver.cuStreamIsCapturing(s, &capturing))
    if capturing != cydriver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
        raise RuntimeError("GraphMemoryResource cannot perform memory operations on "
                           "a non-capturing stream.")


cdef inline Buffer GMR_allocate(cyGraphMemoryResource self, size_t size, Stream stream):
    cdef cydriver.CUstream s = native(stream._h_stream)
    cdef cydriver.CUdeviceptr devptr
    with nogil:
        check_capturing(s)
        HANDLE_RETURN(cydriver.cuMemAllocAsync(&devptr, size, s))
    cdef Buffer buf = Buffer.__new__(Buffer)
    buf._ptr = <intptr_t>(devptr)
    buf._ptr_obj = None
    buf._size = size
    buf._memory_resource = self
    buf._alloc_stream = stream
    return buf


cdef inline void GMR_deallocate(intptr_t ptr, size_t size, Stream stream) noexcept:
    cdef cydriver.CUstream s = native(stream._h_stream)
    cdef cydriver.CUdeviceptr devptr = <cydriver.CUdeviceptr>ptr
    with nogil:
        HANDLE_RETURN(cydriver.cuMemFreeAsync(devptr, s))
