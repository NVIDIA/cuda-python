# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from libc.stdint cimport uintptr_t, intptr_t

from cuda.bindings cimport cydriver
from cuda.core.experimental._memory._buffer cimport Buffer, MemoryResource
from cuda.core.experimental._stream cimport Stream
from cuda.core.experimental._utils.cuda_utils cimport HANDLE_RETURN

from functools import cache
from typing import TYPE_CHECKING

from cuda.core.experimental._utils.cuda_utils import driver

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

    @GMRA_mem_attribute(int)
    def reserved_mem_current(self):
        """Current amount of backing memory allocated."""

    @GMRA_mem_attribute(int, settable=True)
    def reserved_mem_high(self):
        """High watermark of backing memory allocated."""

    @GMRA_mem_attribute(int)
    def used_mem_current(self):
        """Current amount of memory in use."""

    @GMRA_mem_attribute(int, settable=True)
    def used_mem_high(self):
        """High watermark of memory in use."""


cdef GMRA_mem_attribute(property_type: type, settable : bool = False):
    _settable = settable

    def decorator(stub):
        attr_enum = getattr(
            driver.CUgraphMem_attribute, f"CU_GRAPH_MEM_ATTR_{stub.__name__.upper()}"
        )

        def fget(GraphMemoryResourceAttributes self) -> property_type:
            value = GMRA_getattribute(self._dev_id, <cydriver.CUgraphMem_attribute><uintptr_t> attr_enum)
            return property_type(value)

        if _settable:
            def fset(GraphMemoryResourceAttributes self, value: int):
                GMRA_setattribute(self._dev_id, <cydriver.CUgraphMem_attribute><uintptr_t> attr_enum, value)
        else:
            fset = None

        return property(fget=fget, fset=fset, doc=stub.__doc__)
    return decorator


cdef int GMRA_getattribute(int device_id, cydriver.CUgraphMem_attribute attr_enum):
    cdef int value
    with nogil:
        HANDLE_RETURN(cydriver.cuDeviceGetGraphMemAttribute(device_id, attr_enum, <void *> &value))
    return value


cdef int GMRA_setattribute(int device_id, cydriver.CUgraphMem_attribute attr_enum, int value):
    with nogil:
        HANDLE_RETURN(cydriver.cuDeviceSetGraphMemAttribute(device_id, attr_enum, <void *> &value))


cdef class cyGraphMemoryResource(MemoryResource):
    def __cinit__(self, int device_id):
        self._dev_id = device_id

    def allocate(self, size_t size, Stream stream = None) -> Buffer:
        return GMR_allocate(self, size, stream)

    def deallocate(self, ptr: DevicePointerT, size_t size, Stream stream = None):
        return GMR_deallocate(ptr, size, stream)

    def close(self):
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
    @cache
    def __new__(cls, device_id: int | Device):
        cdef int c_device_id = getattr(device_id, 'device_id', device_id)
        return cyGraphMemoryResource.__new__(cls, c_device_id)


cdef Buffer GMR_allocate(cyGraphMemoryResource self, size_t size, Stream stream):
    cdef cydriver.CUstream s = stream._handle
    cdef cydriver.CUdeviceptr devptr
    with nogil:
        HANDLE_RETURN(cydriver.cuMemAllocAsync(&devptr, size, s))
    cdef Buffer buf = Buffer.__new__(Buffer)
    buf._ptr = <intptr_t>(devptr)
    buf._ptr_obj = None
    buf._size = size
    buf._memory_resource = self
    buf._alloc_stream = stream
    return buf


cdef void GMR_deallocate(intptr_t ptr, size_t size, Stream stream) noexcept:
    cdef cydriver.CUstream s = stream._handle
    cdef cydriver.CUdeviceptr devptr = <cydriver.CUdeviceptr>ptr
    with nogil:
        HANDLE_RETURN(cydriver.cuMemFreeAsync(devptr, s))

