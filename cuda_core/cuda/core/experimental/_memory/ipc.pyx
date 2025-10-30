# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport intptr_t

from cuda.bindings cimport cydriver
from cuda.core.experimental._utils.cuda_utils cimport (
    HANDLE_RETURN,
)

from typing import Iterable, Literal, Optional, TypeVar, Union
import multiprocessing
import os
import platform
import uuid
import weakref


cdef object registry = weakref.WeakValueDictionary()

cdef cydriver.CUmemAllocationHandleType IPC_HANDLE_TYPE = cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR \
    if platform.system() == "Linux" else cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE

cdef class IPCBufferDescriptor:
    """Serializable object describing a buffer that can be shared between processes."""

    def __init__(self, *arg, **kwargs):
        raise RuntimeError("IPCBufferDescriptor objects cannot be instantiated directly. Please use MemoryResource APIs.")

    @classmethod
    def _init(cls, reserved: bytes, size: int):
        cdef IPCBufferDescriptor self = IPCBufferDescriptor.__new__(cls)
        self._reserved = reserved
        self._size = size
        return self

    def __reduce__(self):
        return self._init, (self._reserved, self._size)

    @property
    def size(self):
        return self._size


cdef class IPCAllocationHandle:
    """Shareable handle to an IPC-enabled device memory pool."""

    def __init__(self, *arg, **kwargs):
        raise RuntimeError("IPCAllocationHandle objects cannot be instantiated directly. Please use MemoryResource APIs.")

    @classmethod
    def _init(cls, handle: int, uuid: uuid.UUID):
        cdef IPCAllocationHandle self = IPCAllocationHandle.__new__(cls)
        assert handle >= 0
        self._handle = handle
        self._uuid = uuid
        return self

    cpdef close(self):
        """Close the handle."""
        if self._handle >= 0:
            try:
                os.close(self._handle)
            finally:
                self._handle = -1
                self._uuid = None

    def __dealloc__(self):
        self.close()

    def __int__(self) -> int:
        if self._handle < 0:
            raise ValueError(
                f"Cannot convert IPCAllocationHandle to int: the handle (id={id(self)}) is closed."
            )
        return self._handle

    @property
    def handle(self) -> int:
        return self._handle

    @property
    def uuid(self) -> uuid.UUID:
        return self._uuid


def _reduce_allocation_handle(alloc_handle):
    df = multiprocessing.reduction.DupFd(alloc_handle.handle)
    return _reconstruct_allocation_handle, (type(alloc_handle), df, alloc_handle.uuid)

def _reconstruct_allocation_handle(cls, df, uuid):
    return cls._init(df.detach(), uuid)


multiprocessing.reduction.register(IPCAllocationHandle, _reduce_allocation_handle)


cpdef IPCAllocationHandle DMR_get_allocation_handle(DeviceMemoryResource self):
    # Note: This is Linux only (int for file descriptor)
    cdef int alloc_handle

    if self._alloc_handle is None:
        if not self.is_ipc_enabled:
            raise RuntimeError("Memory resource is not IPC-enabled")
        if self._is_mapped:
            raise RuntimeError("Imported memory resource cannot be exported")

        with nogil:
            HANDLE_RETURN(cydriver.cuMemPoolExportToShareableHandle(
                &alloc_handle, self._mempool_handle, IPC_HANDLE_TYPE, 0)
            )
        try:
            assert self._uuid is None
            self._uuid = uuid.uuid4()
            self._alloc_handle = IPCAllocationHandle._init(alloc_handle, self._uuid)
        except:
            os.close(alloc_handle)
            raise
    return self._alloc_handle


cpdef DeviceMemoryResource DMR_from_allocation_handle(cls, device_id: int | Device, alloc_handle: int | IPCAllocationHandle):
    # Quick exit for registry hits.
    uuid = getattr(alloc_handle, 'uuid', None)
    mr = registry.get(uuid)
    if mr is not None:
        return mr

    device_id = getattr(device_id, 'device_id', device_id)

    cdef DeviceMemoryResource self = DeviceMemoryResource.__new__(cls)
    self._dev_id = device_id
    self._ipc_handle_type = IPC_HANDLE_TYPE
    self._mempool_owned = True
    self._is_mapped = True
    #self._alloc_handle = None  # only used for non-imported

    cdef int handle = int(alloc_handle)
    with nogil:
        HANDLE_RETURN(cydriver.cuMemPoolImportFromShareableHandle(
            &(self._mempool_handle), <void*><intptr_t>(handle), IPC_HANDLE_TYPE, 0)
        )
    if uuid is not None:
        registered = self.register(uuid)
        assert registered is self
    return self


cpdef DeviceMemoryResource DMR_register(DeviceMemoryResource self, uuid: uuid.UUID):
    existing = registry.get(uuid)
    if existing is not None:
        return existing
    assert self._uuid is None or self._uuid == uuid
    registry[uuid] = self
    self._uuid = uuid
    return self

cpdef DeviceMemoryResource DMR_from_registry(uuid: uuid.UUID):
    try:
        return registry[uuid]
    except KeyError:
        raise RuntimeError(f"Memory resource {uuid} was not found") from None
