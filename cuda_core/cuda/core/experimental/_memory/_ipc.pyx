# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

cimport cpython
from libc.stdint cimport uintptr_t
from libc.string cimport memcpy

from cuda.bindings cimport cydriver
from cuda.core.experimental._memory._buffer cimport Buffer
from cuda.core.experimental._stream cimport default_stream
from cuda.core.experimental._utils.cuda_utils cimport HANDLE_RETURN
from cuda.core.experimental._utils.cuda_utils import check_multiprocessing_start_method

import multiprocessing
import os
import platform
import uuid
import weakref

__all__ = ['IPCBufferDescriptor', 'IPCAllocationHandle']


cdef object registry = weakref.WeakValueDictionary()


cdef cydriver.CUmemAllocationHandleType IPC_HANDLE_TYPE =                       \
    cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR \
    if platform.system() == "Linux" else                                        \
    cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE

cdef is_supported():
    return IPC_HANDLE_TYPE != cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE


cdef class IPCDataForBuffer:
    """Data members related to sharing memory buffers via IPC."""
    def __cinit__(self, IPCBufferDescriptor ipc_descriptor, bint is_mapped):
        self._ipc_descriptor = ipc_descriptor
        self._is_mapped = is_mapped

    @property
    def ipc_descriptor(self):
        return self._ipc_descriptor

    @property
    def is_mapped(self):
        return self._is_mapped


cdef class IPCDataForMR:
    """Data members related to sharing memory resources via IPC."""
    def __cinit__(self, IPCAllocationHandle alloc_handle, bint is_mapped):
        self._alloc_handle = alloc_handle
        self._is_mapped = is_mapped

    @property
    def alloc_handle(self):
        return self._alloc_handle

    @property
    def is_mapped(self):
        return self._is_mapped

    @property
    def uuid(self):
        return getattr(self._alloc_handle, 'uuid', None)


cdef class IPCBufferDescriptor:
    """Serializable object describing a buffer that can be shared between processes."""

    def __init__(self, *arg, **kwargs):
        raise RuntimeError("IPCBufferDescriptor objects cannot be instantiated directly. Please use MemoryResource APIs.")

    @classmethod
    def _init(cls, reserved: bytes, size: int):
        cdef IPCBufferDescriptor self = IPCBufferDescriptor.__new__(cls)
        self._payload = reserved
        self._size = size
        return self

    def __reduce__(self):
        return self._init, (self._payload, self._size)

    @property
    def size(self):
        return self._size


cdef class IPCAllocationHandle:
    """Shareable handle to an IPC-enabled device memory pool."""

    def __init__(self, *arg, **kwargs):
        raise RuntimeError("IPCAllocationHandle objects cannot be instantiated directly. Please use MemoryResource APIs.")

    @classmethod
    def _init(cls, handle: int, uuid):  # no-cython-lint
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
    check_multiprocessing_start_method()
    df = multiprocessing.reduction.DupFd(alloc_handle.handle)
    return _reconstruct_allocation_handle, (type(alloc_handle), df, alloc_handle.uuid)


def _reconstruct_allocation_handle(cls, df, uuid):  # no-cython-lint
    return cls._init(df.detach(), uuid)


multiprocessing.reduction.register(IPCAllocationHandle, _reduce_allocation_handle)


def _deep_reduce_device_memory_resource(mr):
    check_multiprocessing_start_method()
    from .._device import Device
    device = Device(mr.device_id)
    alloc_handle = mr.get_allocation_handle()
    return mr.from_allocation_handle, (device, alloc_handle)


multiprocessing.reduction.register(DeviceMemoryResource, _deep_reduce_device_memory_resource)


# Buffer IPC Implementation
# -------------------------
cdef IPCBufferDescriptor Buffer_get_ipc_descriptor(Buffer self):
    if not self.memory_resource.is_ipc_enabled:
        raise RuntimeError("Memory resource is not IPC-enabled")
    cdef cydriver.CUmemPoolPtrExportData data
    with nogil:
        HANDLE_RETURN(
            cydriver.cuMemPoolExportPointer(&data, <cydriver.CUdeviceptr>(self._ptr))
        )
    cdef bytes data_b = cpython.PyBytes_FromStringAndSize(
        <char*>(data.reserved), sizeof(data.reserved)
    )
    return IPCBufferDescriptor._init(data_b, self.size)

cdef Buffer Buffer_from_ipc_descriptor(
    cls, DeviceMemoryResource mr, IPCBufferDescriptor ipc_descriptor, stream
):
    """Import a buffer that was exported from another process."""
    if not mr.is_ipc_enabled:
        raise RuntimeError("Memory resource is not IPC-enabled")
    if stream is None:
        # Note: match this behavior to DeviceMemoryResource.allocate()
        stream = default_stream()
    cdef cydriver.CUmemPoolPtrExportData data
    memcpy(
        data.reserved,
        <const void*><const char*>(ipc_descriptor._payload),
        sizeof(data.reserved)
    )
    cdef cydriver.CUdeviceptr ptr
    with nogil:
        HANDLE_RETURN(cydriver.cuMemPoolImportPointer(&ptr, mr._handle, &data))
    return Buffer._init(<uintptr_t>ptr, ipc_descriptor.size, mr, stream, ipc_descriptor)


# DeviceMemoryResource IPC Implementation
# ---------------------------------------

cdef DeviceMemoryResource DMR_from_allocation_handle(cls, device_id, alloc_handle):
    # Quick exit for registry hits.
    uuid = getattr(alloc_handle, 'uuid', None)  # no-cython-lint
    mr = registry.get(uuid)
    if mr is not None:
        return mr

    # Ensure we have an allocation handle. Duplicate the file descriptor, if
    # necessary.
    if isinstance(alloc_handle, int):
        fd = os.dup(alloc_handle)
        try:
            alloc_handle = IPCAllocationHandle._init(fd, None)
        except:
            os.close(fd)
            raise

    # Construct a new DMR.
    cdef DeviceMemoryResource self = DeviceMemoryResource.__new__(cls)
    from .._device import Device
    self._dev_id = Device(device_id).device_id
    self._mempool_owned = True
    self._ipc_data = IPCDataForMR(alloc_handle, True)

    # Map the mempool into this process.
    cdef int handle = int(alloc_handle)
    with nogil:
        HANDLE_RETURN(cydriver.cuMemPoolImportFromShareableHandle(
            &(self._handle), <void*><uintptr_t>(handle), IPC_HANDLE_TYPE, 0)
        )

    # Register it.
    if uuid is not None:
        registered = self.register(uuid)
        assert registered is self

    return self


cdef DeviceMemoryResource DMR_from_registry(uuid):
    try:
        return registry[uuid]
    except KeyError:
        raise RuntimeError(f"Memory resource {uuid} was not found") from None


cdef DeviceMemoryResource DMR_register(DeviceMemoryResource self, uuid):
    existing = registry.get(uuid)
    if existing is not None:
        return existing
    assert self.uuid is None or self.uuid == uuid
    registry[uuid] = self
    self._ipc_data._alloc_handle._uuid = uuid
    return self


cdef IPCAllocationHandle DMR_export_mempool(DeviceMemoryResource self):
    # Note: This is Linux only (int for file descriptor)
    cdef int fd
    with nogil:
        HANDLE_RETURN(cydriver.cuMemPoolExportToShareableHandle(
            &fd, self._handle, IPC_HANDLE_TYPE, 0)
        )
    try:
        return IPCAllocationHandle._init(fd, uuid.uuid4())
    except:
        os.close(fd)
        raise
