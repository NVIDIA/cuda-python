# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

cimport cpython

from libc.stddef cimport size_t
from cuda.bindings cimport cydriver
from cuda.core._memory._buffer cimport Buffer, Buffer_check_open, Buffer_from_deviceptr_handle
from cuda.core._memory._memory_pool cimport _MemPool, MP_check_open
from cuda.core._stream cimport Stream, Stream_accept
from cuda.core._rt cimport (
    DevicePtrHandle,
    create_fd_handle,
    create_mempool_handle_ipc,
    deviceptr_import_ipc,
    get_last_error,
    as_cu,
    as_py,
)

from cuda.core._utils.cuda_utils cimport HANDLE_RETURN
from cuda.core._utils.cuda_utils import check_multiprocessing_start_method

import multiprocessing
import operator
import os
import platform
import uuid
import weakref
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from cuda.core.typing import VirtualMemoryHandleType

__all__ = []


cdef object registry = weakref.WeakValueDictionary()


cdef cydriver.CUmemAllocationHandleType IPC_HANDLE_TYPE =                       \
    cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR \
    if platform.system() == "Linux" else                                        \
    cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE

cdef is_supported():
    return IPC_HANDLE_TYPE != cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE


cdef class IPCDataForBuffer:
    """Data members related to sharing memory buffers via IPC."""
    def __cinit__(self, IPCBufferDescriptor ipc_descriptor, bint is_mapped) -> None:
        self._ipc_descriptor = ipc_descriptor
        self._is_mapped = is_mapped

    @property
    def ipc_descriptor(self) -> IPCBufferDescriptor:
        return self._ipc_descriptor

    @property
    def is_mapped(self) -> bool:
        return self._is_mapped


cdef class IPCDataForMR:
    """Data members related to sharing memory resources via IPC."""
    def __cinit__(self, IPCAllocationHandle alloc_handle, bint is_mapped) -> None:
        self._alloc_handle = alloc_handle
        self._is_mapped = is_mapped

    @property
    def alloc_handle(self) -> IPCAllocationHandle:
        return self._alloc_handle

    @property
    def is_mapped(self) -> bool:
        return self._is_mapped

    @property
    def uuid(self) -> uuid.UUID | None:
        return getattr(self._alloc_handle, 'uuid', None)


cdef class IPCBufferDescriptor:
    """Serializable object describing a buffer that can be shared between processes.

    Note
    ----
    The payload and ``size`` fields are controlled by the exporting peer.
    Receivers must treat them as untrusted and import only through
    :meth:`Buffer.from_ipc_descriptor`.
    """

    def __init__(self, *arg, **kwargs) -> None:
        raise RuntimeError("IPCBufferDescriptor objects cannot be instantiated directly. Please use MemoryResource APIs.")

    @staticmethod
    def _init(reserved: bytes, size: int) -> IPCBufferDescriptor:
        cdef IPCBufferDescriptor self = IPCBufferDescriptor.__new__(IPCBufferDescriptor)
        self._payload = reserved
        self._size = size
        return self

    def __reduce__(self) -> tuple[object, ...]:
        return IPCBufferDescriptor._init, (self._payload, self._size)

    @property
    def size(self) -> int:
        return self._size

    cdef const void* payload_ptr(self) noexcept:
        """Return the payload as a const void* for C API calls."""
        return <const void*><const char*>(self._payload)


cdef inline int IPCAllocationHandle_check_open(IPCAllocationHandle self) except -1:
    if self._h_fd.get() == NULL:
        raise RuntimeError("IPCAllocationHandle has been closed")
    return 0


cdef class IPCAllocationHandle:
    """Shareable OS handle to an IPC-enabled memory pool or to a physical allocation
    of a :class:`VirtualMemoryResource` buffer."""

    def __init__(self, *arg, **kwargs) -> None:
        raise RuntimeError("IPCAllocationHandle objects cannot be instantiated directly. Please use MemoryResource APIs.")

    @classmethod
    def _init(cls, handle: int, uuid: uuid.UUID | None) -> IPCAllocationHandle:  # no-cython-lint
        cdef IPCAllocationHandle self = IPCAllocationHandle.__new__(cls)
        if handle < 0:
            raise ValueError(f"Invalid allocation handle (fd) {handle}: must be non-negative")
        self._h_fd = create_fd_handle(handle)
        self._uuid = uuid
        return self

    cpdef close(self):
        """Close the handle."""
        self._h_fd.reset()

    @property
    def is_closed(self) -> bool:
        """Whether this allocation handle has been closed."""
        return self._h_fd.get() == NULL

    def __int__(self) -> int:
        if self._h_fd.get() == NULL:
            raise ValueError(
                f"Cannot convert IPCAllocationHandle to int: the handle (id={id(self)}) is closed."
            )
        return as_py(self._h_fd)

    @property
    def handle(self) -> int:
        return as_py(self._h_fd)

    @property
    def uuid(self) -> uuid.UUID:
        return self._uuid


def _reduce_allocation_handle(alloc_handle: IPCAllocationHandle) -> tuple[object, ...]:
    IPCAllocationHandle_check_open(alloc_handle)
    check_multiprocessing_start_method()
    df = multiprocessing.reduction.DupFd(alloc_handle.handle)
    return _reconstruct_allocation_handle, (type(alloc_handle), df, alloc_handle.uuid)


def _reconstruct_allocation_handle(cls: type, df: object, uuid: uuid.UUID | None) -> IPCAllocationHandle:  # no-cython-lint
    return cls._init(df.detach(), uuid)


multiprocessing.reduction.register(IPCAllocationHandle, _reduce_allocation_handle)


cdef class VirtualMemoryIPCBufferDescriptor(IPCBufferDescriptor):
    """Serializable object describing a :class:`VirtualMemoryBuffer` that can be shared between processes.

    The descriptor holds one exported handle per physical allocation that
    backs the buffer, in address order (``handles``), with each allocation's
    size (``sizes``) and the buffer's size (``size``, which may be smaller
    than their sum). With POSIX file descriptors the handles are
    :class:`IPCAllocationHandle` objects: the descriptor owns them and closes
    them when it is released.

    A descriptor pins the physical memory while it exists, in every process
    that holds one, and nothing else does: the exporting buffer keeps no
    descriptor, and an imported buffer does not keep the one it came from. A
    descriptor parked in a ``multiprocessing`` ``Queue`` or sent to a ``Pool``
    pins the memory in the sender until the receiver has unpickled it; a
    buffer passed as a ``Process`` argument pins it until the ``Process``
    object is released, because the child receives the file descriptors when
    it is spawned.

    Two transports exist. ``multiprocessing`` (a ``Queue``, a ``Pipe``,
    ``Process`` arguments, or a ``Pool``) pickles the descriptor, or a buffer,
    and duplicates the file descriptors into the receiving process; plain
    ``pickle`` cannot carry them and raises. A process with its own file
    descriptor passing sends ``fds``, ``sizes``, ``handle_type``, and ``size``
    itself, and the receiver rebuilds the descriptor with :meth:`from_fds`.

    Note
    ----
    The sizes are controlled by the exporting peer. Receivers must treat them
    as untrusted and import only through :meth:`Buffer.from_ipc_descriptor`,
    which checks what it can and fails instead of mapping a size the driver
    rejects.
    """

    @staticmethod
    def _from_exports(
        handle_type: int, chunk_sizes: tuple[int, ...], handles: tuple, size: int
    ) -> VirtualMemoryIPCBufferDescriptor:
        cdef VirtualMemoryIPCBufferDescriptor self = VirtualMemoryIPCBufferDescriptor.__new__(
            VirtualMemoryIPCBufferDescriptor)
        self._payload = b""
        self._size = size
        self._handle_type = handle_type
        self._chunk_sizes = tuple(chunk_sizes)
        self._handles = tuple(handles)
        return self

    @classmethod
    def from_fds(
        cls,
        fds: Iterable[int | IPCAllocationHandle],
        sizes: Iterable[int],
        *,
        handle_type: VirtualMemoryHandleType | str = "posix_fd",
        size: int | None = None,
    ) -> VirtualMemoryIPCBufferDescriptor:
        """Build a descriptor from file descriptors received outside ``multiprocessing``.

        The exporter sends ``fds``, ``sizes``, ``handle_type``, and ``size``
        of its descriptor through a transport of its own (a Unix socket with
        ``SCM_RIGHTS``, for example); the receiver rebuilds the descriptor
        here and imports it with :meth:`Buffer.from_ipc_descriptor`.

        Parameters
        ----------
        fds : Iterable[int | IPCAllocationHandle]
            One handle per physical allocation, in address order, as the
            exporter's ``fds`` listed them. An ``int`` is duplicated: the
            caller keeps ownership of the original and closes it. An
            :class:`IPCAllocationHandle` is shared and closes with its last
            holder.
        sizes : Iterable[int]
            The exporter's ``sizes``, one per handle.
        handle_type : VirtualMemoryHandleType | str, optional
            The exporter's ``handle_type``; ``"posix_fd"`` by default.
        size : int, optional
            The exporter's ``size``, at most the sum of ``sizes``; the sum by
            default.

        Raises
        ------
        ValueError
            If the counts differ, a size is not positive, ``size`` exceeds the
            sum of ``sizes``, or ``handle_type`` cannot be shared.
        RuntimeError
            If an :class:`IPCAllocationHandle` is closed.
        OSError
            If an ``int`` is not an open file descriptor.
        """
        # Lazy import: this module is loaded while _virtual_memory_resource imports it.
        from cuda.core._memory._virtual_memory_resource import VirtualMemoryResourceOptions
        from cuda.core.typing import VirtualMemoryHandleType

        if handle_type is None:
            raise ValueError("handle_type=None cannot be shared; the exporter's handle_type is required")
        driver_type = int(VirtualMemoryResourceOptions._handle_type_to_driver(VirtualMemoryHandleType(handle_type)))
        cdef tuple sizes_t = tuple(operator.index(n) for n in sizes)
        cdef list given = list(fds)
        if len(given) != len(sizes_t):
            raise ValueError(f"{len(given)} handles for {len(sizes_t)} sizes; one size per handle is required")
        for n in sizes_t:
            if n <= 0:
                raise ValueError(f"allocation sizes must be positive, got {n}")
        total = sum(sizes_t)
        size = total if size is None else operator.index(size)
        if size < 0 or size > total:
            raise ValueError(f"size {size} is outside 0..{total}, the sum of the allocation sizes")
        cdef list handles = []
        cdef int dup
        for item in given:
            if isinstance(item, IPCAllocationHandle):
                IPCAllocationHandle_check_open(<IPCAllocationHandle>item)
                handles.append(item)
                continue
            dup = os.dup(operator.index(item))
            try:
                handle = IPCAllocationHandle._init(dup, None)
            except:  # noqa: E722  rollback-then-raise: the duplicate is not owned yet
                os.close(dup)
                raise
            # The handle owns the duplicate now; if append fails, the handle closes it once.
            handles.append(handle)
        return VirtualMemoryIPCBufferDescriptor._from_exports(driver_type, sizes_t, tuple(handles), size)

    @property
    def handle_type(self) -> VirtualMemoryHandleType:
        """The handle type the exporter used; the importing resource must be configured with the same."""
        from cuda.core._memory._virtual_memory_resource import VirtualMemoryResourceOptions

        for spec, value in VirtualMemoryResourceOptions._handle_types.items():
            if spec is not None and int(value) == self._handle_type:
                return spec
        raise ValueError(f"the descriptor carries an unknown handle type ({self._handle_type})")

    @property
    def sizes(self) -> tuple[int, ...]:
        """Size of each physical allocation, in address order; the sum is at least ``size``."""
        return self._chunk_sizes

    @property
    def handles(self) -> tuple[IPCAllocationHandle, ...]:
        """The exported handles, one per allocation, in address order; the descriptor owns them."""
        return self._handles

    @property
    def fds(self) -> tuple[int, ...]:
        """The exported file descriptors as integers, for a transport of your own.

        They belong to this descriptor: send or duplicate them while it is
        alive. Each one is released when the descriptor is.
        """
        return tuple(int(handle) for handle in self._handles)

    def __reduce__(self) -> tuple[object, ...]:
        # multiprocessing duplicates the IPCAllocationHandle fds into the receiver; plain pickle raises.
        return VirtualMemoryIPCBufferDescriptor._from_exports, (
            self._handle_type, self._chunk_sizes, self._handles, self._size)


# Buffer IPC Implementation
# -------------------------
cdef IPCBufferDescriptor Buffer_get_ipc_descriptor(Buffer self):
    Buffer_check_open(self)
    if not self.memory_resource.is_ipc_enabled:
        raise RuntimeError("Memory resource is not IPC-enabled")
    if not isinstance(self.memory_resource, _MemPool):
        # VirtualMemoryBuffer overrides ipc_descriptor; a plain Buffer wrapped
        # with Buffer.from_handle(mr=<VirtualMemoryResource>) lands here.
        raise TypeError(
            f"a Buffer that {type(self.memory_resource).__name__} did not allocate cannot be shared")
    cdef cydriver.CUmemPoolPtrExportData data
    with nogil:
        HANDLE_RETURN(
            cydriver.cuMemPoolExportPointer(&data, as_cu(self._h_ptr))
        )
    cdef bytes data_b = cpython.PyBytes_FromStringAndSize(
        <char*>(data.reserved), sizeof(data.reserved)
    )
    return IPCBufferDescriptor._init(data_b, self.size)

cdef Buffer Buffer_from_ipc_descriptor(
    cls, object mr, IPCBufferDescriptor ipc_descriptor, stream
):
    """Import a buffer that was exported from another process."""
    if isinstance(ipc_descriptor, VirtualMemoryIPCBufferDescriptor):
        # Imported here rather than cimported: _virtual_memory_resource
        # cimports this module.
        from cuda.core._memory._virtual_memory_resource import VirtualMemoryResource
        if not isinstance(mr, VirtualMemoryResource):
            raise TypeError(
                "this descriptor was exported from a VirtualMemoryResource buffer and must be "
                f"imported with a VirtualMemoryResource, not {type(mr).__name__}")
        return mr._import_ipc_buffer(ipc_descriptor, stream)
    if not isinstance(mr, _MemPool):
        raise TypeError(
            "this descriptor was exported from a memory pool buffer and must be imported with a "
            f"DeviceMemoryResource or PinnedMemoryResource, not {type(mr).__name__}")
    cdef _MemPool pool = <_MemPool>mr
    MP_check_open(pool)
    if not mr.is_ipc_enabled:
        raise RuntimeError("Memory resource is not IPC-enabled")
    cdef size_t payload_size = len(ipc_descriptor._payload)
    cdef size_t expected_size = sizeof(cydriver.CUmemPoolPtrExportData)
    if payload_size < expected_size:
        raise ValueError(
            f"IPC buffer descriptor payload is {payload_size} bytes; "
            f"expected at least {expected_size}"
        )
    cdef Stream s = Stream_accept(stream)
    cdef DevicePtrHandle h_ptr = deviceptr_import_ipc(
        pool._h_pool,
        ipc_descriptor.payload_ptr(),
        s._h_stream
    )
    if not h_ptr:
        HANDLE_RETURN(get_last_error())
    cdef size_t mapped_size = 0
    cdef size_t claimed_size = ipc_descriptor.size
    with nogil:
        HANDLE_RETURN(cydriver.cuPointerGetAttribute(
            &mapped_size,
            cydriver.CU_POINTER_ATTRIBUTE_RANGE_SIZE,
            as_cu(h_ptr)))
    if claimed_size > mapped_size:
        h_ptr.reset()
        raise ValueError(
            f"IPC buffer descriptor size ({claimed_size}) exceeds "
            f"mapped allocation extent ({mapped_size} bytes)"
        )
    return Buffer_from_deviceptr_handle(h_ptr, claimed_size, pool, ipc_descriptor)


# _MemPool IPC Implementation
# ---------------------------

cdef _MemPool MP_from_allocation_handle(cls, alloc_handle):
    if isinstance(alloc_handle, IPCAllocationHandle):
        IPCAllocationHandle_check_open(<IPCAllocationHandle>alloc_handle)

    # Quick exit for registry hits.
    uuid = getattr(alloc_handle, 'uuid', None)  # no-cython-lint
    mr = registry.get(uuid)
    if mr is not None:
        if not isinstance(mr, cls):
            raise TypeError(
                f"Registry contains a {type(mr).__name__} for uuid "
                f"{uuid}, but {cls.__name__} was requested")
        MP_check_open(<_MemPool>mr)
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

    # Construct a new mempool.
    cdef _MemPool self = <_MemPool>(cls.__new__(cls))
    self._mempool_owned = True
    cdef int ipc_fd = int(alloc_handle)
    self._h_pool = create_mempool_handle_ipc(ipc_fd, IPC_HANDLE_TYPE)
    if not self._h_pool:
        HANDLE_RETURN(get_last_error())
        raise RuntimeError(
            f"Failed to import {cls.__name__} from an allocation handle: "
            "cuda-core returned an empty memory pool handle without recording a CUDA error. "
            "This is an internal cuda-core error; please report it with your CUDA driver, "
            "CUDA Toolkit, and cuda-python versions."
        )
    self._ipc_data = IPCDataForMR(alloc_handle, True)

    # Register it.
    if uuid is not None:
        registered = self.register(uuid)
        assert registered is self

    return self


cdef _MemPool MP_from_registry(uuid):
    cdef _MemPool mr
    try:
        mr = registry[uuid]
        MP_check_open(mr)
        return mr
    except KeyError:
        raise RuntimeError(f"Memory resource {uuid} was not found") from None


cdef _MemPool MP_register(_MemPool self, uuid):
    MP_check_open(self)
    existing = registry.get(uuid)
    if existing is not None:
        MP_check_open(<_MemPool>existing)
        return existing
    if not self.is_ipc_enabled:
        raise RuntimeError("Memory resource is not IPC-enabled")
    assert self.uuid is None or self.uuid == uuid
    registry[uuid] = self
    self._ipc_data._alloc_handle._uuid = uuid
    return self


cdef IPCAllocationHandle MP_export_mempool(_MemPool self):
    # Note: This is Linux only (int for file descriptor)
    MP_check_open(self)
    cdef int fd
    with nogil:
        HANDLE_RETURN(cydriver.cuMemPoolExportToShareableHandle(
            &fd, as_cu(self._h_pool), IPC_HANDLE_TYPE, 0)
        )
    try:
        return IPCAllocationHandle._init(fd, uuid.uuid4())
    except:
        os.close(fd)
        raise
