# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

cimport cpython
from libc.limits cimport ULLONG_MAX
from libc.stdint cimport uintptr_t, intptr_t
from libc.string cimport memset, memcpy

from cuda.bindings cimport cydriver

from cuda.core.experimental._stream cimport Stream as cyStream
from cuda.core.experimental._utils.cuda_utils cimport (
    _check_driver_error as raise_if_driver_error,
    check_or_create_options,
    HANDLE_RETURN,
)

import abc
import array
import contextlib
import cython
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional, TYPE_CHECKING, TypeVar, Union
import multiprocessing
import os
import platform
import weakref

from cuda.core.experimental._dlpack import DLDeviceType, make_py_capsule
from cuda.core.experimental._stream import Stream, default_stream
from cuda.core.experimental._utils.cuda_utils import ( driver, Transaction, get_binding_version )

if platform.system() == "Linux":
    import socket

if TYPE_CHECKING:
    from ._device import Device
    import uuid


PyCapsule = TypeVar("PyCapsule")
"""Represent the capsule type."""

DevicePointerT = Union[driver.CUdeviceptr, int, None]
"""A type union of :obj:`~driver.CUdeviceptr`, `int` and `None` for hinting :attr:`Buffer.handle`."""


cdef class _cyBuffer:
    """
    Internal only. Responsible for offering fast C method access.
    """
    cdef:
        intptr_t _ptr
        size_t _size
        _cyMemoryResource _mr
        object _ptr_obj
        cyStream _alloc_stream


cdef class _cyMemoryResource:
    """
    Internal only. Responsible for offering fast C method access.
    """
    cdef Buffer _allocate(self, size_t size, cyStream stream):
        raise NotImplementedError

    cdef void _deallocate(self, intptr_t ptr, size_t size, cyStream stream) noexcept:
        raise NotImplementedError


class MemoryResourceAttributes(abc.ABC):

    __slots__ = ()

    @property
    @abc.abstractmethod
    def is_device_accessible(self) -> bool:
        """bool: True if buffers allocated by this resource can be accessed on the device."""
        ...

    @property
    @abc.abstractmethod
    def is_host_accessible(self) -> bool:
        """bool: True if buffers allocated by this resource can be accessed on the host."""
        ...

    @property
    @abc.abstractmethod
    def device_id(self) -> int:
        """int: The device ordinal for which this memory resource is responsible.

        Raises
        ------
        RuntimeError
            If the resource is not bound to a specific device.
        """
        ...


cdef class Buffer(_cyBuffer, MemoryResourceAttributes):
    """Represent a handle to allocated memory.

    This generic object provides a unified representation for how
    different memory resources are to give access to their memory
    allocations.

    Support for data interchange mechanisms are provided by DLPack.
    """
    def __cinit__(self):
        self._ptr = 0
        self._size = 0
        self._mr = None
        self._ptr_obj = None
        self._alloc_stream = None

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Buffer objects cannot be instantiated directly. Please use MemoryResource APIs.")

    @classmethod
    def _init(cls, ptr: DevicePointerT, size_t size, mr: MemoryResource | None = None, stream: Stream | None = None):
        cdef Buffer self = Buffer.__new__(cls)
        self._ptr = <intptr_t>(int(ptr))
        self._ptr_obj = ptr
        self._size = size
        self._mr = mr
        self._alloc_stream = <cyStream>(stream) if stream is not None else None
        return self

    def __dealloc__(self):
        self.close(self._alloc_stream)

    def __reduce__(self):
        # Must not serialize the parent's stream!
        return Buffer.from_ipc_descriptor, (self.memory_resource, self.get_ipc_descriptor())

    cpdef close(self, stream: Stream = None):
        """Deallocate this buffer asynchronously on the given stream.

        This buffer is released back to their memory resource
        asynchronously on the given stream.

        Parameters
        ----------
        stream : Stream, optional
            The stream object to use for asynchronous deallocation. If None,
            the behavior depends on the underlying memory resource.
        """
        cdef cyStream s
        if self._ptr and self._mr is not None:
            if stream is None:
                if self._alloc_stream is not None:
                    s = self._alloc_stream
                else:
                    # TODO: remove this branch when from_handle takes a stream
                    s = <cyStream>(default_stream())
            else:
                s = <cyStream>stream
            self._mr._deallocate(self._ptr, self._size, s)
            self._ptr = 0
            self._mr = None
            self._ptr_obj = None
            self._alloc_stream = None

    @property
    def handle(self) -> DevicePointerT:
        """Return the buffer handle object.

        .. caution::

            This handle is a Python object. To get the memory address of the underlying C
            handle, call ``int(Buffer.handle)``.
        """
        if self._ptr_obj is not None:
            return self._ptr_obj
        elif self._ptr:
            return self._ptr
        else:
            # contract: Buffer is closed
            return 0

    @property
    def size(self) -> int:
        """Return the memory size of this buffer."""
        return self._size

    @property
    def memory_resource(self) -> MemoryResource:
        """Return the memory resource associated with this buffer."""
        return self._mr

    @property
    def is_device_accessible(self) -> bool:
        """Return True if this buffer can be accessed by the GPU, otherwise False."""
        if self._mr is not None:
            return self._mr.is_device_accessible
        raise NotImplementedError("WIP: Currently this property only supports buffers with associated MemoryResource")

    @property
    def is_host_accessible(self) -> bool:
        """Return True if this buffer can be accessed by the CPU, otherwise False."""
        if self._mr is not None:
            return self._mr.is_host_accessible
        raise NotImplementedError("WIP: Currently this property only supports buffers with associated MemoryResource")

    @property
    def device_id(self) -> int:
        """Return the device ordinal of this buffer."""
        if self._mr is not None:
            return self._mr.device_id
        raise NotImplementedError("WIP: Currently this property only supports buffers with associated MemoryResource")

    def get_ipc_descriptor(self) -> IPCBufferDescriptor:
        """Export a buffer allocated for sharing between processes."""
        if not self._mr.is_ipc_enabled:
            raise RuntimeError("Memory resource is not IPC-enabled")
        cdef cydriver.CUmemPoolPtrExportData data
        with nogil:
            HANDLE_RETURN(cydriver.cuMemPoolExportPointer(&data, <cydriver.CUdeviceptr>(self._ptr)))
        cdef bytes data_b = cpython.PyBytes_FromStringAndSize(<char*>(data.reserved), sizeof(data.reserved))
        return IPCBufferDescriptor._init(data_b, self.size)

    @classmethod
    def from_ipc_descriptor(cls, mr: DeviceMemoryResource, ipc_buffer: IPCBufferDescriptor, stream: Stream = None) -> Buffer:
        """Import a buffer that was exported from another process."""
        if not mr.is_ipc_enabled:
            raise RuntimeError("Memory resource is not IPC-enabled")
        if stream is None:
            # Note: match this behavior to DeviceMemoryResource.allocate()
            stream = default_stream()
        cdef cydriver.CUmemPoolPtrExportData share_data
        memcpy(share_data.reserved, <const void*><const char*>(ipc_buffer._reserved), sizeof(share_data.reserved))
        cdef cydriver.CUdeviceptr ptr
        with nogil:
            HANDLE_RETURN(cydriver.cuMemPoolImportPointer(&ptr, mr._mempool_handle, &share_data))
        return Buffer._init(<intptr_t>ptr, ipc_buffer.size, mr, stream)

    def copy_to(self, dst: Buffer = None, *, stream: Stream) -> Buffer:
        """Copy from this buffer to the dst buffer asynchronously on the given stream.

        Copies the data from this buffer to the provided dst buffer.
        If the dst buffer is not provided, then a new buffer is first
        allocated using the associated memory resource before the copy.

        Parameters
        ----------
        dst : :obj:`~_memory.Buffer`
            Source buffer to copy data from
        stream : Stream
            Keyword argument specifying the stream for the
            asynchronous copy

        """
        if stream is None:
            raise ValueError("stream must be provided")

        cdef size_t src_size = self._size

        if dst is None:
            if self._mr is None:
                raise ValueError("a destination buffer must be provided (this buffer does not have a memory_resource)")
            dst = self._mr.allocate(src_size, stream)

        cdef size_t dst_size = dst._size
        if dst_size != src_size:
            raise ValueError(
                f"buffer sizes mismatch between src and dst (sizes are: src={src_size}, dst={dst_size})"
            )
        err, = driver.cuMemcpyAsync(dst._ptr, self._ptr, src_size, stream.handle)
        raise_if_driver_error(err)
        return dst

    def copy_from(self, src: Buffer, *, stream: Stream):
        """Copy from the src buffer to this buffer asynchronously on the given stream.

        Parameters
        ----------
        src : :obj:`~_memory.Buffer`
            Source buffer to copy data from
        stream : Stream
            Keyword argument specifying the stream for the
            asynchronous copy

        """
        if stream is None:
            raise ValueError("stream must be provided")

        cdef size_t dst_size = self._size
        cdef size_t src_size = src._size

        if src_size != dst_size:
            raise ValueError(
                f"buffer sizes mismatch between src and dst (sizes are: src={src_size}, dst={dst_size})"
            )
        err, = driver.cuMemcpyAsync(self._ptr, src._ptr, dst_size, stream.handle)
        raise_if_driver_error(err)

    def __dlpack__(
        self,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> PyCapsule:
        # Note: we ignore the stream argument entirely (as if it is -1).
        # It is the user's responsibility to maintain stream order.
        if dl_device is not None:
            raise BufferError("Sorry, not supported: dl_device other than None")
        if copy is True:
            raise BufferError("Sorry, not supported: copy=True")
        if max_version is None:
            versioned = False
        else:
            if not isinstance(max_version, tuple) or len(max_version) != 2:
                raise BufferError(f"Expected max_version tuple[int, int], got {max_version}")
            versioned = max_version >= (1, 0)
        capsule = make_py_capsule(self, versioned)
        return capsule

    def __dlpack_device__(self) -> tuple[int, int]:
        cdef bint d = self.is_device_accessible
        cdef bint h = self.is_host_accessible
        if d and (not h):
            return (DLDeviceType.kDLCUDA, self.device_id)
        if d and h:
            # TODO: this can also be kDLCUDAManaged, we need more fine-grained checks
            return (DLDeviceType.kDLCUDAHost, 0)
        if (not d) and h:
            return (DLDeviceType.kDLCPU, 0)
        raise BufferError("buffer is neither device-accessible nor host-accessible")

    def __buffer__(self, flags: int, /) -> memoryview:
        # Support for Python-level buffer protocol as per PEP 688.
        # This raises a BufferError unless:
        #   1. Python is 3.12+
        #   2. This Buffer object is host accessible
        raise NotImplementedError("WIP: Buffer.__buffer__ hasn't been implemented yet.")

    def __release_buffer__(self, buffer: memoryview, /):
        # Supporting method paired with __buffer__.
        raise NotImplementedError("WIP: Buffer.__release_buffer__ hasn't been implemented yet.")

    @staticmethod
    def from_handle(ptr: DevicePointerT, size_t size, mr: MemoryResource | None = None) -> Buffer:
        """Create a new :class:`Buffer` object from a pointer.

        Parameters
        ----------
        ptr : :obj:`~_memory.DevicePointerT`
            Allocated buffer handle object
        size : int
            Memory size of the buffer
        mr : :obj:`~_memory.MemoryResource`, optional
            Memory resource associated with the buffer
        """
        # TODO: It is better to take a stream for latter deallocation
        return Buffer._init(ptr, size, mr=mr)


cdef class MemoryResource(_cyMemoryResource, MemoryResourceAttributes, abc.ABC):
    """Abstract base class for memory resources that manage allocation and deallocation of buffers.

    Subclasses must implement methods for allocating and deallocation, as well as properties
    associated with this memory resource from which all allocated buffers will inherit. (Since
    all :class:`Buffer` instances allocated and returned by the :meth:`allocate` method would
    hold a reference to self, the buffer properties are retrieved simply by looking up the underlying
    memory resource's respective property.)
    """
    cdef void _deallocate(self, intptr_t ptr, size_t size, cyStream stream) noexcept:
        self.deallocate(ptr, size, stream)

    @abc.abstractmethod
    def allocate(self, size_t size, stream: Stream = None) -> Buffer:
        """Allocate a buffer of the requested size.

        Parameters
        ----------
        size : int
            The size of the buffer to allocate, in bytes.
        stream : Stream, optional
            The stream on which to perform the allocation asynchronously.
            If None, it is up to each memory resource implementation to decide
            and document the behavior.

        Returns
        -------
        Buffer
            The allocated buffer object, which can be used for device or host operations
            depending on the resource's properties.
        """
        ...

    @abc.abstractmethod
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
            If None, it is up to each memory resource implementation to decide
            and document the behavior.
        """
        ...


# IPC is currently only supported on Linux. On other platforms, the IPC handle
# type is set equal to the no-IPC handle type.
cdef cydriver.CUmemAllocationHandleType _IPC_HANDLE_TYPE = cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR \
    if platform.system() == "Linux" else cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE


cdef class IPCBufferDescriptor:
    """Serializable object describing a buffer that can be shared between processes."""

    cdef:
        bytes _reserved
        size_t _size

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

    cdef:
        int _handle
        object _uuid

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
    max_size : cython.int = 0


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


# Holds DeviceMemoryResource objects imported by this process.
# This enables buffer serialization, as buffers can reduce to a pair
# of comprising the memory resource UUID (the key into this registry)
# and the serialized buffer descriptor.
_ipc_registry = {}


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
    cdef:
        int _dev_id
        cydriver.CUmemoryPool _mempool_handle
        object _attributes
        cydriver.CUmemAllocationHandleType _ipc_handle_type
        bint _mempool_owned
        bint _is_mapped
        object _uuid
        IPCAllocationHandle _alloc_handle
        object __weakref__

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
        cdef cydriver.cuuint64_t current_threshold
        cdef cydriver.cuuint64_t max_threshold = ULLONG_MAX
        cdef cydriver.CUmemPoolProps properties

        if opts is None:
            # Get the current memory pool.
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
        else:
            # Create a new memory pool.
            if opts.ipc_enabled and _IPC_HANDLE_TYPE == cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE:
                raise RuntimeError("IPC is not available on {platform.system()}")

            memset(&properties, 0, sizeof(cydriver.CUmemPoolProps))
            properties.allocType = cydriver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            properties.handleTypes = _IPC_HANDLE_TYPE if opts.ipc_enabled else cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
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

    def __dealloc__(self):
        self.close()

    cpdef close(self):
        """Close the device memory resource and destroy the associated memory pool if owned."""
        if self._mempool_handle == NULL:
            return

        try:
            if self._mempool_owned:
                with nogil:
                    HANDLE_RETURN(cydriver.cuMemPoolDestroy(self._mempool_handle))
        finally:
            if self.is_mapped:
                self.unregister()
            self._dev_id = cydriver.CU_DEVICE_INVALID
            self._mempool_handle = NULL
            self._attributes = None
            self._ipc_handle_type = cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_MAX
            self._mempool_owned = False
            self._is_mapped = False
            self._uuid = None
            self._alloc_handle = None

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

        try:
            return _ipc_registry[uuid]
        except KeyError:
            raise RuntimeError(f"Memory resource {uuid} was not found") from None

    def register(self, uuid: uuid.UUID) -> DeviceMemoryResource:
        """
        Register a mapped memory resource.

        Returns
        -------
        The registered mapped memory resource. If one was previously registered
        with the given key, it is returned.
        """
        existing = _ipc_registry.get(uuid)
        if existing is not None:
            return existing
        assert self._uuid is None or self._uuid == uuid
        _ipc_registry[uuid] = self
        self._uuid = uuid
        return self

    def unregister(self):
        """Unregister this mapped memory resource."""
        assert self.is_mapped
        if _ipc_registry is not None:  # can occur during shutdown catastrophe
            with contextlib.suppress(KeyError):
                del _ipc_registry[self.uuid]

    @property
    def uuid(self) -> Optional[uuid.UUID]:
        """
        A universally unique identifier for this memory resource. Meaningful
        only for IPC-enabled memory resources.
        """
        return self._uuid

    @classmethod
    def from_allocation_handle(cls, device_id: int | Device, alloc_handle: int | IPCAllocationHandle) -> DeviceMemoryResource:
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
         # Quick exit for registry hits.
        uuid = getattr(alloc_handle, 'uuid', None)
        mr = _ipc_registry.get(uuid)
        if mr is not None:
            return mr

        device_id = getattr(device_id, 'device_id', device_id)

        cdef DeviceMemoryResource self = DeviceMemoryResource.__new__(cls)
        self._dev_id = device_id
        self._ipc_handle_type = _IPC_HANDLE_TYPE
        self._mempool_owned = True
        self._is_mapped = True
        #self._alloc_handle = None  # only used for non-imported

        cdef int handle = int(alloc_handle)
        with nogil:
            HANDLE_RETURN(cydriver.cuMemPoolImportFromShareableHandle(
                &(self._mempool_handle), <void*><intptr_t>(handle), _IPC_HANDLE_TYPE, 0)
            )
        if uuid is not None:
            registered = self.register(uuid)
            assert registered is self
        return self

    cpdef IPCAllocationHandle get_allocation_handle(self):
        """Export the memory pool handle to be shared (requires IPC).

        The handle can be used to share the memory pool with other processes.
        The handle is cached in this `MemoryResource` and owned by it.

        Returns
        -------
            The shareable handle for the memory pool.
        """
        # Note: This is Linux only (int for file descriptor)
        cdef int alloc_handle

        if self._alloc_handle is None:
            if not self.is_ipc_enabled:
                raise RuntimeError("Memory resource is not IPC-enabled")
            if self._is_mapped:
                raise RuntimeError("Imported memory resource cannot be exported")

            with nogil:
                HANDLE_RETURN(cydriver.cuMemPoolExportToShareableHandle(
                    &alloc_handle, self._mempool_handle, _IPC_HANDLE_TYPE, 0)
                )
            try:
                assert self._uuid is None
                import uuid
                self._uuid = uuid.uuid4()
                self._alloc_handle = IPCAllocationHandle._init(alloc_handle, self._uuid)
            except:
                os.close(alloc_handle)
                raise
        return self._alloc_handle

    cdef Buffer _allocate(self, size_t size, cyStream stream):
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
        return self._allocate(size, <cyStream>stream)

    cdef void _deallocate(self, intptr_t ptr, size_t size, cyStream stream) noexcept:
        cdef cydriver.CUstream s = stream._handle
        cdef cydriver.CUdeviceptr devptr = <cydriver.CUdeviceptr>ptr
        with nogil:
            HANDLE_RETURN(cydriver.cuMemFreeAsync(devptr, s))

    cpdef deallocate(self, ptr: DevicePointerT, size_t size, stream: Stream = None):
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
        self._deallocate(<intptr_t>ptr, size, <cyStream>stream)

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
    def is_handle_owned(self) -> bool:
        """Whether the memory resource handle is owned. If False, ``close`` has no effect."""
        return self._mempool_owned

    @property
    def is_mapped(self) -> bool:
        """
        Whether this is a mapping of an IPC-enabled memory resource from
        another process.  If True, allocation is not permitted.
        """
        return self._is_mapped

    @property
    def is_device_accessible(self) -> bool:
        """Return True. This memory resource provides device-accessible buffers."""
        return True

    @property
    def is_host_accessible(self) -> bool:
        """Return False. This memory resource does not provide host-accessible buffers."""
        return False

    @property
    def is_ipc_enabled(self) -> bool:
        """Whether this memory resource has IPC enabled."""
        return self._ipc_handle_type != cydriver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE


def _deep_reduce_device_memory_resource(mr):
    from . import Device
    device = Device(mr.device_id)
    alloc_handle = mr.get_allocation_handle()
    return mr.from_allocation_handle, (device, alloc_handle)


multiprocessing.reduction.register(DeviceMemoryResource, _deep_reduce_device_memory_resource)


class LegacyPinnedMemoryResource(MemoryResource):
    """Create a pinned memory resource that uses legacy cuMemAllocHost/cudaMallocHost
    APIs.
    """

    # TODO: support creating this MR with flags that are later passed to cuMemHostAlloc?

    def allocate(self, size_t size, stream: Stream = None) -> Buffer:
        """Allocate a buffer of the requested size.

        Parameters
        ----------
        size : int
            The size of the buffer to allocate, in bytes.
        stream : Stream, optional
            Currently ignored

        Returns
        -------
        Buffer
            The allocated buffer object, which is accessible on both host and device.
        """
        if stream is None:
            stream = default_stream()
        err, ptr = driver.cuMemAllocHost(size)
        raise_if_driver_error(err)
        return Buffer._init(ptr, size, self, stream)

    def deallocate(self, ptr: DevicePointerT, size_t size, stream: Stream):
        """Deallocate a buffer previously allocated by this resource.

        Parameters
        ----------
        ptr : :obj:`~_memory.DevicePointerT`
            The pointer or handle to the buffer to deallocate.
        size : int
            The size of the buffer to deallocate, in bytes.
        stream : Stream
            The stream on which to perform the deallocation synchronously.
        """
        stream.sync()
        err, = driver.cuMemFreeHost(ptr)
        raise_if_driver_error(err)

    @property
    def is_device_accessible(self) -> bool:
        """bool: this memory resource provides device-accessible buffers."""
        return True

    @property
    def is_host_accessible(self) -> bool:
        """bool: this memory resource provides host-accessible buffers."""
        return True

    @property
    def device_id(self) -> int:
        """This memory resource is not bound to any GPU."""
        raise RuntimeError("a pinned memory resource is not bound to any GPU")


class _SynchronousMemoryResource(MemoryResource):
    __slots__ = ("_dev_id",)

    def __init__(self, device_id : int | Device):
        self._dev_id = getattr(device_id, 'device_id', device_id)

    def allocate(self, size, stream=None) -> Buffer:
        if stream is None:
            stream = default_stream()
        err, ptr = driver.cuMemAlloc(size)
        raise_if_driver_error(err)
        return Buffer._init(ptr, size, self)

    def deallocate(self, ptr, size, stream):
        stream.sync()
        err, = driver.cuMemFree(ptr)
        raise_if_driver_error(err)

    @property
    def is_device_accessible(self) -> bool:
        return True

    @property
    def is_host_accessible(self) -> bool:
        return False

    @property
    def device_id(self) -> int:
        return self._dev_id


VirtualMemoryHandleTypeT = Literal["posix_fd", "generic", "none", "win32", "win32_kmt", "fabric"]
VirtualMemoryLocationTypeT = Literal["device", "host", "host_numa", "host_numa_current"]
VirtualMemoryGranularityT = Literal["minimum", "recommended"]
VirtualMemoryAccessTypeT = Literal["rw", "r", "none"]
VirtualMemoryAllocationTypeT = Literal["pinned", "managed"]


@dataclass
class VirtualMemoryResourceOptions:
    """A configuration object for the VirtualMemoryResource
       Stores configuration information which tells the resource how to use the CUDA VMM APIs

    Args:
        handle_type: Export handle type for the physical allocation. Use
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR on Linux if you plan to
            import/export the allocation (required for cuMemRetainAllocationHandle).
            Use CU_MEM_HANDLE_TYPE_NONE if you don't need an exportable handle.
        gpu_direct_rdma: Hint that the allocation should be GDR-capable (if supported).
        granularity: 'recommended' or 'minimum'. Controls granularity query and size rounding.
        addr_hint: A (optional) virtual address hint to try to reserve at. 0 -> let CUDA choose.
        addr_align: Alignment for the VA reservation. If None, use the queried granularity.
        peers: Extra device IDs that should be granted access in addition to `device`.
        self_access: Access flags for the owning device ('rw', 'r', or 'none').
        peer_access: Access flags for peers ('rw' or 'r').
    """
    # Human-friendly strings; normalized in __post_init__
    allocation_type: VirtualMemoryAllocationTypeT = "pinned"
    location_type: VirtualMemoryLocationTypeT = "device"
    handle_type: VirtualMemoryHandleTypeT = "posix_fd"
    granularity: VirtualMemoryGranularityT = "recommended"
    gpu_direct_rdma: bool = True
    addr_hint: Optional[int] = 0
    addr_align: Optional[int] = None
    peers: Iterable[int] = field(default_factory=tuple)
    self_access: VirtualMemoryAccessTypeT = "rw"
    peer_access: VirtualMemoryAccessTypeT = "rw"

    _a = driver.CUmemAccess_flags
    _access_flags = {"rw": _a.CU_MEM_ACCESS_FLAGS_PROT_READWRITE, "r": _a.CU_MEM_ACCESS_FLAGS_PROT_READ, "none": 0}
    _h = driver.CUmemAllocationHandleType
    _handle_types = {"none": _h.CU_MEM_HANDLE_TYPE_NONE, "posix_fd": _h.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, "win32": _h.CU_MEM_HANDLE_TYPE_WIN32, "win32_kmt": _h.CU_MEM_HANDLE_TYPE_WIN32_KMT, "fabric": _h.CU_MEM_HANDLE_TYPE_FABRIC}
    _g = driver.CUmemAllocationGranularity_flags
    _granularity = {"recommended": _g.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED, "minimum": _g.CU_MEM_ALLOC_GRANULARITY_MINIMUM}
    _l = driver.CUmemLocationType
    _location_type = {"device": _l.CU_MEM_LOCATION_TYPE_DEVICE, "host": _l.CU_MEM_LOCATION_TYPE_HOST, "host_numa": _l.CU_MEM_LOCATION_TYPE_HOST_NUMA, "host_numa_current": _l.CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT}
    # CUDA 13+ exposes MANAGED in CUmemAllocationType; older 12.x does not
    _a = driver.CUmemAllocationType
    _allocation_type = {"pinned": _a.CU_MEM_ALLOCATION_TYPE_PINNED}
    ver_major, ver_minor = get_binding_version()
    if ver_major >= 13:
        _allocation_type["managed"] = _a.CU_MEM_ALLOCATION_TYPE_MANAGED

    @staticmethod
    def _access_to_flags(spec: str):
        flags = VirtualMemoryResourceOptions._access_flags.get(spec)
        if flags is None:
            raise ValueError(f"Unknown access spec: {spec!r}")
        return flags

    @staticmethod
    def _allocation_type_to_driver(spec: str):
        alloc_type = VirtualMemoryResourceOptions._allocation_type.get(spec)
        if alloc_type is None:
            raise ValueError(f"Unsupported allocation_type: {spec!r}")
        return alloc_type

    @staticmethod
    def _location_type_to_driver(spec: str):
        loc_type = VirtualMemoryResourceOptions._location_type.get(spec)
        if loc_type is None:
            raise ValueError(f"Unsupported location_type: {spec!r}")
        return loc_type

    @staticmethod
    def _handle_type_to_driver(spec: str):
        handle_type = VirtualMemoryResourceOptions._handle_types.get(spec)
        if handle_type is None:
            raise ValueError(f"Unsupported handle_type: {spec!r}")
        return handle_type

    @staticmethod
    def _granularity_to_driver(spec: str):
        granularity = VirtualMemoryResourceOptions._granularity.get(spec)
        if granularity is None:
            raise ValueError(f"Unsupported granularity: {spec!r}")
        return granularity


class VirtualMemoryResource(MemoryResource):
    """Create a device memory resource that uses the CUDA VMM APIs to allocate memory.

    Parameters
    ----------
    device_id : int
        Device ordinal for which a memory resource is constructed.

    config : VirtualMemoryResourceOptions
        A configuration object for the VirtualMemoryResource
    """
    def __init__(self, device, config: VirtualMemoryResourceOptions = None):
        self.device = device
        self.config = check_or_create_options(
            VirtualMemoryResourceOptions, config, "VirtualMemoryResource options", keep_none=False
        )
        if self.config.location_type == "host":
            self.device = None
        if platform.system() == "Windows":
            raise NotImplementedError("VirtualMemoryResource is not supported on Windows")

    @staticmethod
    def _align_up(size: int, gran: int) -> int:
        """
        Align a size up to the nearest multiple of a granularity.
        """
        return (size + gran - 1) & ~(gran - 1)

    def modify_allocation(self, buf: Buffer, new_size: int, config: VirtualMemoryResourceOptions = None) -> Buffer:
        """
        Grow an existing allocation using CUDA VMM, with a configurable policy.

        This implements true growing allocations that preserve the base pointer
        by extending the virtual address range and mapping additional physical memory.

        This function uses transactional allocation: if any step fails, the original buffer is not modified and
        all steps the function took are rolled back so a new allocation is not created.

        Parameters
        ----------
        buf : Buffer
            The existing buffer to grow
        new_size : int
            The new total size for the allocation
        config : VirtualMemoryResourceOptions, optional
            Configuration for the new physical memory chunks. If None, uses current config.

        Returns
        -------
        Buffer
            The same buffer with updated size and properties, preserving the original pointer
        """
        if config is not None:
            self.config = config

        # Build allocation properties for new chunks
        prop = driver.CUmemAllocationProp()
        prop.type = VirtualMemoryResourceOptions._allocation_type_to_driver(self.config.allocation_type)
        prop.location.type = VirtualMemoryResourceOptions._location_type_to_driver(self.config.location_type)
        prop.location.id = self.device.device_id
        prop.allocFlags.gpuDirectRDMACapable = 1 if self.config.gpu_direct_rdma else 0
        prop.requestedHandleTypes = VirtualMemoryResourceOptions._handle_type_to_driver(self.config.handle_type)

        # Query granularity
        gran_flag = VirtualMemoryResourceOptions._granularity_to_driver(self.config.granularity)
        res, gran = driver.cuMemGetAllocationGranularity(prop, gran_flag)
        raise_if_driver_error(res)

        # Calculate sizes
        additional_size = new_size - buf.size
        if additional_size <= 0:
            # Same size: only update access policy if needed; avoid zero-sized driver calls
            descs = self._build_access_descriptors(prop)
            if descs:
                res, = driver.cuMemSetAccess(int(buf.handle), buf.size, descs, len(descs))
                raise_if_driver_error(res)
            return buf

        aligned_additional_size = VirtualMemoryResource._align_up(additional_size, gran)
        total_aligned_size = VirtualMemoryResource._align_up(new_size, gran)
        aligned_prev_size = total_aligned_size - aligned_additional_size
        addr_align = self.config.addr_align or gran

        # Try to extend the existing VA range first
        res, new_ptr = driver.cuMemAddressReserve(
            aligned_additional_size,
            addr_align,
            int(buf.handle) + aligned_prev_size,  # fixedAddr hint - aligned end of current range
            0
        )

        if res != driver.CUresult.CUDA_SUCCESS or new_ptr != (int(buf.handle) + aligned_prev_size):
            # Check for specific errors that are not recoverable with the slow path
            if res in (driver.CUresult.CUDA_ERROR_INVALID_VALUE, driver.CUresult.CUDA_ERROR_NOT_PERMITTED, driver.CUresult.CUDA_ERROR_NOT_INITIALIZED, driver.CUresult.CUDA_ERROR_NOT_SUPPORTED):
                raise_if_driver_error(res)
            res2, = driver.cuMemAddressFree(new_ptr, aligned_additional_size)
            raise_if_driver_error(res2)
            # Fallback: couldn't extend contiguously, need full remapping
            return self._grow_allocation_slow_path(buf, new_size, prop, aligned_additional_size, total_aligned_size, addr_align)
        else:
            # Success! We can extend the VA range contiguously
            return self._grow_allocation_fast_path(buf, new_size, prop, aligned_additional_size, new_ptr)

    def _grow_allocation_fast_path(self, buf: Buffer, new_size: int, prop: driver.CUmemAllocationProp,
                                   aligned_additional_size: int, new_ptr: int) -> Buffer:
        """
        Fast path for growing a virtual memory allocation when the new region can be
        reserved contiguously after the existing buffer.

        This function creates and maps new physical memory for the additional size,
        sets access permissions, and updates the buffer size in place (the pointer
        remains unchanged).

        Args:
            buf (Buffer): The buffer to grow.
            new_size (int): The new total size in bytes.
            prop (driver.CUmemAllocationProp): Allocation properties for the new memory.
            aligned_additional_size (int): The size of the new region to allocate, aligned to granularity.
            new_ptr (int): The address of the newly reserved contiguous VA region (should be at the end of the current buffer).

        Returns:
            Buffer: The same buffer object with its size updated to `new_size`.
        """
        with Transaction() as trans:
            # Create new physical memory for the additional size
            trans.append(lambda np=new_ptr, s=aligned_additional_size: raise_if_driver_error(driver.cuMemAddressFree(np, s)[0]))
            res, new_handle = driver.cuMemCreate(aligned_additional_size, prop, 0)
            raise_if_driver_error(res)
            # Register undo for creation
            trans.append(lambda h=new_handle: raise_if_driver_error(driver.cuMemRelease(h)[0]))

            # Map the new physical memory to the extended VA range
            res, = driver.cuMemMap(new_ptr, aligned_additional_size, 0, new_handle, 0)
            raise_if_driver_error(res)
            # Register undo for mapping
            trans.append(lambda np=new_ptr, s=aligned_additional_size: raise_if_driver_error(driver.cuMemUnmap(np, s)[0]))

            # Set access permissions for the new portion
            descs = self._build_access_descriptors(prop)
            if descs:
                res, = driver.cuMemSetAccess(new_ptr, aligned_additional_size, descs, len(descs))
                raise_if_driver_error(res)

            # All succeeded, cancel undo actions
            trans.commit()

        # Update the buffer size (pointer stays the same)
        buf._size = new_size
        return buf

    def _grow_allocation_slow_path(self, buf: Buffer, new_size: int, prop: driver.CUmemAllocationProp,
                                   aligned_additional_size: int, total_aligned_size: int, addr_align: int) -> Buffer:
        """
        Slow path for growing a virtual memory allocation when the new region cannot be
        reserved contiguously after the existing buffer.

        This function reserves a new, larger virtual address (VA) range, remaps the old
        physical memory to the beginning of the new VA range, creates and maps new physical
        memory for the additional size, sets access permissions, and updates the buffer's
        pointer and size.

        Args:
            buf (Buffer): The buffer to grow.
            new_size (int): The new total size in bytes.
            prop (driver.CUmemAllocationProp): Allocation properties for the new memory.
            aligned_additional_size (int): The size of the new region to allocate, aligned to granularity.
            total_aligned_size (int): The total new size to reserve, aligned to granularity.
            addr_align (int): The required address alignment for the new VA range.

        Returns:
            Buffer: The buffer object updated with the new pointer and size.
        """
        with Transaction() as trans:
            # Reserve a completely new, larger VA range
            res, new_ptr = driver.cuMemAddressReserve(total_aligned_size, addr_align, 0, 0)
            raise_if_driver_error(res)
            # Register undo for VA reservation
            trans.append(lambda np=new_ptr, s=total_aligned_size: raise_if_driver_error(driver.cuMemAddressFree(np, s)[0]))

            # Get the old allocation handle for remapping
            result, old_handle = driver.cuMemRetainAllocationHandle(buf.handle)
            raise_if_driver_error(result)
            # Register undo for old_handle
            trans.append(lambda h=old_handle: raise_if_driver_error(driver.cuMemRelease(h)[0]))

            # Unmap the old VA range (aligned previous size)
            aligned_prev_size = total_aligned_size - aligned_additional_size
            result, = driver.cuMemUnmap(int(buf.handle), aligned_prev_size)
            raise_if_driver_error(result)

            def _remap_old():
                # Try to remap the old physical memory back to the original VA range
                try:
                    res, = driver.cuMemMap(int(buf.handle), aligned_prev_size, 0, old_handle, 0)
                    raise_if_driver_error(res)
                except Exception:
                    pass
            trans.append(_remap_old)

            # Remap the old physical memory to the new VA range (aligned previous size)
            res, = driver.cuMemMap(int(new_ptr), aligned_prev_size, 0, old_handle, 0)
            raise_if_driver_error(res)

            # Register undo for mapping
            trans.append(lambda np=new_ptr, s=aligned_prev_size: raise_if_driver_error(driver.cuMemUnmap(np, s)[0]))

            # Create new physical memory for the additional size
            res, new_handle = driver.cuMemCreate(aligned_additional_size, prop, 0)
            raise_if_driver_error(res)

            # Register undo for new physical memory
            trans.append(lambda h=new_handle: raise_if_driver_error(driver.cuMemRelease(h)[0]))

            # Map the new physical memory to the extended portion (aligned offset)
            res, = driver.cuMemMap(int(new_ptr) + aligned_prev_size, aligned_additional_size, 0, new_handle, 0)
            raise_if_driver_error(res)

            # Register undo for mapping
            trans.append(lambda base=int(new_ptr), offs=aligned_prev_size, s=aligned_additional_size: raise_if_driver_error(driver.cuMemUnmap(base + offs, s)[0]))

            # Set access permissions for the entire new range
            descs = self._build_access_descriptors(prop)
            if descs:
                res, = driver.cuMemSetAccess(new_ptr, total_aligned_size, descs, len(descs))
                raise_if_driver_error(res)

            # All succeeded, cancel undo actions
            trans.commit()

        # Free the old VA range (aligned previous size)
        res2, = driver.cuMemAddressFree(int(buf.handle), aligned_prev_size)
        raise_if_driver_error(res2)

        # Invalidate the old buffer so its destructor won't try to free again
        buf._ptr = 0
        buf._ptr_obj = None
        buf._size = 0
        buf._mr = None

        # Return a new Buffer for the new mapping
        return Buffer.from_handle(ptr=new_ptr, size=new_size, mr=self)


    def _build_access_descriptors(self, prop: driver.CUmemAllocationProp) -> list:
        """
        Build access descriptors for memory access permissions.

        Returns
        -------
        list
            List of CUmemAccessDesc objects for setting memory access
        """
        descs = []

        # Owner access
        owner_flags = VirtualMemoryResourceOptions._access_to_flags(self.config.self_access)
        if owner_flags:
            d = driver.CUmemAccessDesc()
            d.location.type = prop.location.type
            d.location.id = prop.location.id
            d.flags = owner_flags
            descs.append(d)

        # Peer device access
        peer_flags = VirtualMemoryResourceOptions._access_to_flags(self.config.peer_access)
        if peer_flags:
            for peer_dev in self.config.peers:
                d = driver.CUmemAccessDesc()
                d.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
                d.location.id = int(peer_dev)
                d.flags = peer_flags
                descs.append(d)

        return descs


    def allocate(self, size: int, stream: Stream = None) -> Buffer:
        """
        Allocate a buffer of the given size using CUDA virtual memory.

        Parameters
        ----------
        size : int
            The size in bytes of the buffer to allocate.
        stream : Stream, optional
            CUDA stream to associate with the allocation (not currently supported).

        Returns
        -------
        Buffer
            A Buffer object representing the allocated virtual memory.

        Raises
        ------
        NotImplementedError
            If a stream is provided or if the location type is not device memory.
        CUDAError
            If any CUDA driver API call fails during allocation.

        Notes
        -----
        This method uses transactional allocation: if any step fails, all resources
        allocated so far are automatically cleaned up. The allocation is performed
        with the configured granularity, access permissions, and peer access as
        specified in the resource's configuration.
        """
        if stream is not None:
            raise NotImplementedError("Stream is not supported with VirtualMemoryResource")

        config = self.config
        # ---- Build allocation properties ----
        prop = driver.CUmemAllocationProp()
        prop.type = VirtualMemoryResourceOptions._allocation_type_to_driver(config.allocation_type)

        prop.location.type = VirtualMemoryResourceOptions._location_type_to_driver(config.location_type)
        prop.location.id = self.device.device_id if config.location_type == "device" else -1
        prop.allocFlags.gpuDirectRDMACapable = 1 if config.gpu_direct_rdma else 0
        prop.requestedHandleTypes = VirtualMemoryResourceOptions._handle_type_to_driver(config.handle_type)

        # ---- Query and apply granularity ----
        # Choose min vs recommended granularity per config
        gran_flag = VirtualMemoryResourceOptions._granularity_to_driver(config.granularity)
        res, gran = driver.cuMemGetAllocationGranularity(prop, gran_flag)
        raise_if_driver_error(res)

        aligned_size = VirtualMemoryResource._align_up(size, gran)
        addr_align = config.addr_align or gran

        # ---- Transactional allocation ----
        with Transaction() as trans:
            # ---- Create physical memory ----
            res, handle = driver.cuMemCreate(aligned_size, prop, 0)
            raise_if_driver_error(res)
            # Register undo for physical memory
            trans.append(lambda h=handle: raise_if_driver_error(driver.cuMemRelease(h)[0]))

            # ---- Reserve VA space ----
            # Potentially, use a separate size for the VA reservation from the physical allocation size
            res, ptr = driver.cuMemAddressReserve(aligned_size, addr_align, config.addr_hint, 0)
            raise_if_driver_error(res)
            # Register undo for VA reservation
            trans.append(lambda p=ptr, s=aligned_size: raise_if_driver_error(driver.cuMemAddressFree(p, s)[0]))

            # ---- Map physical memory into VA ----
            res, = driver.cuMemMap(ptr, aligned_size, 0, handle, 0)
            trans.append(lambda p=ptr, s=aligned_size: raise_if_driver_error(driver.cuMemUnmap(p, s)[0]))
            raise_if_driver_error(res)

            # ---- Set access for owner + peers ----
            descs = self._build_access_descriptors(prop)
            if descs:
                res, = driver.cuMemSetAccess(ptr, aligned_size, descs, len(descs))
                raise_if_driver_error(res)

            trans.commit()

        # Done — return a Buffer that tracks this VA range
        buf = Buffer.from_handle(ptr=ptr, size=aligned_size, mr=self)
        return buf

    def deallocate(self, ptr: int, size: int, stream: Stream=None) -> None:
        """
        Deallocate memory on the device using CUDA VMM APIs.
        """
        result, handle = driver.cuMemRetainAllocationHandle(ptr)
        raise_if_driver_error(result)
        result, = driver.cuMemUnmap(ptr, size)
        raise_if_driver_error(result)
        result, = driver.cuMemAddressFree(ptr, size)
        raise_if_driver_error(result)
        result, = driver.cuMemRelease(handle)
        raise_if_driver_error(result)


    @property
    def is_device_accessible(self) -> bool:
        """
        Indicates whether the allocated memory is accessible from the device.
        """
        return self.config.location_type == "device"

    @property
    def is_host_accessible(self) -> bool:
        """
        Indicates whether the allocated memory is accessible from the host.
        """
        return self.config.location_type == "host"

    @property
    def device_id(self) -> int:
        """
        Get the device ID associated with this memory resource.

        Returns:
            int: CUDA device ID. -1 if the memory resource allocates host memory
        """
        return self.device.device_id if self.config.location_type == "device" else -1

    def __repr__(self) -> str:
        """
        Return a string representation of the VirtualMemoryResource.

        Returns:
            str: A string describing the object
        """
        return f"<VirtualMemoryResource device={self.device}>"
