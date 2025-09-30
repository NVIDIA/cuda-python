# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from libc.stdint cimport uintptr_t
from cuda.core.experimental._utils.cuda_utils cimport (
    _check_driver_error as raise_if_driver_error,
    check_or_create_options,
)
import sys

from typing import TypeVar, Union, TYPE_CHECKING
import abc
from typing import TypeVar, Union, Optional, Iterable, Literal
from dataclasses import dataclass, field
import array
import cython
import os
import platform
import weakref


from cuda.core.experimental._dlpack import DLDeviceType, make_py_capsule
from cuda.core.experimental._stream import Stream, default_stream
from cuda.core.experimental._utils.cuda_utils import ( driver, Transaction, get_binding_version,
                                                       handle_return,
                                                     )

if platform.system() == "Linux":
    import socket
# Helper to invoke CUDA driver functions with standard error handling.
def _driver_call(func, /, *args):
    handle_return(func(*args))


if TYPE_CHECKING:
    import cuda.bindings.driver
    from cuda.core.experimental._device import Device

# TODO: define a memory property mixin class and make Buffer and
# MemoryResource both inherit from it


PyCapsule = TypeVar("PyCapsule")
"""Represent the capsule type."""

DevicePointerT = Union[driver.CUdeviceptr, int, None]
"""A type union of :obj:`~driver.CUdeviceptr`, `int` and `None` for hinting :attr:`Buffer.handle`."""


cdef class Buffer:
    """Represent a handle to allocated memory.

    This generic object provides a unified representation for how
    different memory resources are to give access to their memory
    allocations.

    Support for data interchange mechanisms are provided by DLPack.
    """

    cdef:
        uintptr_t _ptr
        size_t _size
        object _mr
        object _ptr_obj

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Buffer objects cannot be instantiated directly. Please use MemoryResource APIs.")

    @classmethod
    def _init(cls, ptr: DevicePointerT, size_t size, mr: MemoryResource | None = None):
        cdef Buffer self = Buffer.__new__(cls)
        self._ptr = <uintptr_t>(int(ptr))
        self._ptr_obj = ptr
        self._size = size
        self._mr = mr
        return self

    def __del__(self):
        self._shutdown_safe_close()

    cdef _shutdown_safe_close(self, stream: Stream = None, is_shutting_down=sys.is_finalizing):
        if is_shutting_down and is_shutting_down():
            return
        if self._ptr and self._mr is not None:
            self._mr.deallocate(self._ptr, self._size, stream)
            self._ptr = 0
            self._mr = None
            self._ptr_obj = None

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
        self._shutdown_safe_close(stream, is_shutting_down=None)

    @property
    def handle(self) -> DevicePointerT:
        """Return the buffer handle object.

        .. caution::

            This handle is a Python object. To get the memory address of the underlying C
            handle, call ``int(Buffer.handle)``.
        """
        return self._ptr_obj

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

    def export(self) -> IPCBufferDescriptor:
        """Export a buffer allocated for sharing between processes."""
        if not self._mr.is_ipc_enabled:
            raise RuntimeError("Memory resource is not IPC-enabled")
        err, ptr = driver.cuMemPoolExportPointer(self.handle)
        raise_if_driver_error(err)
        return IPCBufferDescriptor._init(ptr.reserved, self.size)

    @classmethod
    def import_(cls, mr: MemoryResource, ipc_buffer: IPCBufferDescriptor) -> Buffer:
        """Import a buffer that was exported from another process."""
        if not mr.is_ipc_enabled:
            raise RuntimeError("Memory resource is not IPC-enabled")
        share_data = driver.CUmemPoolPtrExportData()
        share_data.reserved = ipc_buffer._reserved
        err, ptr = driver.cuMemPoolImportPointer(mr._mempool_handle, share_data)
        raise_if_driver_error(err)
        return Buffer.from_handle(ptr, ipc_buffer.size, mr)

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
        return Buffer._init(ptr, size, mr=mr)


class MemoryResource(abc.ABC):
    """Abstract base class for memory resources that manage allocation and deallocation of buffers.

    Subclasses must implement methods for allocating and deallocation, as well as properties
    associated with this memory resource from which all allocated buffers will inherit. (Since
    all :class:`Buffer` instances allocated and returned by the :meth:`allocate` method would
    hold a reference to self, the buffer properties are retrieved simply by looking up the underlying
    memory resource's respective property.)
    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the memory resource.

        Subclasses may use additional arguments to configure the resource.
        """
        ...

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


# IPC is currently only supported on Linux. On other platforms, the IPC handle
# type is set equal to the no-IPC handle type.

_NOIPC_HANDLE_TYPE = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
_IPC_HANDLE_TYPE = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR \
    if platform.system() == "Linux" else _NOIPC_HANDLE_TYPE

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
        # This is subject to change if the CUmemPoolPtrExportData struct/object changes.
        return (self._reconstruct, (self._reserved, self._size))

    @property
    def size(self):
        return self._size

    @classmethod
    def _reconstruct(cls, reserved, size):
        instance = cls._init(reserved, size)
        return instance


cdef class IPCAllocationHandle:
    """Shareable handle to an IPC-enabled device memory pool."""

    cdef:
        int _handle

    def __init__(self, *arg, **kwargs):
        raise RuntimeError("IPCAllocationHandle objects cannot be instantiated directly. Please use MemoryResource APIs.")

    @classmethod
    def _init(cls, handle: int):
        cdef IPCAllocationHandle self = IPCAllocationHandle.__new__(cls)
        assert handle >= 0
        self._handle = handle
        return self

    cpdef close(self):
        """Close the handle."""
        if self._handle >= 0:
            try:
                os.close(self._handle)
            finally:
                self._handle = -1

    def __del__(self):
        """Close the handle."""
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


cdef class IPCChannel:
    """Communication channel for sharing IPC-enabled memory pools."""

    cdef:
        object _proxy

    def __init__(self):
        if platform.system() == "Linux":
            self._proxy = IPCChannelUnixSocket._init()
        else:
            raise RuntimeError("IPC is not available on {platform.system()}")


cdef class IPCChannelUnixSocket:
    """Unix-specific channel for sharing memory pools over sockets."""

    cdef:
        object _sock_out
        object _sock_in

    def __init__(self, *arg, **kwargs):
        raise RuntimeError("IPCChannelUnixSocket objects cannot be instantiated directly. Please use MemoryResource APIs.")

    @classmethod
    def _init(cls):
        cdef IPCChannelUnixSocket self = IPCChannelUnixSocket.__new__(cls)
        self._sock_out, self._sock_in = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        return self

    cpdef _send_allocation_handle(self, alloc_handle: IPCAllocationHandle):
        """Sends over this channel an allocation handle for exporting a
        shared memory pool."""
        self._sock_out.sendmsg(
            [],
            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [int(alloc_handle)]))]
        )

    cpdef IPCAllocationHandle _receive_allocation_handle(self):
        """Receives over this channel an allocation handle for importing a
        shared memory pool."""
        fds = array.array("i")
        _, ancillary_data, _, _ = self._sock_in.recvmsg(0, socket.CMSG_LEN(fds.itemsize))
        assert len(ancillary_data) == 1
        cmsg_level, cmsg_type, cmsg_data = ancillary_data[0]
        assert cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS
        fds.frombytes(cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])
        return IPCAllocationHandle._init(int(fds[0]))


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
                err, value = driver.cuMemPoolGetAttribute(mr._mempool_handle, attr_enum)
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


class DeviceMemoryResource(MemoryResource):
    """Create a device memory resource managing a stream-ordered memory pool.

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
    """
    __slots__ = "_dev_id", "_mempool_handle", "_attributes", "_ipc_handle_type", "_mempool_owned", "_is_imported"

    def __init__(self, device_id: int | Device, options=None):
        device_id = getattr(device_id, 'device_id', device_id)
        opts = check_or_create_options(
            DeviceMemoryResourceOptions, options, "DeviceMemoryResource options", keep_none=True
        )

        if opts is None:
            # Get the current memory pool.
            self._dev_id = device_id
            self._mempool_handle = None
            self._attributes = None
            self._ipc_handle_type = _NOIPC_HANDLE_TYPE
            self._mempool_owned = False
            self._is_imported = False

            err, self._mempool_handle = driver.cuDeviceGetMemPool(self.device_id)
            raise_if_driver_error(err)

            # Set a higher release threshold to improve performance when there are no active allocations.
            # By default, the release threshold is 0, which means memory is immediately released back
            # to the OS when there are no active suballocations, causing performance issues.
            # Check current release threshold
            err, current_threshold = driver.cuMemPoolGetAttribute(
                self._mempool_handle, driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD
            )
            raise_if_driver_error(err)
            # If threshold is 0 (default), set it to maximum to retain memory in the pool
            if int(current_threshold) == 0:
                err, = driver.cuMemPoolSetAttribute(
                    self._mempool_handle,
                    driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    driver.cuuint64_t(0xFFFFFFFFFFFFFFFF),
                )
                raise_if_driver_error(err)
        else:
            # Create a new memory pool.
            if opts.ipc_enabled and _IPC_HANDLE_TYPE == _NOIPC_HANDLE_TYPE:
                raise RuntimeError("IPC is not available on {platform.system()}")

            properties = driver.CUmemPoolProps()
            properties.allocType = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            properties.handleTypes = _IPC_HANDLE_TYPE if opts.ipc_enabled else _NOIPC_HANDLE_TYPE
            properties.location = driver.CUmemLocation()
            properties.location.id = device_id
            properties.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            properties.maxSize = opts.max_size
            properties.win32SecurityAttributes = 0
            properties.usage = 0

            self._dev_id = device_id
            self._mempool_handle = None
            self._attributes = None
            self._ipc_handle_type = properties.handleTypes
            self._mempool_owned = True
            self._is_imported = False

            err, self._mempool_handle = driver.cuMemPoolCreate(properties)
            raise_if_driver_error(err)

    def __del__(self):
        self.close()

    def close(self):
        """Close the device memory resource and destroy the associated memory pool if owned."""
        if self._mempool_handle is not None and self._mempool_owned:
            err, = driver.cuMemPoolDestroy(self._mempool_handle)
            raise_if_driver_error(err)

            self._dev_id = None
            self._mempool_handle = None
            self._attributes = None
            self._ipc_handle_type = _NOIPC_HANDLE_TYPE
            self._mempool_owned = False
            self._is_imported = False

    @classmethod
    def from_shared_channel(cls, device_id: int | Device, channel: IPCChannel) -> DeviceMemoryResource:
        """Create a device memory resource from a memory pool shared over an IPC channel."""
        device_id = getattr(device_id, 'device_id', device_id)
        alloc_handle = channel._proxy._receive_allocation_handle()
        return cls._from_allocation_handle(device_id, alloc_handle)

    @classmethod
    def _from_allocation_handle(cls, device_id: int | Device, alloc_handle: IPCAllocationHandle) -> DeviceMemoryResource:
        """Create a device memory resource from an allocation handle.

        Construct a new `DeviceMemoryResource` instance that imports a memory
        pool from a shareable handle. The memory pool is marked as owned, and
        the resource is associated with the specified `device_id`.

        Parameters
        ----------
        device_id : int | Device
            The ID of the device or a Device object for which the memory
            resource is created.

        alloc_handle : int
            The shareable handle of the device memory resource to import.

        Returns
        -------
            A new device memory resource instance with the imported handle.
        """
        device_id = getattr(device_id, 'device_id', device_id)

        self = cls.__new__(cls)
        self._dev_id = device_id
        self._mempool_handle = None
        self._attributes = None
        self._ipc_handle_type = _IPC_HANDLE_TYPE
        self._mempool_owned = True
        self._is_imported = True

        err, self._mempool_handle = driver.cuMemPoolImportFromShareableHandle(int(alloc_handle), _IPC_HANDLE_TYPE, 0)
        raise_if_driver_error(err)

        return self

    def share_to_channel(self, channel : IPCChannel):
        if not self.is_ipc_enabled:
            raise RuntimeError("Memory resource is not IPC-enabled")
        channel._proxy._send_allocation_handle(self._get_allocation_handle())

    def _get_allocation_handle(self) -> IPCAllocationHandle:
        """Export the memory pool handle to be shared (requires IPC).

        The handle can be used to share the memory pool with other processes.
        The handle is cached in this `MemoryResource` and owned by it.

        Returns
        -------
            The shareable handle for the memory pool.
        """
        if not self.is_ipc_enabled:
            raise RuntimeError("Memory resource is not IPC-enabled")
        err, alloc_handle = driver.cuMemPoolExportToShareableHandle(self._mempool_handle, _IPC_HANDLE_TYPE, 0)
        raise_if_driver_error(err)
        return IPCAllocationHandle._init(alloc_handle)

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
        if self._is_imported:
            raise TypeError("Cannot allocate from shared memory pool imported via IPC")
        if stream is None:
            stream = default_stream()
        err, ptr = driver.cuMemAllocFromPoolAsync(size, self._mempool_handle, stream.handle)
        raise_if_driver_error(err)
        return Buffer._init(ptr, size, self)

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
            If None, an internal stream is used.
        """
        if stream is None:
            stream = default_stream()
        err, = driver.cuMemFreeAsync(ptr, stream.handle)
        raise_if_driver_error(err)

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
    def handle(self) -> cuda.bindings.driver.CUmemoryPool:
        """Handle to the underlying memory pool."""
        return self._mempool_handle

    @property
    def is_handle_owned(self) -> bool:
        """Whether the memory resource handle is owned. If False, ``close`` has no effect."""
        return self._mempool_owned

    @property
    def is_imported(self) -> bool:
        """Whether the memory resource was imported from another process. If True, allocation is not permitted."""
        return self._is_imported

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
        return self._ipc_handle_type != _NOIPC_HANDLE_TYPE


class LegacyPinnedMemoryResource(MemoryResource):
    """Create a pinned memory resource that uses legacy cuMemAllocHost/cudaMallocHost
    APIs.
    """

    def __init__(self):
        # TODO: support flags from cuMemHostAlloc?
        self._handle = None

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
        err, ptr = driver.cuMemAllocHost(size)
        raise_if_driver_error(err)
        return Buffer._init(ptr, size, self)

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
            If None, no synchronization would happen.
        """
        if stream:
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
        self._handle = None
        self._dev_id = getattr(device_id, 'device_id', device_id)

    def allocate(self, size, stream=None) -> Buffer:
        err, ptr = driver.cuMemAlloc(size)
        raise_if_driver_error(err)
        return Buffer._init(ptr, size, self)

    def deallocate(self, ptr, size, stream=None):
        if stream is None:
            stream = default_stream()
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

VirtualMemoryHandleTypeT = Literal["posix_fd", "generic", "none"]
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
    a = driver.CUmemAccess_flags
    _access_flags = {"rw": a.CU_MEM_ACCESS_FLAGS_PROT_READWRITE, "r": a.CU_MEM_ACCESS_FLAGS_PROT_READ, "none": 0}
    h = driver.CUmemAllocationHandleType
    _handle_types = {"none": h.CU_MEM_HANDLE_TYPE_NONE, "posix_fd": h.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, "win32": h.CU_MEM_HANDLE_TYPE_WIN32, "win32_kmt": h.CU_MEM_HANDLE_TYPE_WIN32_KMT, "fabric": h.CU_MEM_HANDLE_TYPE_FABRIC}
    g = driver.CUmemAllocationGranularity_flags
    _granularity = {"recommended": g.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED, "minimum": g.CU_MEM_ALLOC_GRANULARITY_MINIMUM}
    l = driver.CUmemLocationType
    _location_type = {"device": l.CU_MEM_LOCATION_TYPE_DEVICE, "host": l.CU_MEM_LOCATION_TYPE_HOST, "host_numa": l.CU_MEM_LOCATION_TYPE_HOST_NUMA, "host_numa_current": l.CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT}
    # CUDA 13+ exposes MANAGED in CUmemAllocationType; older 12.x does not
    a = driver.CUmemAllocationType
    _allocation_type = {"pinned": a.CU_MEM_ALLOCATION_TYPE_PINNED}
    ver_major, ver_minor = get_binding_version()
    if ver_major >= 13:
        _allocation_type["managed"] = a.CU_MEM_ALLOCATION_TYPE_MANAGED

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
                raise RuntimeError(f"Failed to extend VA range: {res}")
            _driver_call(driver.cuMemAddressFree, new_ptr, aligned_additional_size)
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
            trans.append(_driver_call, driver.cuMemAddressFree, new_ptr, aligned_additional_size)
            res, new_handle = driver.cuMemCreate(aligned_additional_size, prop, 0)
            raise_if_driver_error(res)
            # Register undo for creation
            trans.append(_driver_call, driver.cuMemRelease, new_handle)

            # Map the new physical memory to the extended VA range
            res, = driver.cuMemMap(new_ptr, aligned_additional_size, 0, new_handle, 0)
            raise_if_driver_error(res)
            # Register undo for mapping
            trans.append(_driver_call, driver.cuMemUnmap, new_ptr, aligned_additional_size)

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
            trans.append(_driver_call, driver.cuMemAddressFree, new_ptr, total_aligned_size)

            # Get the old allocation handle for remapping
            result, old_handle = driver.cuMemRetainAllocationHandle(buf.handle)
            raise_if_driver_error(result)
            # Register undo for old_handle
            trans.append(_driver_call, driver.cuMemRelease, old_handle)

            # Unmap the old VA range (aligned previous size)
            aligned_prev_size = total_aligned_size - aligned_additional_size
            result, = driver.cuMemUnmap(int(buf.handle), aligned_prev_size)
            raise_if_driver_error(result)

            def _remap_old():
                # Try to remap the old physical memory back to the original VA range
                try:
                    driver.cuMemMap(int(buf.handle), aligned_prev_size, 0, old_handle, 0)
                except Exception:
                    pass
            trans.append(_remap_old)

            # Remap the old physical memory to the new VA range (aligned previous size)
            res, = driver.cuMemMap(int(new_ptr), aligned_prev_size, 0, old_handle, 0)
            raise_if_driver_error(res)

            # Register undo for mapping
            trans.append(_driver_call, driver.cuMemUnmap, new_ptr, aligned_prev_size)

            # Create new physical memory for the additional size
            res, new_handle = driver.cuMemCreate(aligned_additional_size, prop, 0)
            raise_if_driver_error(res)

            # Register undo for new physical memory
            trans.append(_driver_call, driver.cuMemRelease, new_handle)

            # Map the new physical memory to the extended portion (aligned offset)
            res, = driver.cuMemMap(int(new_ptr) + aligned_prev_size, aligned_additional_size, 0, new_handle, 0)
            raise_if_driver_error(res)

            # Register undo for mapping
            trans.append(_driver_call, driver.cuMemUnmap, int(new_ptr) + aligned_prev_size, aligned_additional_size)

            # Set access permissions for the entire new range
            descs = self._build_access_descriptors(prop)
            if descs:
                res, = driver.cuMemSetAccess(new_ptr, total_aligned_size, descs, len(descs))
                raise_if_driver_error(res)

            # All succeeded, cancel undo actions
            trans.commit()

        # Free the old VA range (aligned previous size)
        handle_return(driver.cuMemAddressFree(int(buf.handle), aligned_prev_size))

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

        if prop.type != driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE:
            raise NotImplementedError(f"Location type must be CU_MEM_LOCATION_TYPE_DEVICE, got {config.location_type}")

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
            trans.append(_driver_call, driver.cuMemRelease, handle)

            # ---- Reserve VA space ----
            # Potentially, use a separate size for the VA reservation from the physical allocation size
            res, ptr = driver.cuMemAddressReserve(aligned_size, addr_align, config.addr_hint, 0)
            raise_if_driver_error(res)
            # Register undo for VA reservation
            trans.append(_driver_call, driver.cuMemAddressFree, ptr, aligned_size)

            # ---- Map physical memory into VA ----
            res, = driver.cuMemMap(ptr, aligned_size, 0, handle, 0)
            trans.append(_driver_call, driver.cuMemUnmap, ptr, aligned_size)
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
