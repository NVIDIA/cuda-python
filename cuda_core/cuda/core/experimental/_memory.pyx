# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cuda.core.experimental._utils.cuda_utils cimport (
    _check_driver_error as raise_if_driver_error,
    check_or_create_options,
)

from cuda.core.experimental._dlpack import DLDeviceType, make_py_capsule
from cuda.core.experimental._stream import Stream, default_stream
from cuda.core.experimental._utils.cuda_utils import driver
from dataclasses import dataclass
from functools import wraps
from libc.stdint cimport uintptr_t
from typing import Tuple, TypeVar, Union, TYPE_CHECKING
import abc
import os
import platform

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
        self.close()

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
        if self._ptr and self._mr is not None:
            self._mr.deallocate(self._ptr, self._size, stream)
            self._ptr = 0
            self._mr = None
            self._ptr_obj = None

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

    __slots__ = ("_handle",)

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

def _requires_ipc(func):
    """Decorator to ensure IPC support is enabled before executing a method."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_ipc_enabled:
            raise RuntimeError("Memory resource is not IPC-enabled")
        return func(self, *args, **kwargs)
    return wrapper

def _def_mempool_attr_property(scope: dict, name: str, property_type: type, doc: str):
    """Define a property in the supplied scope for accessing a memory pool attribute.

    Args:
        name: The name of the property (e.g., 'reuse_follow_event_dependencies').
        doc: The docstring for the property.
        property_type: The return type of the property (e.g., bool, int).

    Returns:
        property: A property object that retrieves the attribute value.
    """
    attr_enum = getattr(driver.CUmemPool_attribute, f"CU_MEMPOOL_ATTR_{name.upper()}")
    def getter(self) -> property_type:
        err, value = driver.cuMemPoolGetAttribute(self.handle, attr_enum)
        raise_if_driver_error(err)
        return property_type(value)

    getter.__doc__ = doc
    scope[name] = property(getter)

class IPCBufferDescriptor:
    """Serializable object describing a buffer that can be shared between processes."""
    def __init__(self, reserved: bytes, size: int):
        self.reserved = reserved
        self._size = size

    def __reduce__(self):
        # This is subject to change if the CumemPoolPtrExportData struct/object changes.
        return (self._reconstruct, (self.reserved, self._size))

    @property
    def size(self):
        return self._size

    @classmethod
    def _reconstruct(cls, reserved, size):
        instance = cls(reserved, size)
        return instance

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
    ipc_enabled: bool = False
    max_size: int = 0

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
        device memory resource does not own the pool (`handle_is_owned` is
        `False`), and closing the resource has no effect.
    """
    __slots__ = "_dev_id", "_mempool_handle", "_ipc_handle_type", "_mempool_owned"

    def __init__(self, device_id: int | Device, options = None):
        device_id = getattr(device_id, 'device_id', device_id)
        opts = check_or_create_options(
            DeviceMemoryResourceOptions, options, "DeviceMemoryResource options", keep_none=True
        )

        if opts is None:
            # Get the current memory pool.
            self._dev_id = device_id
            self._mempool_handle = None
            self._ipc_handle_type = _NOIPC_HANDLE_TYPE # FIXME: need to query this?
            self._mempool_owned = False

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
            self._ipc_handle_type = properties.handleTypes
            self._mempool_owned = True

            err, self._mempool_handle = driver.cuMemPoolCreate(properties)
            raise_if_driver_error(err)

    def __del__(self):
        self.close()

    def close(self):
        """Close the device memory resource and destroy the associated memory pool if owned."""
        if self and self._mempool_owned:
            err, = driver.cuMemPoolDestroy(self._mempool_handle)
            raise_if_driver_error(err)
            self._dev_id = None
            self._mempool_handle = None
            self._ipc_handle_type = _NOIPC_HANDLE_TYPE
            self._mempool_owned = False

    def __bool__(self):
        """Check if the device memory resource is valid."""
        return self._mempool_handle is not None

    @classmethod
    def from_shared_handle(cls, device_id: int | Device, shared_handle: int) -> DeviceMemoryResource:
        """Create a device memory resource from a shared handle.

        Construct a new `DeviceMemoryResource` instance that imports a memory
        pool from a shareable handle. The memory pool is marked as owned, and
        the resource is associated with the specified `device_id`.

        Parameters
        ----------
        device_id : int | Device
            The ID of the device or a Device object for which the memory
            resource is created.

        shared_handle : int
            The shareable handle of the device memory resource to import.

        Returns
        -------
            A new device memory resource instance with the imported handle.
        """
        device_id = getattr(device_id, 'device_id', device_id)

        self = cls.__new__(cls)
        self._dev_id = device_id
        self._mempool_handle = None
        self._ipc_handle_type = _IPC_HANDLE_TYPE
        self._mempool_owned = True

        err, self._mempool_handle = driver.cuMemPoolImportFromShareableHandle(shared_handle, _IPC_HANDLE_TYPE, 0)
        raise_if_driver_error(err)

        return self

    @_requires_ipc
    def get_shareable_handle(self) -> int:
        """Export the memory pool handle to be shared (requires IPC).  The
        handle can be used to share the memory pool with other processes.

        Returns
        -------
            The shareable handle for the memory pool.
        """
        err, shared_handle = driver.cuMemPoolExportToShareableHandle(self._mempool_handle, _IPC_HANDLE_TYPE, 0)
        raise_if_driver_error(err)
        return shared_handle

    def close_shareable_handle(self, shared_handle) -> None:
        """Close a shareable handle for the memory pool."""
        assert self._ipc_handle_type == driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        os.close(shared_handle)

    @_requires_ipc
    def export_buffer(self, buffer: Buffer) -> IPCBufferDescriptor:
        """Export a buffer allocated from this pool for sharing between processes."""
        err, ptr = driver.cuMemPoolExportPointer(buffer.handle)
        raise_if_driver_error(err)
        return IPCBufferDescriptor(ptr.reserved, buffer.size)

    @_requires_ipc
    def import_buffer(self, ipc_buffer: IPCBufferDescriptor) -> Buffer:
        """Import a buffer that was exported from another process."""
        share_data = driver.CUmemPoolPtrExportData()
        share_data.reserved = ipc_buffer.reserved
        err, ptr = driver.cuMemPoolImportPointer(self._mempool_handle, share_data)
        raise_if_driver_error(err)
        return Buffer.from_handle(ptr, ipc_buffer.size, self)

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
    def device_id(self) -> int:
        """The associated device ordinal."""
        return self._dev_id

    @property
    def handle(self) -> cuda.bindings.driver.CUmemoryPool:
        """The memory resource handle."""
        return self._mempool_handle

    @property
    def handle_is_owned(self) -> bool:
        """Whether the memory resource handle is owned. If False, ``close`` has no effect."""
        return self._mempool_owned

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

    # Define additional properties corresponding to mempool attributes.
    _def_mempool_attr_property(locals(), "reuse_follow_event_dependencies", bool, "Allow memory to be reused when there are event dependencies between streams.")
    _def_mempool_attr_property(locals(), "reuse_allow_opportunistic", bool, "Allow reuse of completed frees without dependencies.")
    _def_mempool_attr_property(locals(), "reuse_allow_internal_dependencies", bool, "Allow insertion of new stream dependencies for memory reuse.")
    _def_mempool_attr_property(locals(), "release_threshold", int, "Amount of reserved memory to hold before OS release.")
    _def_mempool_attr_property(locals(), "reserved_mem_current", int, "Current amount of backing memory allocated.")
    _def_mempool_attr_property(locals(), "reserved_mem_high", int, "High watermark of backing memory allocated.")
    _def_mempool_attr_property(locals(), "used_mem_current", int, "Current amount of memory in use.")
    _def_mempool_attr_property(locals(), "used_mem_high", int, "High watermark of memory in use.")

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

del _requires_ipc
del _def_mempool_attr_property
