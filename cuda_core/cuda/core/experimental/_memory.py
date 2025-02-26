# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from __future__ import annotations

import abc

# Register cleanup function to be called at interpreter shutdown
import atexit

# Add ctypes import for Windows security attributes
import ctypes
import platform
import weakref
from typing import Optional, Tuple, TypeVar

from cuda.core.experimental._dlpack import DLDeviceType, make_py_capsule
from cuda.core.experimental._stream import default_stream
from cuda.core.experimental._utils import driver, handle_return

# Check if pywin32 is available on Windows
_PYWIN32_AVAILABLE = False
if platform.system() == "Windows":
    try:
        import win32security

        _PYWIN32_AVAILABLE = True
    except ImportError:
        import warnings

        warnings.warn(
            "pywin32 module not found. For better IPC support on Windows, " "install it with: pip install pywin32",
            stacklevel=2,
        )

PyCapsule = TypeVar("PyCapsule")


# TODO: define a memory property mixin class and make Buffer and
# MemoryResource both inherit from it


class Buffer:
    """Represent a handle to allocated memory.

    This generic object provides a unified representation for how
    different memory resources are to give access to their memory
    allocations.

    Support for data interchange mechanisms are provided by
    establishing both the DLPack and the Python-level buffer
    protocols.

    Parameters
    ----------
    ptr : Any
        Allocated buffer handle object
    size : Any
        Memory size of the buffer
    mr : :obj:`~_memory.MemoryResource`, optional
        Memory resource associated with the buffer

    """

    class _MembersNeededForFinalize:
        __slots__ = ("ptr", "size", "mr")

        def __init__(self, buffer_obj, ptr, size, mr):
            self.ptr = ptr
            self.size = size
            self.mr = mr
            weakref.finalize(buffer_obj, self.close)

        def close(self, stream=None):
            if self.ptr and self.mr is not None:
                if stream is None:
                    stream = default_stream()
                self.mr.deallocate(self.ptr, self.size, stream)
                self.ptr = 0
                self.mr = None

    # TODO: handle ownership? (_mr could be None)
    __slots__ = ("__weakref__", "_mnff")

    def __init__(self, ptr, size, mr: MemoryResource = None):
        self._mnff = Buffer._MembersNeededForFinalize(self, ptr, size, mr)

    def close(self, stream=None):
        """Deallocate this buffer asynchronously on the given stream.

        This buffer is released back to their memory resource
        asynchronously on the given stream.

        Parameters
        ----------
        stream : Any, optional
            The stream object with a __cuda_stream__ protocol to
            use for asynchronous deallocation. Defaults to using
            the default stream.

        """
        if self._mnff.mr is None:
            raise RuntimeError(
                "Cannot close a buffer that was not allocated from a memory resource, this buffer is: ",
                self,
            )
        self._mnff.close(stream)

    @property
    def handle(self):
        """Return the buffer handle object."""
        return self._mnff.ptr

    @property
    def size(self):
        """Return the memory size of this buffer."""
        return self._mnff.size

    @property
    def memory_resource(self) -> MemoryResource:
        """Return the memory resource associated with this buffer."""
        return self._mnff.mr

    @property
    def is_device_accessible(self) -> bool:
        """Return True if this buffer can be accessed by the GPU, otherwise False."""
        if self._mnff.mr is not None:
            return self._mnff.mr.is_device_accessible
        raise NotImplementedError

    @property
    def is_host_accessible(self) -> bool:
        """Return True if this buffer can be accessed by the CPU, otherwise False."""
        if self._mnff.mr is not None:
            return self._mnff.mr.is_host_accessible
        raise NotImplementedError

    @property
    def device_id(self) -> int:
        """Return the device ordinal of this buffer."""
        if self._mnff.mr is not None:
            return self._mnff.mr.device_id
        raise NotImplementedError

    def copy_to(self, dst: Buffer = None, *, stream) -> Buffer:
        """Copy from this buffer to the dst buffer asynchronously on the given stream.

        Copies the data from this buffer to the provided dst buffer.
        If the dst buffer is not provided, then a new buffer is first
        allocated using the associated memory resource before the copy.

        Parameters
        ----------
        dst : :obj:`~_memory.Buffer`
            Source buffer to copy data from
        stream : Any
            Keyword argument specifying the stream for the
            asynchronous copy

        """
        if stream is None:
            raise ValueError("stream must be provided")
        if dst is None:
            if self._mnff.mr is None:
                raise ValueError("a destination buffer must be provided")
            dst = self._mnff.mr.allocate(self._mnff.size, stream)
        if dst._mnff.size != self._mnff.size:
            raise ValueError("buffer sizes mismatch between src and dst")
        handle_return(driver.cuMemcpyAsync(dst._mnff.ptr, self._mnff.ptr, self._mnff.size, stream.handle))
        return dst

    def copy_from(self, src: Buffer, *, stream):
        """Copy from the src buffer to this buffer asynchronously on the given stream.

        Parameters
        ----------
        src : :obj:`~_memory.Buffer`
            Source buffer to copy data from
        stream : Any
            Keyword argument specifying the stream for the
            asynchronous copy

        """
        if stream is None:
            raise ValueError("stream must be provided")
        if src._mnff.size != self._mnff.size:
            raise ValueError("buffer sizes mismatch between src and dst")
        handle_return(driver.cuMemcpyAsync(self._mnff.ptr, src._mnff.ptr, self._mnff.size, stream.handle))

    def __dlpack__(
        self,
        *,
        stream: Optional[int] = None,
        max_version: Optional[Tuple[int, int]] = None,
        dl_device: Optional[Tuple[int, int]] = None,
        copy: Optional[bool] = None,
    ) -> PyCapsule:
        # Note: we ignore the stream argument entirely (as if it is -1).
        # It is the user's responsibility to maintain stream order.
        if dl_device is not None or copy is True:
            raise BufferError
        if max_version is None:
            versioned = False
        else:
            assert len(max_version) == 2
            versioned = max_version >= (1, 0)
        capsule = make_py_capsule(self, versioned)
        return capsule

    def __dlpack_device__(self) -> Tuple[int, int]:
        if self.is_device_accessible and not self.is_host_accessible:
            return (DLDeviceType.kDLCUDA, self.device_id)
        elif self.is_device_accessible and self.is_host_accessible:
            # TODO: this can also be kDLCUDAManaged, we need more fine-grained checks
            return (DLDeviceType.kDLCUDAHost, 0)
        elif not self.is_device_accessible and self.is_host_accessible:
            return (DLDeviceType.kDLCPU, 0)
        else:  # not self.is_device_accessible and not self.is_host_accessible
            raise BufferError("invalid buffer")

    def __buffer__(self, flags: int, /) -> memoryview:
        # Support for Python-level buffer protocol as per PEP 688.
        # This raises a BufferError unless:
        #   1. Python is 3.12+
        #   2. This Buffer object is host accessible
        raise NotImplementedError("TODO")

    def __release_buffer__(self, buffer: memoryview, /):
        # Supporting method paired with __buffer__.
        raise NotImplementedError("TODO")


class IPCBufferDescriptor:
    """Buffer class to represent a buffer description which can be shared across processes.
    It is not a valid buffer containing data, but rather a description used by the importing
    process to construct a valid buffer. It's primary use is to provide a serialization
    mechanism for passing exported buffers between processes."""

    def __init__(self, reserved: bytes, size: int):
        self.reserved = reserved
        self._size = size

    def __reduce__(self):
        # This is subject to change if the CumemPoolPtrExportData struct/object changes.
        return (self._reconstruct, (self.reserved, self._size))

    @classmethod
    def _reconstruct(cls, reserved, size):
        instance = cls(reserved, size)
        return instance


class MemoryResource(abc.ABC):
    """Base class for memory resources.

    This class provides an abstract interface for memory resources.
    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs): ...

    @abc.abstractmethod
    def allocate(self, size, stream=None) -> Buffer: ...

    @abc.abstractmethod
    def deallocate(self, ptr, size, stream=None): ...

    @property
    @abc.abstractmethod
    def is_device_accessible(self) -> bool:
        # Check if the buffers allocated from this MR can be accessed from
        # GPUs.
        ...

    @property
    @abc.abstractmethod
    def is_host_accessible(self) -> bool:
        # Check if the buffers allocated from this MR can be accessed from
        # CPUs.
        ...

    @property
    @abc.abstractmethod
    def device_id(self) -> int:
        # Return the device ID if this MR is for single devices. Raise an
        # exception if it is not.
        ...


def _get_platform_handle_type() -> int:
    """Returns the appropriate handle type for the current platform."""
    system = platform.system()
    if system == "Linux":
        return driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    elif system == "Windows":
        return driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_WIN32
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def _create_win32_security_attributes():
    """Creates a Windows SECURITY_ATTRIBUTES structure with default settings.

    The security descriptor is configured to allow access across processes,
    which is appropriate for shared memory.

    Returns:
        A pointer to a SECURITY_ATTRIBUTES structure or None if not on Windows.
    """
    if platform.system() != "Windows":
        return None

    # Define the Windows SECURITY_ATTRIBUTES structure
    class SECURITY_ATTRIBUTES(ctypes.Structure):
        _fields_ = [
            ("nLength", ctypes.c_ulong),
            ("lpSecurityDescriptor", ctypes.c_void_p),
            ("bInheritHandle", ctypes.c_int),
        ]

    if _PYWIN32_AVAILABLE:
        # Create a security descriptor using pywin32
        sd = win32security.SECURITY_DESCRIPTOR()

        # Create a blank DACL (this allows all access)
        dacl = win32security.ACL()

        # Set the DACL to the security descriptor
        sd.SetSecurityDescriptorDacl(1, dacl, 0)

        # Create and initialize the security attributes structure
        sa = SECURITY_ATTRIBUTES()
        sa.nLength = ctypes.sizeof(SECURITY_ATTRIBUTES)
        sa.lpSecurityDescriptor = ctypes.c_void_p(int(sd.SECURITY_DESCRIPTOR))
        sa.bInheritHandle = False

        # Store the security descriptor to prevent garbage collection
        if not hasattr(_create_win32_security_attributes, "_security_descriptors"):
            _create_win32_security_attributes._security_descriptors = []
        _create_win32_security_attributes._security_descriptors.append(sd)

        return ctypes.addressof(sa)
    else:
        # If pywin32 is not available, use a NULL security descriptor
        # This is less secure but should work for testing
        try:
            sa = SECURITY_ATTRIBUTES()
            sa.nLength = ctypes.sizeof(SECURITY_ATTRIBUTES)
            sa.lpSecurityDescriptor = ctypes.c_void_p(0)  # NULL security descriptor
            sa.bInheritHandle = False

            return ctypes.addressof(sa)

        except Exception as e:
            print(f"Warning: Failed to create security attributes: {e}")
            return 0  # Return 0 as a fallback


# Add cleanup function for security descriptors
def _cleanup_security_descriptors():
    """Free any allocated security descriptors when the module is unloaded."""
    if hasattr(_create_win32_security_attributes, "_security_descriptors"):
        # The security descriptors are now pywin32 objects that will be garbage collected
        # or simple ctypes structures, so we just need to clear the list
        _create_win32_security_attributes._security_descriptors.clear()


atexit.register(_cleanup_security_descriptors)


class AsyncMempool(MemoryResource):
    """A CUDA memory pool for efficient memory allocation.

    This class creates a CUDA memory pool that provides better allocation and
    deallocation performance compared to individual allocations. The pool can
    optionally be configured to support sharing across process boundaries.

    Use the static methods create() or from_shared_handle() to instantiate.
    Direct instantiation is not supported.

    Notes
    -----
    The _from_device() method is for internal use by the Device class only and
    should not be called directly by users.
    """

    class _MembersNeededForFinalize:
        __slots__ = ("handle", "need_close")

        def __init__(self, mr_obj, handle, need_close):
            self.handle = handle
            self.need_close = need_close
            weakref.finalize(mr_obj, self.close)

        def close(self):
            if self.handle and self.need_close:
                handle_return(driver.cuMemPoolDestroy(self.handle))
                self.handle = None
                self.need_close = False

    __slots__ = ("_mnff", "_dev_id", "_ipc_enabled")

    def __init__(self):
        """Direct instantiation is not supported.

        Use the static methods create() or from_shared_handle() instead.
        """
        raise NotImplementedError(
            "directly creating an AsyncMempool object is not supported. Please use either "
            "AsyncMempool.create() or from_shared_handle()"
        )

    @staticmethod
    def _init(dev_id: int, handle: int, ipc_enabled: bool = False, need_close: bool = False) -> AsyncMempool:
        """Internal constructor for AsyncMempool objects.

        Parameters
        ----------
        dev_id : int
            The ID of the GPU device where the memory pool will be created
        handle : int
            The handle to the CUDA memory pool
        ipc_enabled : bool
            Whether the pool supports inter-process sharing capabilities

        Returns
        -------
        AsyncMempool
            A new memory pool instance
        """
        self = AsyncMempool.__new__(AsyncMempool)
        self._dev_id = dev_id
        self._ipc_enabled = ipc_enabled
        self._mnff = AsyncMempool._MembersNeededForFinalize(self, handle, need_close)
        return self

    @staticmethod
    def _from_device(dev_id: int) -> AsyncMempool:
        """Internal method to create an AsyncMempool for a device's default memory pool.

        This method is intended for internal use by the Device class only.
        Users should not call this method directly.

        Parameters
        ----------
        dev_id : int
            The ID of the GPU device to get the default memory pool from

        Returns
        -------
        AsyncMempool
            A memory pool instance connected to the device's default pool
        """
        handle = handle_return(driver.cuDeviceGetMemPool(dev_id))
        return AsyncMempool._init(dev_id, handle, ipc_enabled=False, need_close=False)

    @staticmethod
    def create(
        dev_id: int, max_size: int, ipc_enabled: bool = False, win32_security_attributes: int = 0
    ) -> AsyncMempool:
        """Create a new memory pool.

        Parameters
        ----------
        dev_id : int
            The ID of the GPU device where the memory pool will be created
        max_size : int
            Maximum size in bytes that the memory pool can grow to
        ipc_enabled : bool, optional
            Whether to enable inter-process sharing capabilities. Default is False.
        win32_security_attributes : int, optional
            Custom Windows security attributes pointer. If 0 (default), a default security
            attributes structure will be created when needed on Windows platforms.

        Returns
        -------
        AsyncMempool
            A new memory pool instance

        Raises
        ------
        ValueError
            If max_size is None
        CUDAError
            If pool creation fails
        """
        if max_size is None:
            raise ValueError("max_size must be provided when creating a new memory pool")

        properties = driver.CUmemPoolProps()
        properties.allocType = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        properties.handleTypes = (
            _get_platform_handle_type() if ipc_enabled else driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
        )
        properties.location = driver.CUmemLocation()
        properties.location.id = dev_id
        properties.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        properties.maxSize = max_size

        # Set up Windows security attributes if needed
        if platform.system() == "Windows" and ipc_enabled:
            if win32_security_attributes == 0:
                # Create default security attributes if none provided
                win32_security_attributes = _create_win32_security_attributes()
            properties.win32SecurityAttributes = win32_security_attributes
        else:
            properties.win32SecurityAttributes = 0

        properties.usage = 0

        handle = handle_return(driver.cuMemPoolCreate(properties))
        return AsyncMempool._init(dev_id, handle, ipc_enabled=ipc_enabled, need_close=True)

    @staticmethod
    def from_shared_handle(dev_id: int, shared_handle: int) -> AsyncMempool:
        """Create an AsyncMempool from an existing handle.

        Parameters
        ----------
        dev_id : int
            The ID of the GPU device where the memory pool will be created
        shared_handle : int
            A platform-specific handle to import an existing memory pool

        Returns
        -------
        AsyncMempool
            A memory pool instance connected to the existing pool
        """
        handle = handle_return(driver.cuMemPoolImportFromShareableHandle(shared_handle, _get_platform_handle_type(), 0))
        return AsyncMempool._init(
            dev_id, handle, ipc_enabled=True, need_close=True
        )  # Imported pools are always IPC-enabled

    def get_shareable_handle(self) -> int:
        """Get a platform-specific handle that can be shared with other processes."""
        if not self._ipc_enabled:
            raise RuntimeError("This memory pool was not created with IPC support enabled")
        return handle_return(driver.cuMemPoolExportToShareableHandle(self._mnff.handle, _get_platform_handle_type(), 0))

    def export_buffer(self, buffer: Buffer) -> IPCBufferDescriptor:
        """Export a buffer allocated from this pool for sharing between processes."""
        if not self._ipc_enabled:
            raise RuntimeError("This memory pool was not created with IPC support enabled")
        return IPCBufferDescriptor(
            handle_return(driver.cuMemPoolExportPointer(buffer.handle)).reserved, buffer._mnff.size
        )

    def import_buffer(self, ipc_buffer: IPCBufferDescriptor) -> Buffer:
        """Import a buffer that was exported from another process."""
        if not self._ipc_enabled:
            raise RuntimeError("This memory pool was not created with IPC support enabled")
        share_data = driver.CUmemPoolPtrExportData()
        share_data.reserved = ipc_buffer.reserved
        return Buffer(
            handle_return(driver.cuMemPoolImportPointer(self._mnff.handle, share_data)), ipc_buffer._size, self
        )

    def allocate(self, size: int, stream=None) -> Buffer:
        """Allocate memory from the pool."""
        if stream is None:
            stream = default_stream()
        ptr = handle_return(driver.cuMemAllocFromPoolAsync(size, self._mnff.handle, stream.handle))
        return Buffer(ptr, size, self)

    def deallocate(self, ptr: int, size: int, stream=None) -> None:
        """Deallocate memory back to the pool."""
        if stream is None:
            stream = default_stream()
        handle_return(driver.cuMemFreeAsync(ptr, stream.handle))

    @property
    def is_device_accessible(self) -> bool:
        """Whether memory from this pool is accessible from device code."""
        return True

    @property
    def is_host_accessible(self) -> bool:
        """Whether memory from this pool is accessible from host code."""
        return False

    @property
    def device_id(self) -> int:
        """The ID of the GPU device this memory pool is associated with."""
        return self._dev_id

    @property
    def reuse_follow_event_dependencies(self) -> bool:
        """Allow memory to be reused when there are event dependencies between streams."""
        return bool(
            handle_return(
                driver.cuMemPoolGetAttribute(
                    self._mnff.handle, driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES
                )
            )
        )

    @property
    def reuse_allow_opportunistic(self) -> bool:
        """Allow reuse of completed frees without dependencies."""
        return bool(
            handle_return(
                driver.cuMemPoolGetAttribute(
                    self._mnff.handle, driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC
                )
            )
        )

    @property
    def reuse_allow_internal_dependencies(self) -> bool:
        """Allow insertion of new stream dependencies for memory reuse."""
        return bool(
            handle_return(
                driver.cuMemPoolGetAttribute(
                    self._mnff.handle, driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES
                )
            )
        )

    @property
    def release_threshold(self) -> int:
        """Amount of reserved memory to hold before OS release."""
        return int(
            handle_return(
                driver.cuMemPoolGetAttribute(
                    self._mnff.handle, driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD
                )
            )
        )

    @property
    def reserved_mem_current(self) -> int:
        """Current amount of backing memory allocated."""
        return int(
            handle_return(
                driver.cuMemPoolGetAttribute(
                    self._mnff.handle, driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT
                )
            )
        )

    @property
    def reserved_mem_high(self) -> int:
        """High watermark of backing memory allocated."""
        return int(
            handle_return(
                driver.cuMemPoolGetAttribute(
                    self._mnff.handle, driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH
                )
            )
        )

    @property
    def used_mem_current(self) -> int:
        """Current amount of memory in use."""
        return int(
            handle_return(
                driver.cuMemPoolGetAttribute(
                    self._mnff.handle, driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_USED_MEM_CURRENT
                )
            )
        )

    @property
    def used_mem_high(self) -> int:
        """High watermark of memory in use."""
        return int(
            handle_return(
                driver.cuMemPoolGetAttribute(
                    self._mnff.handle, driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_USED_MEM_HIGH
                )
            )
        )


class _DefaultPinnedMemorySource(MemoryResource):
    def __init__(self):
        # TODO: support flags from cuMemHostAlloc?
        self._handle = None

    def allocate(self, size, stream=None) -> Buffer:
        ptr = handle_return(driver.cuMemAllocHost(size))
        return Buffer(ptr, size, self)

    def deallocate(self, ptr, size, stream=None):
        handle_return(driver.cuMemFreeHost(ptr))

    @property
    def is_device_accessible(self) -> bool:
        return True

    @property
    def is_host_accessible(self) -> bool:
        return True

    @property
    def device_id(self) -> int:
        raise RuntimeError("the pinned memory resource is not bound to any GPU")


class _SynchronousMemoryResource(MemoryResource):
    __slots__ = ("_dev_id",)

    def __init__(self, dev_id):
        self._handle = None
        self._dev_id = dev_id

    def allocate(self, size, stream=None) -> Buffer:
        ptr = handle_return(driver.cuMemAlloc(size))
        return Buffer(ptr, size, self)

    def deallocate(self, ptr, size, stream=None):
        if stream is None:
            stream = default_stream()
        stream.sync()
        handle_return(driver.cuMemFree(ptr))

    @property
    def is_device_accessible(self) -> bool:
        return True

    @property
    def is_host_accessible(self) -> bool:
        return False

    @property
    def device_id(self) -> int:
        return self._dev_id
