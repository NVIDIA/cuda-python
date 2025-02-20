# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from __future__ import annotations

import abc
import platform
import weakref
from typing import Optional, Tuple, TypeVar

from cuda.core.experimental._dlpack import DLDeviceType, make_py_capsule
from cuda.core.experimental._stream import default_stream
from cuda.core.experimental._utils import driver, handle_return, runtime

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


class MemoryResource(abc.ABC):
    __slots__ = ("_handle",)

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


class SharedMempool(MemoryResource):
    """A memory pool that can be shared between processes on the same device.

    This class creates a CUDA memory pool that can be exported and imported
    across process boundaries, enabling efficient memory sharing between processes.

    Use the static methods `create` or `from_shared_handle` to instantiate a SharedMempool.
    """

    __slots__ = ("_dev_id", "_handle")

    def __init__(self, dev_id: int, handle: int) -> None:
        """Internal constructor. Use create() or from_shared_handle() instead."""
        self._dev_id = dev_id
        self._handle = handle

    @staticmethod
    def create(dev_id: int, max_size: int) -> SharedMempool:
        """Create a new memory pool.

        Parameters
        ----------
        dev_id : int
            The ID of the GPU device where the memory pool will be created
        max_size : int
            Maximum size in bytes that the memory pool can grow to

        Returns
        -------
        SharedMempool
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
        properties.handleTypes = _get_platform_handle_type()

        properties.location = driver.CUmemLocation()
        properties.location.id = dev_id
        properties.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        properties.maxSize = max_size
        properties.win32SecurityAttributes = 0
        properties.usage = 0

        handle = handle_return(driver.cuMemPoolCreate(properties))
        return SharedMempool(dev_id, handle)

    @staticmethod
    def from_shared_handle(dev_id: int, shared_handle: int) -> SharedMempool:
        """Create a SharedMempool from an existing handle.

        Parameters
        ----------
        dev_id : int
            The ID of the GPU device where the memory pool will be created
        shared_handle : int
            A platform-specific handle to import an existing memory pool

        Returns
        -------
        SharedMempool
            A memory pool instance connected to the existing pool

        Raises
        ------
        CUDAError
            If pool import fails
        """
        handle = handle_return(driver.cuMemPoolImportFromShareableHandle(shared_handle, _get_platform_handle_type(), 0))
        return SharedMempool(dev_id, handle)

    def get_shareable_handle(self) -> int:
        """Get a platform-specific handle that can be shared with other processes.

        Returns
        -------
        int
            A shareable handle that can be used to import this memory pool
            in another process
        """
        return handle_return(driver.cuMemPoolExportToShareableHandle(self._handle, _get_platform_handle_type(), 0))

    def allocate(self, size, stream=None) -> Buffer:
        if stream is None:
            stream = default_stream()
        ptr = handle_return(driver.cuMemAllocFromPoolAsync(size, self._handle, stream.handle))
        return Buffer(ptr, size, self)

    def deallocate(self, ptr, size, stream=None):
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

    def get_attribute(self, attr: driver.CUmemPool_attribute) -> int:
        """Get a memory pool attribute.

        Parameters
        ----------
        attr : CUmemPool_attribute
            The attribute to query. Supported attributes are:
            - CU_MEMPOOL_ATTR_RELEASE_THRESHOLD: Amount of reserved memory to hold before releasing
            - CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES: Allow reuse with event dependencies
            - CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC: Allow reuse without dependencies
            - CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES: Allow reuse with internal dependencies
            - CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT: Current reserved memory
            - CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH: High watermark of reserved memory
            - CU_MEMPOOL_ATTR_USED_MEM_CURRENT: Current used memory
            - CU_MEMPOOL_ATTR_USED_MEM_HIGH: High watermark of used memory

        Returns
        -------
        int
            The value of the requested attribute

        Raises
        ------
        CUDAError
            If the attribute query fails
        """
        return handle_return(driver.cuMemPoolGetAttribute(self._handle, attr))

    def export_pointer(self, ptr: int) -> bytes:
        """Export a pointer allocated from this pool for sharing between processes.

        This method constructs export data for sharing a specific allocation from
        this shared memory pool. The recipient process can import the allocation
        using import_pointer(). The data is not a handle and may be shared through
        any IPC mechanism.

        Parameters
        ----------
        ptr : int
            Pointer to memory being exported (must have been allocated from this pool)

        Returns
        -------
        bytes
            Export data that can be used to import the pointer in another process

        Raises
        ------
        CUDAError
            If the export operation fails
        """
        export_data = handle_return(driver.cuMemPoolExportPointer(ptr))
        return bytes(export_data)

    def import_pointer(self, share_data: bytes) -> Buffer:
        """Import a pointer that was exported from another process.

        The imported memory must not be accessed before the allocation operation
        completes in the exporting process. The imported memory must be freed from
        all importing processes before being freed in the exporting process.

        The pointer may be freed with cuMemFree or cuMemFreeAsync. If cuMemFreeAsync
        is used, the free must be completed on the importing process before the free
        operation on the exporting process.

        Parameters
        ----------
        share_data : bytes
            Export data obtained from export_pointer() in another process

        Returns
        -------
        Buffer
            A Buffer object wrapping the imported pointer

        Raises
        ------
        CUDAError
            If the import operation fails
        """
        # Convert bytes back to CUmemPoolPtrExportData
        export_data = driver.CUmemPoolPtrExportData.from_bytes(share_data)
        ptr = handle_return(driver.cuMemPoolImportPointer(self._handle, export_data))
        return Buffer(ptr, export_data.size, self)


class ShareableAllocator(MemoryResource):
    """Memory resource that creates allocations that can be shared between processes."""

    __slots__ = ("_dev_id",)

    def __init__(self, dev_id):
        self._dev_id = dev_id

    def get_shareable_allocation(self, size) -> tuple[Buffer, int]:
        """Create an allocation that can be shared between processes.

        Parameters
        ----------
        size : int
            Size of the allocation in bytes

        Returns
        -------
        tuple[Buffer, int]
            A tuple containing the Buffer object and a shareable handle that can be
            used to import this allocation in another process
        """
        # Create allocation properties for the device
        # Get minimum granularity for allocation
        prop = driver.CUmemAllocationProp()
        prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location = driver.CUmemLocation()
        prop.location.id = self._dev_id
        prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.requestedHandleTypes = _get_platform_handle_type()
        granularity = handle_return(
            driver.cuMemGetAllocationGranularity(
                prop, driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
            )
        )
        # Size must be a multiple of granularity
        if size % granularity != 0:
            raise ValueError(f"Size {size} is not a multiple of minimum allocation granularity {granularity}")

        # Create the allocation
        handle = handle_return(driver.cuMemCreate(size, prop, 0))

        # Export a shareable handle
        shareable_handle = handle_return(driver.cuMemExportToShareableHandle(handle, _get_platform_handle_type(), 0))

        # Reserve virtual address space
        ptr = handle_return(driver.cuMemAddressReserve(size, 0, 0, 0))

        # Map allocation to address space
        handle_return(driver.cuMemMap(ptr, size, 0, handle, 0))

        # Set access permissions
        access_desc = driver.CUmemAccessDesc()
        access_desc.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access_desc.location.id = self._dev_id
        access_desc.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        access_descs = [access_desc]
        handle_return(driver.cuMemSetAccess(ptr, size, access_descs, len(access_descs)))
        handle_return(runtime.cudaGetLastError())
        return Buffer(ptr, size, self), shareable_handle

    def import_shareable_allocation(self, size: int, shareable_handle: int) -> Buffer:
        """Import a shareable allocation from another process.

        Parameters
        ----------
        size : int
            Size of the allocation in bytes
        shareable_handle : int
            Handle obtained from get_shareable_allocation in another process

        Returns
        -------
        Buffer
            A Buffer object that can access the imported allocation
        """
        handle_return(runtime.cudaGetLastError())
        # Import the handle into a memory allocation
        handle = handle_return(driver.cuMemImportFromShareableHandle(shareable_handle, _get_platform_handle_type()))
        handle_return(runtime.cudaGetLastError())
        # Reserve virtual address space
        ptr = handle_return(driver.cuMemAddressReserve(size, 0, 0, 0))
        handle_return(runtime.cudaGetLastError())
        # Map allocation to address space
        handle_return(driver.cuMemMap(ptr, size, 0, handle, 0))

        # Set access permissions
        access_desc = driver.CUmemAccessDesc()
        access_desc.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access_desc.location.id = self._dev_id
        access_desc.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        access_descs = [access_desc]
        handle_return(driver.cuMemSetAccess(ptr, size, access_descs, len(access_descs)))

        return Buffer(ptr, size, self)

    def allocate(self, size, stream=None) -> Buffer:
        """Allocate memory that is accessible only from the device."""
        # Create allocation properties for the device
        prop = driver.CUmemAllocationProp()
        prop.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self._dev_id

        # Create the allocation
        handle = handle_return(driver.cuMemCreate(size, prop, 0))

        # Reserve virtual address space
        ptr = handle_return(driver.cuMemAddressReserve(size, 0, 0, 0))

        # Map allocation to address space
        handle_return(driver.cuMemMap(ptr, size, 0, handle, 0))

        # Set access permissions
        access_desc = driver.CUmemAccessDesc()
        access_desc.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access_desc.location.id = self._dev_id
        access_desc.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        access_descs = [access_desc]
        handle_return(driver.cuMemSetAccess(ptr, size, access_descs, len(access_descs)))

        return Buffer(ptr, size, self)

    def deallocate(self, ptr, size, stream=None):
        """Free allocated memory."""
        handle_return(driver.cuMemUnmap(ptr, size))
        handle_return(driver.cuMemAddressFree(ptr, size))

    @property
    def is_device_accessible(self) -> bool:
        """Whether memory from this allocator is accessible from device code."""
        return True

    @property
    def is_host_accessible(self) -> bool:
        """Whether memory from this allocator is accessible from host code."""
        return False

    @property
    def device_id(self) -> int:
        """The ID of the GPU device this allocator is associated with."""
        return self._dev_id


class _DefaultAsyncMempool(MemoryResource):
    __slots__ = ("_dev_id",)

    def __init__(self, dev_id):
        self._handle = handle_return(driver.cuDeviceGetMemPool(dev_id))
        self._dev_id = dev_id

    def allocate(self, size, stream=None) -> Buffer:
        if stream is None:
            stream = default_stream()
        ptr = handle_return(driver.cuMemAllocFromPoolAsync(size, self._handle, stream.handle))
        return Buffer(ptr, size, self)

    def deallocate(self, ptr, size, stream=None):
        if stream is None:
            stream = default_stream()
        handle_return(driver.cuMemFreeAsync(ptr, stream.handle))

    @property
    def is_device_accessible(self) -> bool:
        return True

    @property
    def is_host_accessible(self) -> bool:
        return False

    @property
    def device_id(self) -> int:
        return self._dev_id


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
