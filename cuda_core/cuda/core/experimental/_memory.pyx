# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from libc.stdint cimport uintptr_t

from cuda.core.experimental._utils.cuda_utils cimport (
    _check_driver_error as raise_if_driver_error,
)

import abc
from typing import TypeVar, Union

from cuda.core.experimental._dlpack import DLDeviceType, make_py_capsule
from cuda.core.experimental._stream import Stream, default_stream
from cuda.core.experimental._utils.cuda_utils import driver

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


class DeviceMemoryResource(MemoryResource):
    """Create a device memory resource that uses the driver's stream-ordered memory pool.

    Parameters
    ----------
    device_id : int
        Device ordinal for which a memory resource is constructed. The mempool that is
        set to *current* on ``device_id`` is used. If no mempool is set to current yet,
        the driver would use the *default* mempool on the device.
    """

    __slots__ = ("_dev_id",)

    def __init__(self, device_id: int):
        err, self._handle = driver.cuDeviceGetMemPool(device_id)
        raise_if_driver_error(err)
        self._dev_id = device_id

        # Set a higher release threshold to improve performance when there are no active allocations.
        # By default, the release threshold is 0, which means memory is immediately released back
        # to the OS when there are no active suballocations, causing performance issues.
        # Check current release threshold
        err, current_threshold = driver.cuMemPoolGetAttribute(
            self._handle, driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD
        )
        raise_if_driver_error(err)
        # If threshold is 0 (default), set it to maximum to retain memory in the pool
        if int(current_threshold) == 0:
            err, = driver.cuMemPoolSetAttribute(
                self._handle,
                driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                driver.cuuint64_t(0xFFFFFFFFFFFFFFFF),
            )
            raise_if_driver_error(err)

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
        err, ptr = driver.cuMemAllocFromPoolAsync(size, self._handle, stream.handle)
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
    def is_device_accessible(self) -> bool:
        """bool: this memory resource provides device-accessible buffers."""
        return True

    @property
    def is_host_accessible(self) -> bool:
        """bool: this memory resource does not provides host-accessible buffers."""
        return False

    @property
    def device_id(self) -> int:
        """int: the associated device ordinal."""
        return self._dev_id


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

    def __init__(self, device_id):
        self._handle = None
        self._dev_id = device_id

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

@dataclass
class VMMConfig:
    """A configuration object for the VMMAllocatedMemoryResource
       Stores configuration information which tells the resource how to use the CUDA VMM APIs
    """
    """
    Configuration for CUDA VMM allocations.

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
    # TODO: for enums, do we re-expose them as cuda-core Enums or leave them as driver enums?
    allocation_type: driver.CUmemAllocationType
    location_type: driver.CUmemLocationType # Only supports CU_MEM_LOCATION_TYPE_DEVICE
    handle_type: driver.CUmemAllocationHandleType
    gpu_direct_rdma: bool = True
    granularity: driver.CUmemAllocationGranularity_flags
    addr_hint: Optional[int] = 0
    addr_align: Optional[int] = None
    peers: Iterable[int] = field(default_factory=tuple)
    self_access: str = "rw"   # 'rw' | 'r' | 'none'
    peer_access: str = "rw"   # 'rw' | 'r'

    @staticmethod
    def _access_to_flags(driver, spec: str) -> int:
        f = driver.CUmemAccess_flags
        if spec == "rw":
            return f.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        if spec == "r":
            return f.CU_MEM_ACCESS_FLAGS_PROT_READ
        if spec == "none":
            return 0
        raise ValueError(f"Unknown access spec: {spec!r}")
    

class VMMAllocatedMemoryResource(MemoryResource):
    """Create a device memory resource that uses the CUDA VMM APIs to allocate memory.

    Parameters
    ----------
    device_id : int
        Device ordinal for which a memory resource is constructed. The mempool that is
        set to *current* on ``device_id`` is used. If no mempool is set to current yet,
        the driver would use the *default* mempool on the device.
    
    config : VMMConfig
        A configuration object for the VMMAllocatedMemoryResource
    """
    def __init__(self, device, config: VMMConfig = None):
        self.device = device
        if config is None:
            config = VMMConfig(
                allocation_type=driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED,
                location_type=driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE,
                handle_type=driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                gpu_direct_rdma=True,
                granularity=driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                addr_hint=0,
                addr_align=None,
                peers=(),
                self_access="rw",
                peer_access="rw",
            )
        self.config = config

    def _align_up(self, size: int, gran: int) -> int:
        """
        Align a size up to the nearest multiple of a granularity.
        """
        return (size + gran - 1) & ~(gran - 1)

    def modify_allocation(self, buf: Buffer, new_size: int, config: VMMConfig = None) -> Buffer:
        """
        Grow an existing allocation using CUDA VMM, with a configurable policy.
        
        This implements true growing allocations that preserve the base pointer
        by extending the virtual address range and mapping additional physical memory.
        
        Parameters
        ----------
        buf : Buffer
            The existing buffer to grow
        new_size : int
            The new total size for the allocation
        config : VMMConfig, optional
            Configuration for the new physical memory chunks. If None, uses current config.
            
        Returns
        -------
        Buffer
            The same buffer with updated size, preserving the original pointer
        """
        if new_size <= buf.size:
            # No growth needed, return original buffer
            return buf
            
        if config is not None:
            self.config = config
            
        # Build allocation properties for new chunks
        prop = driver.CUmemAllocationProp()
        prop.type = self.config.allocation_type
        prop.location.type = self.config.location_type
        prop.location.id = self.device.device_id
        prop.allocFlags.gpuDirectRDMACapable = 1 if self.config.gpu_direct_rdma else 0
        prop.requestedHandleTypes = self.config.handle_type
        
        # Query granularity
        gran_flag = self.config.granularity
        res, gran = driver.cuMemGetAllocationGranularity(prop, gran_flag)
        if res != driver.CUresult.CUDA_SUCCESS:
            raise Exception(f"cuMemGetAllocationGranularity failed: {res}")
            
        # Calculate sizes
        additional_size = new_size - buf.size
        aligned_additional_size = self._align_up(additional_size, gran)
        total_aligned_size = self._align_up(new_size, gran)
        addr_align = self.config.addr_align or gran
        
        # Try to extend the existing VA range first
        res, new_ptr = driver.cuMemAddressReserve(
            aligned_additional_size, 
            addr_align, 
            buf.ptr + buf.size,  # fixedAddr hint - try to extend at end of current range
            0
        )
        
        if res != driver.CUresult.CUDA_SUCCESS or new_ptr != (buf.ptr + buf.size):
            # Fallback: couldn't extend contiguously, need full remapping
            return self._grow_allocation_slow_path(buf, new_size, prop, aligned_additional_size, total_aligned_size, addr_align)
        else:
            # Success! We can extend the VA range contiguously
            return self._grow_allocation_fast_path(buf, new_size, prop, aligned_additional_size, new_ptr)

    def _grow_allocation_fast_path(self, buf: Buffer, new_size: int, prop: driver.CUmemAllocationProp, 
                                   aligned_additional_size: int, new_ptr: int) -> Buffer:
        """
        Fast path: extend the VA range contiguously.
        
        This preserves the original pointer by mapping new physical memory
        to the extended portion of the virtual address range.
        """
        # Create new physical memory for the additional size
        res, new_handle = driver.cuMemCreate(aligned_additional_size, prop, 0)
        if res != driver.CUresult.CUDA_SUCCESS:
            driver.cuMemAddressFree(new_ptr, aligned_additional_size)
            raise Exception(f"cuMemCreate failed: {res}")
        
        # Map the new physical memory to the extended VA range
        res, = driver.cuMemMap(new_ptr, aligned_additional_size, 0, new_handle, 0)
        if res != driver.CUresult.CUDA_SUCCESS:
            driver.cuMemAddressFree(new_ptr, aligned_additional_size)
            driver.cuMemRelease(new_handle)
            raise Exception(f"cuMemMap failed: {res}")
        
        # Set access permissions for the new portion
        descs = self._build_access_descriptors(prop)
        if descs:
            res, = driver.cuMemSetAccess(new_ptr, aligned_additional_size, descs, len(descs))
            if res != driver.CUresult.CUDA_SUCCESS:
                driver.cuMemUnmap(new_ptr, aligned_additional_size)
                driver.cuMemAddressFree(new_ptr, aligned_additional_size)
                driver.cuMemRelease(new_handle)
                raise Exception(f"cuMemSetAccess failed: {res}")
        
        # Update the buffer size (pointer stays the same!)
        buf._size = new_size
        
        return buf

    def _grow_allocation_slow_path(self, buf: Buffer, new_size: int, prop: driver.CUmemAllocationProp,
                                   aligned_additional_size: int, total_aligned_size: int, addr_align: int) -> Buffer:
        """
        Slow path: full remapping when contiguous extension fails.
        
        This creates a new VA range and remaps both old and new physical memory.
        The buffer's pointer will change.
        """
        # Reserve a completely new, larger VA range
        res, new_ptr = driver.cuMemAddressReserve(total_aligned_size, addr_align, 0, 0)
        if res != driver.CUresult.CUDA_SUCCESS:
            raise Exception(f"cuMemAddressReserve failed: {res}")
        
        # Get the old allocation handle for remapping
        result, old_handle = driver.cuMemRetainAllocationHandle(buf.ptr)
        if result != driver.CUresult.CUDA_SUCCESS:
            driver.cuMemAddressFree(new_ptr, total_aligned_size)
            raise Exception(f"Failed to retain old allocation handle: {result}")
        
        # Unmap the old VA range
        result, = driver.cuMemUnmap(buf.ptr, buf.size)
        if result != driver.CUresult.CUDA_SUCCESS:
            driver.cuMemAddressFree(new_ptr, total_aligned_size)
            driver.cuMemRelease(old_handle)
            raise Exception(f"Failed to unmap old allocation: {result}")
        
        # Remap the old physical memory to the new VA range
        res, = driver.cuMemMap(new_ptr, buf.size, 0, old_handle, 0)
        if res != driver.CUresult.CUDA_SUCCESS:
            driver.cuMemAddressFree(new_ptr, total_aligned_size)
            driver.cuMemRelease(old_handle)
            raise Exception(f"cuMemMap failed for old memory: {res}")
        
        # Create new physical memory for the additional size
        res, new_handle = driver.cuMemCreate(aligned_additional_size, prop, 0)
        if res != driver.CUresult.CUDA_SUCCESS:
            driver.cuMemUnmap(new_ptr, total_aligned_size)
            driver.cuMemAddressFree(new_ptr, total_aligned_size)
            driver.cuMemRelease(old_handle)
            raise Exception(f"cuMemCreate failed for new memory: {res}")
        
        # Map the new physical memory to the extended portion
        res, = driver.cuMemMap(new_ptr + buf.size, aligned_additional_size, 0, new_handle, 0)
        if res != driver.CUresult.CUDA_SUCCESS:
            driver.cuMemUnmap(new_ptr, total_aligned_size)
            driver.cuMemAddressFree(new_ptr, total_aligned_size)
            driver.cuMemRelease(old_handle)
            driver.cuMemRelease(new_handle)
            raise Exception(f"cuMemMap failed for new memory: {res}")
        
        # Set access permissions for the entire new range
        descs = self._build_access_descriptors(prop)
        if descs:
            res, = driver.cuMemSetAccess(new_ptr, total_aligned_size, descs, len(descs))
            if res != driver.CUresult.CUDA_SUCCESS:
                driver.cuMemUnmap(new_ptr, total_aligned_size)
                driver.cuMemAddressFree(new_ptr, total_aligned_size)
                driver.cuMemRelease(old_handle)
                driver.cuMemRelease(new_handle)
                raise Exception(f"cuMemSetAccess failed: {res}")
        
        # Free the old VA range
        driver.cuMemAddressFree(buf.ptr, buf.size)
        
        # Update the buffer with new pointer and size
        buf._ptr = new_ptr
        buf._size = total_aligned_size
        buf._ptr_obj = new_ptr
        
        return buf

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
        owner_flags = VMMConfig._access_to_flags(driver, self.config.self_access)
        if owner_flags:
            d = driver.CUmemAccessDesc()
            d.location.type = prop.location.type
            d.location.id = prop.location.id
            d.flags = owner_flags
            descs.append(d)
        
        # Peer device access
        peer_flags = VMMConfig._access_to_flags(driver, self.config.peer_access)
        for peer_dev in self.config.peers:
            if peer_flags:
                d = driver.CUmemAccessDesc()
                d.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
                d.location.id = int(peer_dev)
                d.flags = peer_flags
                descs.append(d)
        
        return descs
        

    def allocate(self, size: int, stream: Stream = None) -> Buffer:
        """
        Allocate memory using CUDA VMM with a configurable policy.
        """
        config = self.config
        # ---- Build allocation properties ----
        prop = driver.CUmemAllocationProp()
        prop.type = config.allocation_type
        # TODO: Support host alloation if required
        if config.location_type != driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE:
            raise NotImplementedError(f"Location type must be CU_MEM_LOCATION_TYPE_DEVICE, got {config.location_type}")
        prop.location.type = config.location_type
        prop.location.id = self.device.device_id
        prop.allocFlags.gpuDirectRDMACapable = 1 if config.gpu_direct_rdma else 0
        prop.requestedHandleTypes = config.handle_type

        # ---- Query and apply granularity ----
        # Choose min vs recommended granularity per config
        gran_flag = config.granularity
        res, gran = driver.cuMemGetAllocationGranularity(prop, gran_flag)
        if res != driver.CUresult.CUDA_SUCCESS:
            raise Exception(f"cuMemGetAllocationGranularity failed: {res}")

        aligned_size = self._align_up(size, gran)
        addr_align = config.addr_align or gran

        # ---- Create physical memory ----
        res, handle = driver.cuMemCreate(aligned_size, prop, 0)
        if res != driver.CUresult.CUDA_SUCCESS:
            raise Exception(f"cuMemCreate failed: {res}")

        # ---- Reserve VA space ----
        # Potentially, use a separate size for the VA reservation from the physical allocation size
        res, ptr = driver.cuMemAddressReserve(aligned_size, addr_align, config.addr_hint, 0)
        if res != driver.CUresult.CUDA_SUCCESS:
            # tidy up physical handle on failure
            driver.cuMemRelease(handle)
            raise Exception(f"cuMemAddressReserve failed: {res}")

        # ---- Map physical memory into VA ----
        res, = driver.cuMemMap(ptr, aligned_size, 0, handle, 0)
        if res != driver.CUresult.CUDA_SUCCESS:
            driver.cuMemAddressFree(ptr, aligned_size)
            driver.cuMemRelease(handle)
            raise Exception(f"cuMemMap failed: {res}")

        # ---- Set access for owner + peers ----
        descs = []

        # Owner access
        owner_flags = VMMAllocationConfig._access_to_flags(driver, config.self_access)
        if owner_flags:
            d = driver.CUmemAccessDesc()
            d.location.type = prop.location.type
            d.location.id = prop.location.id
            d.flags = owner_flags
            descs.append(d)

        # Peer device access
        peer_flags = VMMAllocationConfig._access_to_flags(driver, config.peer_access)
        for peer_dev in config.peers:
            if peer_flags:
                d = driver.CUmemAccessDesc()
                d.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
                d.location.id = int(peer_dev)
                d.flags = peer_flags
                descs.append(d)

        if descs:
            res, = driver.cuMemSetAccess(ptr, aligned_size, descs, len(descs))
            if res != driver.CUresult.CUDA_SUCCESS:
                # Try to unwind on failure
                driver.cuMemUnmap(ptr, aligned_size)
                driver.cuMemAddressFree(ptr, aligned_size)
                driver.cuMemRelease(handle)
                raise Exception(f"cuMemSetAccess failed: {res}")

        # Done — return a Buffer that tracks this VA range
        buf = Buffer.from_handle(ptr=ptr, size=aligned_size, mr=self)
        return buf

    def deallocate(self, ptr: int, size: int, stream: Stream=None) -> None:
        """
        Deallocate memory on the device using CUDA VMM APIs.
        """
        result, handle = driver.cuMemRetainAllocationHandle(ptr)
        if result != driver.CUresult.CUDA_SUCCESS:
            raise Exception(f"Failed to retain allocation handle: {result}")
        result, = driver.cuMemUnmap(ptr, size)
        if result != driver.CUresult.CUDA_SUCCESS:
            raise Exception(f"Failed to unmap physical allocation: {result}")
        result, = driver.cuMemAddressFree(ptr, size)
        if result != driver.CUresult.CUDA_SUCCESS:
            raise Exception(f"Failed to free address: {result}")
        result, = driver.cuMemRelease(handle)
        if result != driver.CUresult.CUDA_SUCCESS:
            raise Exception(f"Failed to release physical allocation: {result}")


    @property
    def is_device_accessible(self) -> bool:
        """
        Indicates whether the allocated memory is accessible from the device.

        Returns:
            bool: Always True for NVSHMEM memory.
        """
        return True

    @property
    def is_host_accessible(self) -> bool:
        """
        Indicates whether the allocated memory is accessible from the host.

        Returns:
            bool: Always False for NVSHMEM memory.
        """
        return False

    @property
    def device_id(self) -> int:
        """
        Get the device ID associated with this memory resource.

        Returns:
            int: CUDA device ID.
        """
        return self.device.device_id

    def __repr__(self) -> str:
        """
        Return a string representation of the NvshmemResource.

        Returns:
            str: A string describing the object
        """
        return f"<VMMAllocatedMemoryResource device={self.device}>"
