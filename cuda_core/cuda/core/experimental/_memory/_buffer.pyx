# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

cimport cython
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uintptr_t
from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, Py_buffer, PyBUF_SIMPLE

from cuda.bindings cimport cydriver
from cuda.core.experimental._memory._device_memory_resource import DeviceMemoryResource
from cuda.core.experimental._memory._pinned_memory_resource import PinnedMemoryResource
from cuda.core.experimental._memory._ipc cimport IPCBufferDescriptor, IPCDataForBuffer
from cuda.core.experimental._memory cimport _ipc
from cuda.core.experimental._resource_handles cimport (
    DevicePtrHandle,
    StreamHandle,
    deviceptr_create_with_owner,
    intptr,
    native,
    set_deallocation_stream,
)
from cuda.core.experimental._stream cimport Stream_accept, Stream
from cuda.core.experimental._utils.cuda_utils cimport (
    _check_driver_error as raise_if_driver_error,
    HANDLE_RETURN,
)

import abc
import sys
from typing import TypeVar, Union

if sys.version_info >= (3, 12):
    from collections.abc import Buffer as BufferProtocol
else:
    BufferProtocol = object

from cuda.core.experimental._dlpack import DLDeviceType, make_py_capsule
from cuda.core.experimental._utils.cuda_utils import driver
from cuda.core.experimental._device import Device

__all__ = ['Buffer', 'MemoryResource']


DevicePointerT = Union[driver.CUdeviceptr, int, None]
"""
A type union of :obj:`~driver.CUdeviceptr`, `int` and `None` for hinting
:attr:`Buffer.handle`.
"""

cdef class Buffer:
    """Represent a handle to allocated memory.

    This generic object provides a unified representation for how
    different memory resources are to give access to their memory
    allocations.

    Support for data interchange mechanisms are provided by DLPack.
    """
    def __cinit__(self):
        self._clear()

    def _clear(self):
        # _h_ptr is default-initialized (empty shared_ptr) by C++
        self._size = 0
        self._memory_resource = None
        self._ipc_data = None
        self._owner = None
        self._mem_attrs_inited = False

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Buffer objects cannot be instantiated directly. "
                           "Please use MemoryResource APIs.")

    @classmethod
    def _init(
        cls, ptr: DevicePointerT, size_t size, mr: MemoryResource | None = None,
        stream: Stream | None = None, ipc_descriptor: IPCBufferDescriptor | None = None,
        owner : object | None = None
    ):
        """Legacy init for compatibility - creates a non-owning ref handle.

        Note: The stream parameter is accepted for API compatibility but is
        ignored since non-owning refs are never freed by the handle.
        """
        if mr is not None and owner is not None:
            raise ValueError("owner and memory resource cannot be both specified together")
        cdef Buffer self = Buffer.__new__(cls)
        self._h_ptr = deviceptr_create_with_owner(<uintptr_t>(int(ptr)), owner)
        self._size = size
        self._memory_resource = mr
        self._ipc_data = IPCDataForBuffer(ipc_descriptor, True) if ipc_descriptor is not None else None
        self._owner = owner
        self._mem_attrs_inited = False
        return self

    # No __dealloc__ needed - RAII handles cleanup via _h_ptr destructor

    def __reduce__(self):
        # Must not serialize the parent's stream!
        return Buffer.from_ipc_descriptor, (self.memory_resource, self.get_ipc_descriptor())

    @staticmethod
    def from_handle(
        ptr: DevicePointerT, size_t size, mr: MemoryResource | None = None,
        owner: object | None = None,
    ) -> Buffer:
        """Create a new :class:`Buffer` object from a pointer.

        Parameters
        ----------
        ptr : :obj:`~_memory.DevicePointerT`
            Allocated buffer handle object
        size : int
            Memory size of the buffer
        mr : :obj:`~_memory.MemoryResource`, optional
            Memory resource associated with the buffer
        owner : object, optional
            An object holding external allocation that the ``ptr`` points to.
            The reference is kept as long as the buffer is alive.
            The ``owner`` and ``mr`` cannot be specified together.

        Note
        ----
        This creates a non-owning reference. The pointer will NOT be freed
        when the Buffer is closed or garbage collected.
        """
        return Buffer._init(ptr, size, mr=mr, owner=owner)

    @classmethod
    def from_ipc_descriptor(
        cls, mr: DeviceMemoryResource | PinnedMemoryResource, ipc_descriptor: IPCBufferDescriptor,
        stream: Stream = None
    ) -> Buffer:
        """Import a buffer that was exported from another process."""
        return _ipc.Buffer_from_ipc_descriptor(cls, mr, ipc_descriptor, stream)

    def get_ipc_descriptor(self) -> IPCBufferDescriptor:
        """Export a buffer allocated for sharing between processes."""
        if self._ipc_data is None:
            self._ipc_data = IPCDataForBuffer(_ipc.Buffer_get_ipc_descriptor(self), False)
        return self._ipc_data.ipc_descriptor

    def close(self, stream: Stream | GraphBuilder | None = None):
        """Deallocate this buffer asynchronously on the given stream.

        This buffer is released back to their memory resource
        asynchronously on the given stream.

        Parameters
        ----------
        stream : :obj:`~_stream.Stream` | :obj:`~_graph.GraphBuilder`, optional
            The stream object to use for asynchronous deallocation. If None,
            the deallocation stream stored in the handle is used.
        """
        Buffer_close(self, stream)

    def copy_to(self, dst: Buffer = None, *, stream: Stream | GraphBuilder) -> Buffer:
        """Copy from this buffer to the dst buffer asynchronously on the given stream.

        Copies the data from this buffer to the provided dst buffer.
        If the dst buffer is not provided, then a new buffer is first
        allocated using the associated memory resource before the copy.

        Parameters
        ----------
        dst : :obj:`~_memory.Buffer`
            Source buffer to copy data from
        stream : :obj:`~_stream.Stream` | :obj:`~_graph.GraphBuilder`
            Keyword argument specifying the stream for the
            asynchronous copy

        """
        stream = Stream_accept(stream)
        cdef size_t src_size = self._size

        if dst is None:
            if self._memory_resource is None:
                raise ValueError("a destination buffer must be provided (this "
                                 "buffer does not have a memory_resource)")
            dst = self._memory_resource.allocate(src_size, stream)

        cdef size_t dst_size = dst._size
        if dst_size != src_size:
            raise ValueError( "buffer sizes mismatch between src and dst (sizes "
                             f"are: src={src_size}, dst={dst_size})"
            )
        err, = driver.cuMemcpyAsync(native(dst._h_ptr), native(self._h_ptr), src_size, stream.handle)
        raise_if_driver_error(err)
        return dst

    def copy_from(self, src: Buffer, *, stream: Stream | GraphBuilder):
        """Copy from the src buffer to this buffer asynchronously on the given stream.

        Parameters
        ----------
        src : :obj:`~_memory.Buffer`
            Source buffer to copy data from
        stream : :obj:`~_stream.Stream` | :obj:`~_graph.GraphBuilder`
            Keyword argument specifying the stream for the
            asynchronous copy

        """
        stream = Stream_accept(stream)
        cdef size_t dst_size = self._size
        cdef size_t src_size = src._size

        if src_size != dst_size:
            raise ValueError( "buffer sizes mismatch between src and dst (sizes "
                             f"are: src={src_size}, dst={dst_size})"
            )
        err, = driver.cuMemcpyAsync(native(self._h_ptr), native(src._h_ptr), dst_size, stream.handle)
        raise_if_driver_error(err)

    def fill(self, value: int | BufferProtocol, *, stream: Stream | GraphBuilder):
        """Fill this buffer with a repeating byte pattern.

        Parameters
        ----------
        value : int | :obj:`collections.abc.Buffer`
            - int: Must be in range [0, 256). Converted to 1 byte.
            - :obj:`collections.abc.Buffer`: Must be 1, 2, or 4 bytes.
        stream : :obj:`~_stream.Stream` | :obj:`~_graph.GraphBuilder`
            Stream for the asynchronous fill operation.

        Raises
        ------
        TypeError
            If value is not an int and does not support the buffer protocol.
        ValueError
            If value byte length is not 1, 2, or 4.
            If buffer size is not divisible by value byte length.
        OverflowError
            If int value is outside [0, 256).

        """
        cdef Stream s_stream = Stream_accept(stream)

        # Handle int case: 1-byte fill with automatic overflow checking.
        if isinstance(value, int):
            Buffer_fill_uint8(self, value, s_stream._h_stream)
            return

        # Handle bytes case: direct pointer access without intermediate objects.
        if isinstance(value, bytes):
            Buffer_fill_from_ptr(self, <const char*><bytes>value, len(value), s_stream._h_stream)
            return

        # General buffer protocol path using C buffer API.
        cdef Py_buffer buf
        if PyObject_GetBuffer(value, &buf, PyBUF_SIMPLE) != 0:
            raise TypeError(
                f"value must be an int or support the buffer protocol, got {type(value).__name__}"
            )
        try:
            Buffer_fill_from_ptr(self, <const char*>buf.buf, buf.len, s_stream._h_stream)
        finally:
            PyBuffer_Release(&buf)

    def __dlpack__(
        self,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> TypeVar("PyCapsule"):
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

    @property
    def device_id(self) -> int:
        """Return the device ordinal of this buffer."""
        if self._memory_resource is not None:
            return self._memory_resource.device_id
        _init_mem_attrs(self)
        return self._mem_attrs.device_id

    @property
    def handle(self) -> DevicePointerT:
        """Return the buffer handle object.

        .. caution::

            This handle is a Python object. To get the memory address of the underlying C
            handle, call ``int(Buffer.handle)``.
        """
        # Return raw integer for compatibility with ctypes and other tools
        # that expect a raw pointer value
        return intptr(self._h_ptr)

    @property
    def is_device_accessible(self) -> bool:
        """Return True if this buffer can be accessed by the GPU, otherwise False."""
        if self._memory_resource is not None:
            return self._memory_resource.is_device_accessible
        _init_mem_attrs(self)
        return self._mem_attrs.is_device_accessible

    @property
    def is_host_accessible(self) -> bool:
        """Return True if this buffer can be accessed by the CPU, otherwise False."""
        if self._memory_resource is not None:
            return self._memory_resource.is_host_accessible
        _init_mem_attrs(self)
        return self._mem_attrs.is_host_accessible

    @property
    def is_mapped(self) -> bool:
        """Return True if this buffer is mapped into the process via IPC."""
        return getattr(self._ipc_data, "is_mapped", False)


    @property
    def memory_resource(self) -> MemoryResource:
        """Return the memory resource associated with this buffer."""
        return self._memory_resource

    @property
    def size(self) -> int:
        """Return the memory size of this buffer."""
        return self._size

    @property
    def owner(self) -> object:
        """Return the object holding external allocation."""
        return self._owner


# Memory Attribute Query Helpers
# ------------------------------
cdef inline void _init_mem_attrs(Buffer self):
    """Initialize memory attributes by querying the pointer."""
    if not self._mem_attrs_inited:
        _query_memory_attrs(self._mem_attrs, native(self._h_ptr))
        self._mem_attrs_inited = True


cdef inline int _query_memory_attrs(
    _MemAttrs& out,
    cydriver.CUdeviceptr ptr
) except -1 nogil:
    """Query memory attributes for a device pointer."""
    cdef unsigned int memory_type = 0
    cdef int is_managed = 0
    cdef int device_id = 0
    cdef cydriver.CUpointer_attribute attrs[3]
    cdef uintptr_t vals[3]

    attrs[0] = cydriver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE
    attrs[1] = cydriver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_IS_MANAGED
    attrs[2] = cydriver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
    vals[0] = <uintptr_t><void*>&memory_type
    vals[1] = <uintptr_t><void*>&is_managed
    vals[2] = <uintptr_t><void*>&device_id

    cdef cydriver.CUresult ret
    ret = cydriver.cuPointerGetAttributes(3, attrs, <void**>vals, ptr)
    if ret == cydriver.CUresult.CUDA_ERROR_NOT_INITIALIZED:
        with cython.gil:
            # Device class handles the cuInit call internally
            Device()
        ret = cydriver.cuPointerGetAttributes(3, attrs, <void**>vals, ptr)
    HANDLE_RETURN(ret)

    if memory_type == 0:
        # unregistered host pointer
        out.is_host_accessible = True
        out.is_device_accessible = False
        out.device_id = -1
    elif (
        is_managed
        or memory_type == cydriver.CUmemorytype.CU_MEMORYTYPE_HOST
    ):
        # Managed memory or pinned host memory
        out.is_host_accessible = True
        out.is_device_accessible = True
        out.device_id = device_id
    elif memory_type == cydriver.CUmemorytype.CU_MEMORYTYPE_DEVICE:
        out.is_host_accessible = False
        out.is_device_accessible = True
        out.device_id = device_id
    else:
        with cython.gil:
            raise ValueError(f"Unsupported memory type: {memory_type}")
    return 0


cdef class MemoryResource:
    """Abstract base class for memory resources that manage allocation and
    deallocation of buffers.

    Subclasses must implement methods for allocating and deallocation, as well
    as properties associated with this memory resource from which all allocated
    buffers will inherit. (Since all :class:`Buffer` instances allocated and
    returned by the :meth:`allocate` method would hold a reference to self, the
    buffer properties are retrieved simply by looking up the underlying memory
    resource's respective property.)
    """

    @abc.abstractmethod
    def allocate(self, size_t size, stream: Stream | GraphBuilder | None = None) -> Buffer:
        """Allocate a buffer of the requested size.

        Parameters
        ----------
        size : int
            The size of the buffer to allocate, in bytes.
        stream : :obj:`~_stream.Stream` | :obj:`~_graph.GraphBuilder`, optional
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
    def deallocate(self, ptr: DevicePointerT, size_t size, stream: Stream | GraphBuilder | None = None):
        """Deallocate a buffer previously allocated by this resource.

        Parameters
        ----------
        ptr : :obj:`~_memory.DevicePointerT`
            The pointer or handle to the buffer to deallocate.
        size : int
            The size of the buffer to deallocate, in bytes.
        stream : :obj:`~_stream.Stream` | :obj:`~_graph.GraphBuilder`, optional
            The stream on which to perform the deallocation asynchronously.
            If None, it is up to each memory resource implementation to decide
            and document the behavior.
        """
        ...

# Buffer Implementation Helpers
# -----------------------------
cdef inline Buffer Buffer_from_deviceptr_handle(
    DevicePtrHandle h_ptr,
    size_t size,
    MemoryResource mr,
    object ipc_descriptor = None
):
    """Create a Buffer from an existing DevicePtrHandle."""
    cdef Buffer buf = Buffer.__new__(Buffer)
    buf._h_ptr = h_ptr
    buf._size = size
    buf._memory_resource = mr
    buf._ipc_data = IPCDataForBuffer(ipc_descriptor, True) if ipc_descriptor is not None else None
    buf._owner = None
    buf._mem_attrs_inited = False
    return buf


cdef inline void Buffer_close(Buffer self, object stream):
    """Close a buffer, freeing its memory."""
    cdef Stream s
    if not self._h_ptr:
        return
    # Update deallocation stream if provided
    if stream is not None:
        s = Stream_accept(stream)
        set_deallocation_stream(self._h_ptr, s._h_stream)
    # Reset handle - RAII deleter will free the memory (and release owner ref in C++)
    self._h_ptr.reset()
    self._size = 0
    self._memory_resource = None
    self._ipc_data = None
    self._owner = None


cdef inline int Buffer_fill_uint8(Buffer self, uint8_t value, StreamHandle h_stream) except? -1:
    cdef cydriver.CUdeviceptr ptr = native(self._h_ptr)
    cdef cydriver.CUstream s = native(h_stream)
    with nogil:
        HANDLE_RETURN(cydriver.cuMemsetD8Async(ptr, value, self._size, s))
    return 0


cdef inline int Buffer_fill_from_ptr(
    Buffer self, const char* ptr, size_t width, StreamHandle h_stream
) except? -1:
    cdef size_t buffer_size = self._size
    cdef cydriver.CUdeviceptr dst = native(self._h_ptr)
    cdef cydriver.CUstream s = native(h_stream)

    if width == 1:
        with nogil:
            HANDLE_RETURN(cydriver.cuMemsetD8Async(dst, (<uint8_t*>ptr)[0], buffer_size, s))
    elif width == 2:
        if buffer_size & 0x1:
            raise ValueError(f"buffer size ({buffer_size}) must be divisible by 2")
        with nogil:
            HANDLE_RETURN(cydriver.cuMemsetD16Async(dst, (<uint16_t*>ptr)[0], buffer_size // 2, s))
    elif width == 4:
        if buffer_size & 0x3:
            raise ValueError(f"buffer size ({buffer_size}) must be divisible by 4")
        with nogil:
            HANDLE_RETURN(cydriver.cuMemsetD32Async(dst, (<uint32_t*>ptr)[0], buffer_size // 4, s))
    else:
        raise ValueError(f"value must be 1, 2, or 4 bytes, got {width}")
    return 0
