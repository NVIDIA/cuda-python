# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

cimport cython
from libc.stdint cimport uintptr_t
from libcpp.atomic cimport memory_order_acquire, memory_order_release

from cuda.bindings cimport cydriver
from cuda.core._memory._device_memory_resource import DeviceMemoryResource
from cuda.core._memory._pinned_memory_resource import PinnedMemoryResource
from cuda.core._memory._ipc cimport IPCBufferDescriptor, IPCDataForBuffer
from cuda.core._memory cimport _ipc
from cuda.core._resource_handles cimport (
    DevicePtrHandle,
    StreamHandle,
    ContextHandle,
    deviceptr_create_with_owner,
    deviceptr_create_with_mr,
    register_mr_dealloc_callback,
    as_intptr,
    as_cu,
    get_current_context,
    set_deallocation_stream,
)
from cuda.core.typing import DevicePointerType

from cuda.core._memory._copy_attributes cimport _with_attributes_available
from cuda.core._memory._copy_attributes cimport _to_cu_memcpy_attributes  # no-cython-lint

IF CUDA_CORE_BUILD_MAJOR >= 13:
    from cuda.core._resource_handles cimport memcpy_with_attributes_async

from cuda.core._stream cimport Stream, Stream_accept, Stream_is_legacy_default_token, default_stream
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN, _parse_fill_value

import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

from cuda.core._memory._copy_enums import CopyOptions, _reject_unsupported_during_api_call
from cuda.core._utils.pycompat import BufferProtocol
from cuda.core._dlpack import classify_dl_device, make_py_capsule
from cuda.core._device import Device

if TYPE_CHECKING:
    from cuda.core.graph import GraphBuilder


# =============================================================================
# MR deallocation callback (invoked from C++ shared_ptr deleter)
# =============================================================================

cdef void _mr_dealloc_callback(
    object mr,
    cydriver.CUdeviceptr ptr,
    size_t size,
    const StreamHandle& h_stream,
) noexcept:
    """Called by the C++ deleter to deallocate via MemoryResource.deallocate."""
    cdef Stream stream
    try:
        if not h_stream:
            # No stream was recorded: host-only memory (Buffer._init records
            # none) or a Buffer released before one was set. The default-stream
            # token needs no CUDA context to construct.
            stream = default_stream()
        else:
            stream = Stream._from_handle(Stream, h_stream)
        mr.deallocate(int(ptr), size, stream=stream)
    except Exception as exc:
        print(f"Warning: mr.deallocate() failed during Buffer destruction: {exc}",
              file=sys.stderr)

register_mr_dealloc_callback(_mr_dealloc_callback)


cdef inline void _apply_deallocation_stream(
        const DevicePtrHandle& h_ptr, const StreamHandle& h_stream) except *:
    """Record h_stream as the deallocation stream for h_ptr.

    Translates CUDA_ERROR_INVALID_CONTEXT (default-stream token with no current
    context) into a descriptive RuntimeError instead of a raw CUDAError.
    """
    cdef cydriver.CUresult status = set_deallocation_stream(h_ptr, h_stream)
    if status == cydriver.CUresult.CUDA_ERROR_INVALID_CONTEXT:
        raise RuntimeError(
            "Cannot record a default deallocation stream when no CUDA context is "
            "current. Call Device.set_current() first, or pass stream= with a "
            "non-default Stream."
        )
    HANDLE_RETURN(status)


__all__ = ['Buffer', 'MemoryResource']


# Memory Attribute Query Helpers
# ------------------------------
cdef inline int _query_memory_attrs(
    _MemAttrs& out,
    cydriver.CUdeviceptr ptr
) except -1 nogil:
    """Query memory attributes for a device pointer."""
    cdef unsigned int memory_type = 0
    cdef int is_managed = 0
    cdef int device_id = 0
    cdef cydriver.CUpointer_attribute[3] attrs = [
        cydriver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
        cydriver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_IS_MANAGED,
        cydriver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
    ]
    cdef uintptr_t[3] vals = [
        <uintptr_t><void*>&memory_type,
        <uintptr_t><void*>&is_managed,
        <uintptr_t><void*>&device_id,
    ]

    cdef cydriver.CUresult ret
    ret = cydriver.cuPointerGetAttributes(3, attrs, <void**>vals, ptr)
    if ret == cydriver.CUresult.CUDA_ERROR_NOT_INITIALIZED:
        with cython.gil:
            # Device class handles the cuInit call internally
            Device()
        ret = cydriver.cuPointerGetAttributes(3, attrs, <void**>vals, ptr)
    HANDLE_RETURN(ret)

    # TODO: HMM/ATS-enabled sysmem should also report is_managed=True; the
    # CU_POINTER_ATTRIBUTE_IS_MANAGED query does not capture that yet.
    out.is_managed = is_managed != 0

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


cdef inline void _init_memory_attrs(Buffer self):
    """Initialize memory attributes by querying the pointer."""
    Buffer_check_open(self)
    if not self._mem_attrs_inited.load(memory_order_acquire):
        _query_memory_attrs(self._mem_attrs, as_cu(self._h_ptr))
        self._mem_attrs_inited.store(True, memory_order_release)


cdef bint _stream_is_capturing(Stream s):
    cdef cydriver.CUstreamCaptureStatus cap_status
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        HANDLE_RETURN(cydriver.cuStreamGetCaptureInfo(as_cu(s._h_stream), &cap_status,
                                                     NULL, NULL, NULL, NULL, NULL))
    ELSE:
        HANDLE_RETURN(cydriver.cuStreamGetCaptureInfo(as_cu(s._h_stream), &cap_status,
                                                     NULL, NULL, NULL, NULL))
    return cap_status == cydriver.CU_STREAM_CAPTURE_STATUS_ACTIVE


cdef void _do_copy_with_attributes(
    cydriver.CUdeviceptr dst, cydriver.CUdeviceptr src, size_t nbytes,
    object options, cydriver.CUstream hstream,
):
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        # Routed through the memcpy_with_attributes_async() C++ shim since
        # cydriver.cuMemcpyWithAttributesAsync is absent from cuda-bindings < 13.2.
        cdef cydriver.CUmemcpyAttributes cu_attr = _to_cu_memcpy_attributes(options)
        with nogil:
            HANDLE_RETURN(memcpy_with_attributes_async(dst, src, nbytes, <void*>&cu_attr, hstream))
    ELSE:
        pass  # unreachable: _with_attributes_available() is always False on CUDA 12


cdef void _dispatch_buffer_copy(
    cydriver.CUdeviceptr dst, cydriver.CUdeviceptr src, size_t nbytes,
    Stream s, object options, str method_name,
):
    """Submit a single copy, honoring CopyOptions when the attributes path is usable."""
    if options is None:
        with nogil:
            HANDLE_RETURN(cydriver.cuMemcpyAsync(dst, src, nbytes, as_cu(s._h_stream)))
        return
    if not isinstance(options, CopyOptions):
        raise TypeError(
            f"{method_name}: options must be CopyOptions, got {type(options).__name__}"
        )
    if Stream_is_legacy_default_token(s):
        raise TypeError(
            f"{method_name} does not accept LEGACY_DEFAULT_STREAM with options "
            "(matches copy_batch); cuMemcpyWithAttributesAsync rejects it outright, "
            "unlike PER_THREAD_DEFAULT_STREAM, which is a real stream to the driver "
            "and is accepted. Pass an explicit stream, PER_THREAD_DEFAULT_STREAM, "
            "or options=None."
        )
    if _stream_is_capturing(s):
        raise TypeError(
            f"{method_name} does not support graph capture with options "
            "(matches copy_batch); the driver has no graph-node form of "
            "cuMemcpyWithAttributesAsync, so options cannot be honored in a graph. "
            "Use GraphNode.memcpy for a plain (non-attributed) copy node, or pass "
            "options=None."
        )
    if _with_attributes_available():
        _do_copy_with_attributes(dst, src, nbytes, options, as_cu(s._h_stream))
    else:
        _reject_unsupported_during_api_call(
            options.src_access_order,
            "cuda.bindings and the driver to both report CUDA 13.2 or newer "
            "(cuMemcpyWithAttributesAsync is unavailable here)",
        )
        # STREAM and ANY never require access sooner than stream order, so
        # cuMemcpyAsync satisfies them; options are otherwise silently
        # ignored on this pre-CUDA-13.2 fallback path, matching copy_batch.
        with nogil:
            HANDLE_RETURN(cydriver.cuMemcpyAsync(dst, src, nbytes, as_cu(s._h_stream)))


cdef class Buffer:
    """Represent a handle to allocated memory.

    This generic object provides a unified representation for how
    different memory resources are to give access to their memory
    allocations.

    Support for data interchange mechanisms are provided by DLPack.

    Note
    ----
    Pickling an IPC-enabled :class:`Buffer` embeds an
    :class:`~_memory.IPCBufferDescriptor`. Unpickling reconstructs the buffer
    by calling :meth:`from_ipc_descriptor` and therefore performs an IPC
    import. Do not unpickle buffers from untrusted sources.
    """
    def __cinit__(self) -> None:
        self._clear()

    def _clear(self) -> None:
        self._h_ptr.reset()  # Release the handle
        self._size = 0
        self._memory_resource = None
        self._ipc_data = None
        self._owner = None
        self._mem_attrs_inited.store(False)

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("Buffer objects cannot be instantiated directly. "
                           "Please use MemoryResource APIs.")

    @classmethod
    def _init(
        cls, ptr: DevicePointerType, size_t size, mr: MemoryResource | None = None,
        ipc_descriptor: IPCBufferDescriptor | None = None,
        owner : object | None = None,
        *,
        stream: Stream | GraphBuilder | None = None,
    ) -> Buffer:
        """Create a Buffer from a raw pointer.

        When ``mr`` is provided, the buffer takes ownership: ``mr.deallocate()``
        is called when the buffer is closed or garbage collected.  When ``owner``
        is provided, the owner is kept alive but no deallocation is performed.
        When ``mr`` is provided, a deallocation stream is recorded at creation
        (``stream`` if given, otherwise ``default_stream()``). Recording a
        default-stream token requires a CUDA context to be current. Host-only
        resources (``mr.is_device_accessible`` is ``False``) record no stream
        and need no context.
        """
        if mr is not None and owner is not None:
            raise ValueError("owner and memory resource cannot be both specified together")
        if stream is not None and mr is None:
            raise ValueError("stream requires a memory resource (mr)")
        cdef Buffer self = Buffer.__new__(cls)
        cdef uintptr_t c_ptr = <uintptr_t>(int(ptr))
        cdef Stream s
        cdef cydriver.CUresult _ds_status
        cdef bint record_stream
        if mr is not None:
            s = Stream_accept(default_stream() if stream is None else stream)
            # Host-only memory needs no CUDA context to free, so no deallocation
            # stream is recorded and the driver is not called.
            record_stream = mr.is_device_accessible
            self._h_ptr = deviceptr_create_with_mr(c_ptr, size, mr)
            if record_stream:
                _ds_status = set_deallocation_stream(self._h_ptr, s._h_stream)
                if _ds_status != cydriver.CUresult.CUDA_SUCCESS:
                    # Reset before raising: the DevicePtrHandle destructor would otherwise
                    # invoke _mr_dealloc_callback, which catches any inner exception and
                    # clears the exception state, swallowing the error we're about to raise.
                    self._h_ptr.reset()
                    if _ds_status == cydriver.CUresult.CUDA_ERROR_INVALID_CONTEXT:
                        raise RuntimeError(
                            "Cannot record a default deallocation stream when no CUDA context is "
                            "current. Call Device.set_current() first, or pass stream= with a "
                            "non-default Stream."
                        )
                    HANDLE_RETURN(_ds_status)
        else:
            self._h_ptr = deviceptr_create_with_owner(c_ptr, owner)
        self._size = size
        self._memory_resource = mr
        self._ipc_data = IPCDataForBuffer(ipc_descriptor, True) if ipc_descriptor is not None else None
        self._owner = owner
        self._mem_attrs_inited.store(False)
        return self

    @staticmethod
    def _reduce_helper(mr, ipc_descriptor):
        cdef ContextHandle h_ctx = get_current_context()
        cdef int device_id
        if not h_ctx:
            # Spawned processes unpickle arguments before entering their target,
            # so initialize the context needed to bind the default-stream token.
            device_id = mr.device_id
            (Device(device_id) if device_id >= 0 else Device()).set_current()
        # The parent process's stream is not portable across processes, so the
        # pickle path cannot thread an explicit stream through. Seed the
        # imported buffer's deallocation with the current context's default
        # stream; the receiver can override it before or during close.
        return Buffer.from_ipc_descriptor(mr, ipc_descriptor, stream=default_stream())

    def __reduce__(self) -> tuple[object, ...]:
        # Unpickling performs a live CUDA IPC import from descriptor bytes in the
        # pickle stream. Only deserialize Buffers from a trusted principal.
        # Must not serialize the parent's stream!
        Buffer_check_open(self)
        return Buffer._reduce_helper, (self.memory_resource, self.ipc_descriptor)

    @staticmethod
    def from_handle(
        ptr: DevicePointerType, size_t size, mr: MemoryResource | None = None,
        owner: object | None = None,
        *,
        stream: Stream | GraphBuilder | None = None,
    ) -> Buffer:
        """Create a new :class:`Buffer` object from a pointer.

        Parameters
        ----------
        ptr : :obj:`~_memory.DevicePointerType`
            Allocated buffer handle object
        size : int
            Memory size of the buffer
        mr : :obj:`~_memory.MemoryResource`, optional
            Memory resource associated with the buffer.  When provided,
            :meth:`MemoryResource.deallocate` is called when the buffer is
            closed or garbage collected.
        owner : object, optional
            An object holding external allocation that the ``ptr`` points to.
            The reference is kept as long as the buffer is alive.
            The ``owner`` and ``mr`` cannot be specified together.
        stream : :obj:`~_stream.Stream` | :obj:`~graph.GraphBuilder`, optional
            Keyword-only. The stream used to order the buffer's deallocation
            when ``mr`` owns the pointer. Defaults to ``default_stream()``.
            Recording a default-stream token requires a CUDA context to be
            current. Host-only resources (``mr.is_device_accessible`` is
            ``False``) record no stream and need no context. If the buffer may
            be freed from a different host thread,
            pass a stream other than the per-thread default stream, which
            refers to a different stream on each thread.

        Note
        ----
        When neither ``mr`` nor ``owner`` is specified, this creates a
        non-owning reference.  The pointer will NOT be freed when the
        :class:`Buffer` is closed or garbage collected.
        """
        return Buffer._init(ptr, size, mr=mr, owner=owner, stream=stream)

    @classmethod
    def from_ipc_descriptor(
        cls, mr: DeviceMemoryResource | PinnedMemoryResource, ipc_descriptor: IPCBufferDescriptor,
        *, stream: Stream
    ) -> Buffer:
        """Import a buffer that was exported from another process.

        Parameters
        ----------
        mr : :obj:`~_memory.DeviceMemoryResource` | :obj:`~_memory.PinnedMemoryResource`
            The IPC-enabled memory resource matching the exporting process.
        ipc_descriptor : :obj:`~_memory.IPCBufferDescriptor`
            The descriptor exported from another process.
        stream : :obj:`~_stream.Stream`
            Keyword-only. The stream used for asynchronous deallocation when
            the buffer is closed or garbage collected.

        Note
        ----
        The descriptor payload and ``size`` are supplied by the exporting peer
        and must be treated as untrusted input unless the peer is known to be
        cooperating.
        """
        return _ipc.Buffer_from_ipc_descriptor(cls, mr, ipc_descriptor, stream)

    @property
    @cython.critical_section
    def ipc_descriptor(self) -> IPCBufferDescriptor:
        """Descriptor for sharing this buffer with other processes."""
        Buffer_check_open(self)
        cdef object ipc_data
        if self._ipc_data is None:
            ipc_data = IPCDataForBuffer(_ipc.Buffer_get_ipc_descriptor(self), False)
            if self._ipc_data is None:
                self._ipc_data = ipc_data
        return self._ipc_data.ipc_descriptor

    def close(self, stream: Stream | GraphBuilder | None = None) -> None:
        """Deallocate this buffer asynchronously on the given stream.

        This buffer is released back to their memory resource
        asynchronously on the given stream.

        Parameters
        ----------
        stream : :obj:`~_stream.Stream` | :obj:`~graph.GraphBuilder`, optional
            The stream object to use for asynchronous deallocation. If None,
            the deallocation stream stored in the handle is used.

        See Also
        --------
        set_deallocation_stream
            Change the deallocation stream without closing the buffer.
        """
        Buffer_close(self, stream)

    def set_deallocation_stream(self, stream: Stream | GraphBuilder) -> None:
        """Change the stream that orders this buffer's eventual deallocation.

        The buffer remains open and usable. A later :meth:`close` without a
        stream, garbage collection, or release of the final retained device
        pointer handle uses the replacement stream.

        This method does not synchronize streams or establish dependencies.
        The caller must ensure that allocation and all accesses are ordered
        before the deallocation on ``stream``.

        Parameters
        ----------
        stream : :obj:`~_stream.Stream` | :obj:`~graph.GraphBuilder`
            The stream to use for eventual asynchronous deallocation.

        Raises
        ------
        RuntimeError
            If the buffer is already closed, or if a default-stream token
            cannot be bound because no CUDA context is current.
        TypeError
            If ``stream`` is ``None`` or is not an accepted stream object.

        Notes
        -----
        Synchronizing concurrent mutation and destruction of the same buffer
        is the caller's responsibility.
        """
        Buffer_set_deallocation_stream(self, stream)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def copy_to(self, dst: Buffer | None = None, *, stream: Stream | GraphBuilder,
                options: CopyOptions | None = None) -> Buffer:
        """Copy from this buffer to the dst buffer asynchronously on the given stream.

        Copies the data from this buffer to the provided dst buffer.
        If the dst buffer is not provided, then a new buffer is first
        allocated using the associated memory resource before the copy.

        Parameters
        ----------
        dst : :obj:`~_memory.Buffer`, optional
            Destination buffer to copy data to. If not provided, a new buffer
            is allocated using this buffer's memory resource.
        stream : :obj:`~_stream.Stream` | :obj:`~graph.GraphBuilder`
            Keyword argument specifying the stream for the
            asynchronous copy
        options : :class:`~utils.CopyOptions`, optional
            Transfer hints (source access order, location hints, overlap mode).
            Honored when cuda.bindings and the driver are both CUDA 13.2 or
            newer. Not accepted with ``LEGACY_DEFAULT_STREAM``; use
            ``PER_THREAD_DEFAULT_STREAM`` instead. Not accepted with a
            capturing stream either, since a graph cannot represent these
            attributes; use :meth:`graph.GraphNode.memcpy` for a plain,
            non-attributed copy node, or pass ``options=None``. On an older
            cuda.bindings/driver, ``src_access_order`` values of ``STREAM``
            and ``ANY`` are silently ignored; ``DURING_API_CALL`` raises
            instead of silently downgrading its guarantee.

        Raises
        ------
        TypeError
            If ``options`` is not a :class:`~utils.CopyOptions` instance, or
            if ``options`` is given together with ``LEGACY_DEFAULT_STREAM``
            or a stream currently in graph capture mode.
        RuntimeError
            If ``options.src_access_order`` is ``DURING_API_CALL`` and
            cuda.bindings or the driver is older than CUDA 13.2: falling
            back to a plain copy cannot honor that guarantee.

        """
        Buffer_check_open(self)
        cdef Stream s = Stream_accept(stream)
        cdef size_t src_size = self._size

        if dst is None:
            if self._memory_resource is None:
                raise ValueError("a destination buffer must be provided (this "
                                 "buffer does not have a memory_resource)")
            dst = self._memory_resource.allocate(src_size, stream=s)
        else:
            Buffer_check_open(<Buffer>dst)

        cdef size_t dst_size = dst._size
        if dst_size != src_size:
            raise ValueError( "buffer sizes mismatch between src and dst (sizes "
                             f"are: src={src_size}, dst={dst_size})"
            )
        _dispatch_buffer_copy(
            as_cu(dst._h_ptr), as_cu(self._h_ptr), src_size, s, options, "copy_to")
        return dst

    def copy_from(self, src: Buffer, *, stream: Stream | GraphBuilder,
                  options: CopyOptions | None = None) -> None:
        """Copy from the src buffer to this buffer asynchronously on the given stream.

        Parameters
        ----------
        src : :obj:`~_memory.Buffer`
            Source buffer to copy data from
        stream : :obj:`~_stream.Stream` | :obj:`~graph.GraphBuilder`
            Keyword argument specifying the stream for the
            asynchronous copy
        options : :class:`~utils.CopyOptions`, optional
            Transfer hints (source access order, location hints, overlap mode).
            Honored when cuda.bindings and the driver are both CUDA 13.2 or
            newer. Not accepted with ``LEGACY_DEFAULT_STREAM``; use
            ``PER_THREAD_DEFAULT_STREAM`` instead. Not accepted with a
            capturing stream either, since a graph cannot represent these
            attributes; use :meth:`graph.GraphNode.memcpy` for a plain,
            non-attributed copy node, or pass ``options=None``. On an older
            cuda.bindings/driver, ``src_access_order`` values of ``STREAM``
            and ``ANY`` are silently ignored; ``DURING_API_CALL`` raises
            instead of silently downgrading its guarantee.

        Raises
        ------
        TypeError
            If ``options`` is not a :class:`~utils.CopyOptions` instance, or
            if ``options`` is given together with ``LEGACY_DEFAULT_STREAM``
            or a stream currently in graph capture mode.
        RuntimeError
            If ``options.src_access_order`` is ``DURING_API_CALL`` and
            cuda.bindings or the driver is older than CUDA 13.2: falling
            back to a plain copy cannot honor that guarantee.
        """
        Buffer_check_open(self)
        Buffer_check_open(src)
        cdef Stream s = Stream_accept(stream)
        cdef size_t dst_size = self._size
        cdef size_t src_size = src._size

        if src_size != dst_size:
            raise ValueError( "buffer sizes mismatch between src and dst (sizes "
                             f"are: src={src_size}, dst={dst_size})"
            )
        _dispatch_buffer_copy(
            as_cu(self._h_ptr), as_cu(src._h_ptr), dst_size, s, options, "copy_from")

    def fill(self, value: int | BufferProtocol, *, stream: Stream | GraphBuilder) -> None:
        """Fill this buffer with a repeating byte pattern.

        Parameters
        ----------
        value : int | :obj:`collections.abc.Buffer`
            - int: Must be in range [0, 256). Converted to 1 byte.
            - :obj:`collections.abc.Buffer`: Must be 1, 2, or 4 bytes.
        stream : :obj:`~_stream.Stream` | :obj:`~graph.GraphBuilder`
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
        Buffer_check_open(self)
        cdef Stream s_stream = Stream_accept(stream)
        cdef unsigned int val
        cdef unsigned int elem_size
        val, elem_size = _parse_fill_value(value)

        cdef size_t buffer_size = self._size
        cdef cydriver.CUdeviceptr dst = as_cu(self._h_ptr)
        cdef cydriver.CUstream s = as_cu(s_stream._h_stream)

        if elem_size == 1:
            with nogil:
                HANDLE_RETURN(cydriver.cuMemsetD8Async(dst, val, buffer_size, s))
        elif elem_size == 2:
            if buffer_size & 0x1:
                raise ValueError(f"buffer size ({buffer_size}) must be divisible by 2")
            with nogil:
                HANDLE_RETURN(cydriver.cuMemsetD16Async(dst, val, buffer_size // 2, s))
        elif elem_size == 4:
            if buffer_size & 0x3:
                raise ValueError(f"buffer size ({buffer_size}) must be divisible by 4")
            with nogil:
                HANDLE_RETURN(cydriver.cuMemsetD32Async(dst, val, buffer_size // 4, s))

    def __dlpack__(
        self,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> object:
        # Note: we ignore the stream argument entirely (as if it is -1).
        # It is the user's responsibility to maintain stream order.
        Buffer_check_open(self)
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
        return make_py_capsule(self, versioned)

    def __dlpack_device__(self) -> tuple[int, int]:
        Buffer_check_open(self)
        return classify_dl_device(self)

    def __buffer__(self, flags: int, /) -> memoryview:
        # Support for Python-level buffer protocol as per PEP 688.
        # This raises a BufferError unless:
        #   1. Python is 3.12+
        #   2. This Buffer object is host accessible
        raise NotImplementedError("WIP: Buffer.__buffer__ hasn't been implemented yet.")

    def __release_buffer__(self, buffer: memoryview, /) -> None:
        # Supporting method paired with __buffer__.
        raise NotImplementedError("WIP: Buffer.__release_buffer__ hasn't been implemented yet.")

    @property
    def device_id(self) -> int:
        """Return the device ordinal of this buffer, or -1 for memory not bound to a device."""
        Buffer_check_open(self)
        if self._memory_resource is not None:
            return self._memory_resource.device_id
        _init_memory_attrs(self)
        return self._mem_attrs.device_id

    @property
    def handle(self) -> int:
        """Return the buffer handle object.

        .. caution::

            This handle is a Python object. To get the memory address of the underlying C
            handle, call ``int(Buffer.handle)``.
        """
        # Return raw integer for compatibility with ctypes and other tools
        # that expect a raw pointer value
        return as_intptr(self._h_ptr)

    @property
    def is_closed(self) -> bool:
        """Whether this buffer has been closed."""
        return self._h_ptr.get() == NULL

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Buffer):
            return NotImplemented
        cdef Buffer other_buf = <Buffer>other
        return (as_intptr(self._h_ptr) == as_intptr(other_buf._h_ptr) and
                self._size == other_buf._size)

    def __hash__(self) -> int:
        return hash((as_intptr(self._h_ptr), self._size))

    def __repr__(self) -> str:
        maybe_is_mapped = " is_mapped=True" if self.is_mapped else ""
        return f"<Buffer ptr={as_intptr(self._h_ptr):#x} size={self._size}{maybe_is_mapped}>"

    @property
    def is_device_accessible(self) -> bool:
        """Return True if this buffer can be accessed by the GPU, otherwise False."""
        Buffer_check_open(self)
        if self._memory_resource is not None:
            return self._memory_resource.is_device_accessible
        _init_memory_attrs(self)
        return self._mem_attrs.is_device_accessible

    @property
    def is_host_accessible(self) -> bool:
        """Return True if this buffer can be accessed by the CPU, otherwise False."""
        Buffer_check_open(self)
        if self._memory_resource is not None:
            return self._memory_resource.is_host_accessible
        _init_memory_attrs(self)
        return self._mem_attrs.is_host_accessible

    @property
    def is_managed(self) -> bool:
        """Return True if this buffer is CUDA managed (unified) memory, otherwise False."""
        Buffer_check_open(self)
        _init_memory_attrs(self)
        if self._mem_attrs.is_managed:
            return True
        # Pool-allocated managed memory does not set CU_POINTER_ATTRIBUTE_IS_MANAGED,
        # so fall back to the memory resource when it advertises managed allocations.
        return self._memory_resource is not None and self._memory_resource.is_managed

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

    def allocate(self, size_t size, *, stream: Stream | GraphBuilder) -> Buffer:
        """Allocate a buffer of the requested size.

        Parameters
        ----------
        size : int
            The size of the buffer to allocate, in bytes.
        stream : :obj:`~_stream.Stream` | :obj:`~graph.GraphBuilder`
            Keyword-only. The stream on which to perform the allocation
            asynchronously. Must be passed explicitly; pass
            ``device.default_stream`` to use the default stream. For subclasses
            that support stream-ordered deallocation, this stream also orders
            the buffer's eventual deallocation, so if the buffer may be freed
            from a different host thread, prefer a stream other than the
            per-thread default stream, which refers to a different stream on
            each thread.

        Returns
        -------
        Buffer
            The allocated buffer object, which can be used for device or host operations
            depending on the resource's properties.
        """
        raise TypeError("MemoryResource.allocate must be implemented by subclasses.")

    def deallocate(
        self,
        ptr: DevicePointerType,
        size_t size,
        *,
        stream: Stream | GraphBuilder
    ) -> None:
        """Deallocate a buffer previously allocated by this resource.

        Parameters
        ----------
        ptr : :obj:`~_memory.DevicePointerType`
            The pointer or handle to the buffer to deallocate.
        size : int
            The size of the buffer to deallocate, in bytes.
        stream : :obj:`~_stream.Stream` | :obj:`~graph.GraphBuilder`
            Keyword-only. The stream on which to perform the deallocation
            asynchronously. Must be passed explicitly; pass
            ``device.default_stream`` to use the default stream.
        """
        raise TypeError("MemoryResource.deallocate must be implemented by subclasses.")

    @property
    def is_device_accessible(self) -> bool:
        """Whether buffers allocated by this resource are device-accessible."""
        raise TypeError("MemoryResource.is_device_accessible must be implemented by subclasses.")

    @property
    def is_host_accessible(self) -> bool:
        """Whether buffers allocated by this resource are host-accessible."""
        raise TypeError("MemoryResource.is_host_accessible must be implemented by subclasses.")

    @property
    def is_managed(self) -> bool:
        """Whether buffers allocated by this resource are CUDA managed (unified) memory."""
        return False

    @property
    def device_id(self) -> int:
        """Device ID associated with this memory resource, or -1 if not applicable."""
        raise TypeError("MemoryResource.device_id must be implemented by subclasses.")


# Buffer Implementation Helpers
# -----------------------------
cdef Buffer Buffer_from_deviceptr_handle(
    DevicePtrHandle h_ptr,
    size_t size,
    MemoryResource mr,
    object ipc_descriptor = None,
    type cls = Buffer,
):
    """Create a Buffer (or subclass instance) from an existing DevicePtrHandle."""
    cdef Buffer buf = cls.__new__(cls)
    buf._h_ptr = h_ptr
    buf._size = size
    buf._memory_resource = mr
    buf._ipc_data = IPCDataForBuffer(ipc_descriptor, True) if ipc_descriptor is not None else None
    buf._owner = None
    buf._mem_attrs_inited.store(False)
    return buf


cdef tuple Buffer_coerce_batch(object buffers, str what, str single_hint):
    """Coerce ``buffers`` to a ``tuple[Buffer, ...]``; reject a bare Buffer.

    Shared by the batched free functions. Passing one Buffer is rejected
    rather than treated as a one-element batch so that the per-buffer API
    named by ``single_hint`` stays the single obvious way to do it.
    """
    cdef list out
    if isinstance(buffers, Buffer):
        raise TypeError(
            f"{what}: pass a sequence of Buffers; for a single buffer use {single_hint}"
        )
    if not isinstance(buffers, Sequence):
        raise TypeError(
            f"{what}: buffers must be a sequence of Buffer, got {type(buffers).__name__}"
        )
    if not buffers:
        raise ValueError(f"{what}: empty buffers sequence")
    out = []
    for item in buffers:
        if not isinstance(item, Buffer):
            raise TypeError(f"{what}: expected Buffer, got {type(item).__name__}")
        Buffer_check_open(<Buffer>item)
        out.append(item)
    return tuple(out)

cdef inline void Buffer_set_deallocation_stream(Buffer self, object stream):
    """Validate and replace a live buffer's deallocation recipe."""
    Buffer_check_open(self)
    cdef Stream s = Stream_accept(stream)
    _apply_deallocation_stream(self._h_ptr, s._h_stream)


cdef inline void Buffer_close(Buffer self, object stream):
    """Close a buffer, freeing its memory."""
    if not self._h_ptr:
        return
    # Update deallocation stream if provided
    if stream is not None:
        Buffer_set_deallocation_stream(self, stream)
    # Reset handle - RAII deleter will free the memory (and release owner ref in C++)
    self._h_ptr.reset()
    self._size = 0
    self._memory_resource = None
    self._ipc_data = None
    self._owner = None
    self._mem_attrs_inited.store(False)
