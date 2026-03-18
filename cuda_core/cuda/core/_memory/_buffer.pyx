# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

cimport cython
from libc.stdint cimport uintptr_t

from cuda.bindings cimport cydriver
from cuda.core._memory._device_memory_resource import DeviceMemoryResource
from cuda.core._memory._pinned_memory_resource import PinnedMemoryResource
from cuda.core._memory._ipc cimport IPCBufferDescriptor, IPCDataForBuffer
from cuda.core._memory cimport _ipc
from cuda.core._resource_handles cimport (
    DevicePtrHandle,
    StreamHandle,
    deviceptr_create_with_owner,
    deviceptr_create_with_mr,
    register_mr_dealloc_callback,
    as_intptr,
    as_cu,
    set_deallocation_stream,
)

from cuda.core._stream cimport Stream, Stream_accept
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN, _parse_fill_value

import sys
from typing import TypeVar

if sys.version_info >= (3, 12):
    from collections.abc import Buffer as BufferProtocol
else:
    BufferProtocol = object

from cuda.core._dlpack import DLDeviceType, make_py_capsule
from cuda.core._utils.cuda_utils import driver, get_binding_version, handle_return
from cuda.core._device import Device


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
    try:
        stream = None
        if h_stream:
            stream = Stream._from_handle(Stream, h_stream)
        mr.deallocate(int(ptr), size, stream)
    except Exception as exc:
        print(f"Warning: mr.deallocate() failed during Buffer destruction: {exc}",
              file=sys.stderr)

register_mr_dealloc_callback(_mr_dealloc_callback)


__all__ = ['Buffer', 'MemoryResource']


DevicePointerT = driver.CUdeviceptr | int | None
"""
A type union of :obj:`~driver.CUdeviceptr`, `int` and `None` for hinting
:attr:`Buffer.handle`.
"""


cdef tuple _VALID_MANAGED_LOCATION_TYPES = (
    "device",
    "host",
    "host_numa",
    "host_numa_current",
)

cdef dict _MANAGED_LOCATION_TYPE_ATTRS = {
    "device": "CU_MEM_LOCATION_TYPE_DEVICE",
    "host": "CU_MEM_LOCATION_TYPE_HOST",
    "host_numa": "CU_MEM_LOCATION_TYPE_HOST_NUMA",
    "host_numa_current": "CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT",
}

cdef dict _MANAGED_ADVICE_ALIASES = {
    "set_read_mostly": "CU_MEM_ADVISE_SET_READ_MOSTLY",
    "unset_read_mostly": "CU_MEM_ADVISE_UNSET_READ_MOSTLY",
    "set_preferred_location": "CU_MEM_ADVISE_SET_PREFERRED_LOCATION",
    "unset_preferred_location": "CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION",
    "set_accessed_by": "CU_MEM_ADVISE_SET_ACCESSED_BY",
    "unset_accessed_by": "CU_MEM_ADVISE_UNSET_ACCESSED_BY",
}

cdef frozenset _MANAGED_ADVICE_IGNORE_LOCATION = frozenset((
    "set_read_mostly",
    "unset_read_mostly",
    "unset_preferred_location",
))

cdef frozenset _ALL_LOCATION_TYPES = frozenset(("device", "host", "host_numa", "host_numa_current"))
cdef frozenset _DEVICE_HOST_NUMA = frozenset(("device", "host", "host_numa"))
cdef frozenset _DEVICE_HOST_ONLY = frozenset(("device", "host"))

cdef dict _MANAGED_ADVICE_ALLOWED_LOCTYPES = {
    "set_read_mostly": _DEVICE_HOST_NUMA,
    "unset_read_mostly": _DEVICE_HOST_NUMA,
    "set_preferred_location": _ALL_LOCATION_TYPES,
    "unset_preferred_location": _DEVICE_HOST_NUMA,
    "set_accessed_by": _DEVICE_HOST_ONLY,
    "unset_accessed_by": _DEVICE_HOST_ONLY,
}

cdef int _MANAGED_SIZE_NOT_PROVIDED = -1
cdef int _HOST_NUMA_CURRENT_ID = 0
cdef int _FIRST_PREFETCH_LOCATION_INDEX = 0
cdef size_t _SINGLE_RANGE_COUNT = 1
cdef size_t _SINGLE_PREFETCH_LOCATION_COUNT = 1
cdef unsigned long long _MANAGED_OPERATION_FLAGS = 0

# Lazily cached values for immutable runtime properties.
cdef object _CU_DEVICE_CPU = None
cdef dict _ADVICE_ENUM_TO_ALIAS = None
cdef int _V2_BINDINGS = -1
cdef int _DISCARD_PREFETCH_SUPPORTED = -1


cdef inline object _managed_location_enum(str location_type):
    cdef str attr_name = _MANAGED_LOCATION_TYPE_ATTRS[location_type]
    cdef object result = getattr(driver.CUmemLocationType, attr_name, None)
    if result is None:
        raise RuntimeError(
            f"Managed-memory location type {location_type!r} is not supported by the "
            f"installed cuda.bindings package."
        )
    return result


cdef inline object _make_managed_location(str location_type, int location_id):
    global _CU_DEVICE_CPU
    cdef object location = driver.CUmemLocation()
    location.type = _managed_location_enum(location_type)
    if location_type == "host":
        if _CU_DEVICE_CPU is None:
            _CU_DEVICE_CPU = int(getattr(driver, "CU_DEVICE_CPU", -1))
        location.id = _CU_DEVICE_CPU
    elif location_type == "host_numa_current":
        location.id = _HOST_NUMA_CURRENT_ID
    else:
        location.id = location_id
    return location


cdef inline tuple _normalize_managed_advice(object advice):
    cdef str alias
    cdef str attr_name
    if isinstance(advice, str):
        alias = advice.lower()
        attr_name = _MANAGED_ADVICE_ALIASES.get(alias)
        if attr_name is None:
            raise ValueError(
                "advice must be one of "
                f"{tuple(sorted(_MANAGED_ADVICE_ALIASES))!r}, got {advice!r}"
            )
        return alias, getattr(driver.CUmem_advise, attr_name)

    if isinstance(advice, driver.CUmem_advise):
        global _ADVICE_ENUM_TO_ALIAS
        if _ADVICE_ENUM_TO_ALIAS is None:
            _ADVICE_ENUM_TO_ALIAS = {}
            for alias, attr_name in _MANAGED_ADVICE_ALIASES.items():
                enum_val = getattr(driver.CUmem_advise, attr_name, None)
                if enum_val is not None:
                    _ADVICE_ENUM_TO_ALIAS[enum_val] = alias
        alias = _ADVICE_ENUM_TO_ALIAS.get(advice)
        if alias is None:
            raise ValueError(f"Unsupported advice value: {advice!r}")
        return alias, advice

    raise TypeError(
        "advice must be a cuda.bindings.driver.CUmem_advise value or a supported string alias"
    )


cdef inline object _normalize_managed_location(
    object location,
    object location_type,
    str what,
    bint allow_none=False,
    frozenset allowed_loctypes=_ALL_LOCATION_TYPES,
):
    cdef object loc_type
    cdef int loc_id

    if isinstance(location, Device):
        location = location.device_id

    if location_type is not None and not isinstance(location_type, str):
        raise TypeError(f"{what} location_type must be a string or None, got {type(location_type).__name__}")

    loc_type = None if location_type is None else (<str>location_type).lower()
    if loc_type is not None and loc_type not in _VALID_MANAGED_LOCATION_TYPES:
        raise ValueError(
            f"{what} location_type must be one of {_VALID_MANAGED_LOCATION_TYPES!r} "
            f"or None, got {location_type!r}"
        )

    if loc_type is not None and loc_type not in allowed_loctypes:
        raise ValueError(f"{what} does not support location_type='{loc_type}'")

    if loc_type is None:
        if location is None:
            if allow_none:
                return _make_managed_location("host", -1)
            raise ValueError(f"{what} requires a location")
        if not isinstance(location, int):
            raise TypeError(
                f"{what} location must be a Device, int, or None, got {type(location).__name__}"
            )
        loc_id = <int>location
        if loc_id == -1:
            if "host" not in allowed_loctypes:
                raise ValueError(f"{what} does not support host locations")
            return _make_managed_location("host", -1)
        elif loc_id >= 0:
            return _make_managed_location("device", loc_id)
        else:
            raise ValueError(
                f"{what} location must be a device ordinal (>= 0), -1 for host, or None; got {location!r}"
            )
    elif loc_type == "device":
        if isinstance(location, int) and <int>location >= 0:
            loc_id = <int>location
        else:
            raise ValueError(
                f"{what} location must be a device ordinal (>= 0) when location_type is 'device', got {location!r}"
            )
        return _make_managed_location(loc_type, loc_id)
    elif loc_type == "host":
        if location not in (None, -1):
            raise ValueError(
                f"{what} location must be None or -1 when location_type is 'host', got {location!r}"
            )
        return _make_managed_location(loc_type, -1)
    elif loc_type == "host_numa":
        if not isinstance(location, int) or <int>location < 0:
            raise ValueError(
                f"{what} location must be a NUMA node ID (>= 0) when location_type is 'host_numa', got {location!r}"
            )
        return _make_managed_location(loc_type, <int>location)
    else:
        if location is not None:
            raise ValueError(
                f"{what} location must be None when location_type is 'host_numa_current', got {location!r}"
            )
        return _make_managed_location(loc_type, _HOST_NUMA_CURRENT_ID)


cdef inline bint _managed_location_uses_v2_bindings():
    # cuda.bindings 13.x switches these APIs to CUmemLocation-based wrappers.
    global _V2_BINDINGS
    if _V2_BINDINGS < 0:
        _V2_BINDINGS = 1 if get_binding_version() >= (13, 0) else 0
    return _V2_BINDINGS != 0


cdef object _LEGACY_LOC_DEVICE = None
cdef object _LEGACY_LOC_HOST = None

cdef inline int _managed_location_to_legacy_device(object location, str what):
    global _LEGACY_LOC_DEVICE, _LEGACY_LOC_HOST
    if _LEGACY_LOC_DEVICE is None:
        _LEGACY_LOC_DEVICE = _managed_location_enum("device")
        _LEGACY_LOC_HOST = _managed_location_enum("host")
    cdef object loc_type = location.type
    if loc_type == _LEGACY_LOC_DEVICE or loc_type == _LEGACY_LOC_HOST:
        return <int>location.id
    raise RuntimeError(
        f"{what} requires cuda.bindings 13.x for location_type={loc_type!r}"
    )


cdef inline void _require_managed_buffer(Buffer self, str what):
    _init_mem_attrs(self)
    if not self._mem_attrs.is_managed:
        raise ValueError(f"{what} requires a managed-memory allocation")


cdef inline void _require_managed_discard_prefetch_support(str what):
    global _DISCARD_PREFETCH_SUPPORTED
    if _DISCARD_PREFETCH_SUPPORTED < 0:
        _DISCARD_PREFETCH_SUPPORTED = 1 if hasattr(driver, "cuMemDiscardAndPrefetchBatchAsync") else 0
    if not _DISCARD_PREFETCH_SUPPORTED:
        raise RuntimeError(
            f"{what} requires cuda.bindings support for cuMemDiscardAndPrefetchBatchAsync"
        )


cdef inline tuple _managed_range_from_buffer(
    Buffer buffer,
    int size,
    str what,
):
    if size != _MANAGED_SIZE_NOT_PROVIDED:
        raise TypeError(f"{what} does not accept size= when target is a Buffer")
    _require_managed_buffer(buffer, what)
    return buffer.handle, buffer._size


cdef inline uintptr_t _coerce_raw_pointer(object target, str what) except? 0:
    cdef object ptr_obj
    try:
        ptr_obj = int(target)
    except Exception as exc:
        raise TypeError(
            f"{what} target must be a Buffer or a raw pointer, got {type(target).__name__}"
        ) from exc
    if ptr_obj < 0:
        raise ValueError(f"{what} target pointer must be >= 0, got {target!r}")
    return <uintptr_t>ptr_obj


cdef inline int _require_managed_pointer(uintptr_t ptr, str what) except -1:
    cdef _MemAttrs mem_attrs
    with nogil:
        _query_memory_attrs(mem_attrs, <cydriver.CUdeviceptr>ptr)
    if not mem_attrs.is_managed:
        raise ValueError(f"{what} requires a managed-memory allocation")
    return 0


cdef inline tuple _normalize_managed_target_range(
    object target,
    int size,
    str what,
):
    cdef uintptr_t ptr

    if isinstance(target, Buffer):
        return _managed_range_from_buffer(<Buffer>target, size, what)

    if size == _MANAGED_SIZE_NOT_PROVIDED:
        raise TypeError(f"{what} requires size= when target is a raw pointer")
    ptr = _coerce_raw_pointer(target, what)
    _require_managed_pointer(ptr, what)
    return ptr, <size_t>size


def advise(
    target,
    advice: driver.CUmem_advise | str,
    location: Device | int | None = None,
    *,
    int size=_MANAGED_SIZE_NOT_PROVIDED,
    location_type: str | None = None,
):
    """Apply managed-memory advice to an allocation range.

    Parameters
    ----------
    target : :class:`Buffer` | int | object
        Managed allocation to operate on. This may be a :class:`Buffer` or a
        raw pointer (requires ``size=``).
    advice : :obj:`~driver.CUmem_advise` | str
        Managed-memory advice to apply. String aliases such as
        ``"set_read_mostly"``, ``"set_preferred_location"``, and
        ``"set_accessed_by"`` are accepted.
    location : :obj:`~_device.Device` | int | None, optional
        Target location. When ``location_type`` is ``None``, values are
        interpreted as a device ordinal, ``-1`` for host, or ``None`` for
        advice values that ignore location.
    size : int, optional
        Allocation size in bytes. Required when ``target`` is a raw pointer.
    location_type : str | None, optional
        Explicit location kind. Supported values are ``"device"``, ``"host"``,
        ``"host_numa"``, and ``"host_numa_current"``.
    """
    cdef str advice_name
    cdef object ptr
    cdef size_t nbytes

    ptr, nbytes = _normalize_managed_target_range(target, size, "advise")
    advice_name, advice = _normalize_managed_advice(advice)
    location = _normalize_managed_location(
        location,
        location_type,
        "advise",
        allow_none=advice_name in _MANAGED_ADVICE_IGNORE_LOCATION,
        allowed_loctypes=_MANAGED_ADVICE_ALLOWED_LOCTYPES[advice_name],
    )
    if _managed_location_uses_v2_bindings():
        handle_return(driver.cuMemAdvise(ptr, nbytes, advice, location))
    else:
        handle_return(
            driver.cuMemAdvise(
                ptr,
                nbytes,
                advice,
                _managed_location_to_legacy_device(location, "advise"),
            )
        )


def prefetch(
    target,
    location: Device | int | None = None,
    *,
    stream: Stream | GraphBuilder,
    int size=_MANAGED_SIZE_NOT_PROVIDED,
    location_type: str | None = None,
):
    """Prefetch a managed-memory allocation range to a target location.

    Parameters
    ----------
    target : :class:`Buffer` | int | object
        Managed allocation to operate on. This may be a :class:`Buffer` or a
        raw pointer (requires ``size=``).
    location : :obj:`~_device.Device` | int | None, optional
        Target location. When ``location_type`` is ``None``, values are
        interpreted as a device ordinal, ``-1`` for host, or ``None``.
        A location is required for prefetch.
    stream : :obj:`~_stream.Stream` | :obj:`~_graph.GraphBuilder`
        Keyword argument specifying the stream for the asynchronous prefetch.
    size : int, optional
        Allocation size in bytes. Required when ``target`` is a raw pointer.
    location_type : str | None, optional
        Explicit location kind. Supported values are ``"device"``, ``"host"``,
        ``"host_numa"``, and ``"host_numa_current"``.
    """
    cdef Stream s = Stream_accept(stream)
    cdef object ptr
    cdef size_t nbytes

    ptr, nbytes = _normalize_managed_target_range(target, size, "prefetch")
    location = _normalize_managed_location(
        location,
        location_type,
        "prefetch",
    )
    if _managed_location_uses_v2_bindings():
        handle_return(
            driver.cuMemPrefetchAsync(
                ptr,
                nbytes,
                location,
                _MANAGED_OPERATION_FLAGS,
                s.handle,
            )
        )
    else:
        handle_return(
            driver.cuMemPrefetchAsync(
                ptr,
                nbytes,
                _managed_location_to_legacy_device(location, "prefetch"),
                s.handle,
            )
        )


def discard_prefetch(
    target,
    location: Device | int | None = None,
    *,
    stream: Stream | GraphBuilder,
    int size=_MANAGED_SIZE_NOT_PROVIDED,
    location_type: str | None = None,
):
    """Discard a managed-memory allocation range and prefetch it to a target location.

    Parameters
    ----------
    target : :class:`Buffer` | int | object
        Managed allocation to operate on. This may be a :class:`Buffer` or a
        raw pointer (requires ``size=``).
    location : :obj:`~_device.Device` | int | None, optional
        Target location. When ``location_type`` is ``None``, values are
        interpreted as a device ordinal, ``-1`` for host, or ``None``.
        A location is required for discard_prefetch.
    stream : :obj:`~_stream.Stream` | :obj:`~_graph.GraphBuilder`
        Keyword argument specifying the stream for the asynchronous operation.
    size : int, optional
        Allocation size in bytes. Required when ``target`` is a raw pointer.
    location_type : str | None, optional
        Explicit location kind. Supported values are ``"device"``, ``"host"``,
        ``"host_numa"``, and ``"host_numa_current"``.
    """
    _require_managed_discard_prefetch_support("discard_prefetch")
    cdef Stream s = Stream_accept(stream)
    cdef object ptr
    cdef object batch_ptr
    cdef size_t nbytes

    ptr, nbytes = _normalize_managed_target_range(target, size, "discard_prefetch")
    batch_ptr = driver.CUdeviceptr(int(ptr))
    location = _normalize_managed_location(
        location,
        location_type,
        "discard_prefetch",
    )
    handle_return(
        driver.cuMemDiscardAndPrefetchBatchAsync(
            [batch_ptr],
            [nbytes],
            _SINGLE_RANGE_COUNT,
            [location],
            [_FIRST_PREFETCH_LOCATION_INDEX],
            _SINGLE_PREFETCH_LOCATION_COUNT,
            _MANAGED_OPERATION_FLAGS,
            s.handle,
        )
    )

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
        self._h_ptr.reset()  # Release the handle
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
        ipc_descriptor: IPCBufferDescriptor | None = None,
        owner : object | None = None
    ):
        """Create a Buffer from a raw pointer.

        When ``mr`` is provided, the buffer takes ownership: ``mr.deallocate()``
        is called when the buffer is closed or garbage collected.  When ``owner``
        is provided, the owner is kept alive but no deallocation is performed.
        """
        if mr is not None and owner is not None:
            raise ValueError("owner and memory resource cannot be both specified together")
        cdef Buffer self = Buffer.__new__(cls)
        cdef uintptr_t c_ptr = <uintptr_t>(int(ptr))
        if mr is not None:
            self._h_ptr = deviceptr_create_with_mr(c_ptr, size, mr)
        else:
            self._h_ptr = deviceptr_create_with_owner(c_ptr, owner)
        self._size = size
        self._memory_resource = mr
        self._ipc_data = IPCDataForBuffer(ipc_descriptor, True) if ipc_descriptor is not None else None
        self._owner = owner
        self._mem_attrs_inited = False
        return self

    @staticmethod
    def _reduce_helper(mr, ipc_descriptor):
        return Buffer.from_ipc_descriptor(mr, ipc_descriptor)

    def __reduce__(self):
        # Must not serialize the parent's stream!
        return Buffer._reduce_helper, (self.memory_resource, self.get_ipc_descriptor())

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
            Memory resource associated with the buffer.  When provided,
            :meth:`MemoryResource.deallocate` is called when the buffer is
            closed or garbage collected.
        owner : object, optional
            An object holding external allocation that the ``ptr`` points to.
            The reference is kept as long as the buffer is alive.
            The ``owner`` and ``mr`` cannot be specified together.

        Note
        ----
        When neither ``mr`` nor ``owner`` is specified, this creates a
        non-owning reference.  The pointer will NOT be freed when the
        :class:`Buffer` is closed or garbage collected.
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

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
        cdef Stream s = Stream_accept(stream)
        cdef size_t src_size = self._size

        if dst is None:
            if self._memory_resource is None:
                raise ValueError("a destination buffer must be provided (this "
                                 "buffer does not have a memory_resource)")
            dst = self._memory_resource.allocate(src_size, s)

        cdef size_t dst_size = dst._size
        if dst_size != src_size:
            raise ValueError( "buffer sizes mismatch between src and dst (sizes "
                             f"are: src={src_size}, dst={dst_size})"
            )
        with nogil:
            HANDLE_RETURN(cydriver.cuMemcpyAsync(
                as_cu(dst._h_ptr), as_cu(self._h_ptr), src_size, as_cu(s._h_stream)))
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
        cdef Stream s = Stream_accept(stream)
        cdef size_t dst_size = self._size
        cdef size_t src_size = src._size

        if src_size != dst_size:
            raise ValueError( "buffer sizes mismatch between src and dst (sizes "
                             f"are: src={src_size}, dst={dst_size})"
            )
        with nogil:
            HANDLE_RETURN(cydriver.cuMemcpyAsync(
                as_cu(self._h_ptr), as_cu(src._h_ptr), dst_size, as_cu(s._h_stream)))

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
        return as_intptr(self._h_ptr)

    def __eq__(self, other) -> bool:
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
        _query_memory_attrs(self._mem_attrs, as_cu(self._h_ptr))
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
        out.is_managed = False
    elif (
        is_managed
        or memory_type == cydriver.CUmemorytype.CU_MEMORYTYPE_HOST
    ):
        # Managed memory or pinned host memory
        out.is_host_accessible = True
        out.is_device_accessible = True
        out.device_id = device_id
        out.is_managed = is_managed != 0
    elif memory_type == cydriver.CUmemorytype.CU_MEMORYTYPE_DEVICE:
        out.is_host_accessible = False
        out.is_device_accessible = True
        out.device_id = device_id
        out.is_managed = False
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
        raise TypeError("MemoryResource.allocate must be implemented by subclasses.")

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
    def device_id(self) -> int:
        """Device ID associated with this memory resource, or -1 if not applicable."""
        raise TypeError("MemoryResource.device_id must be implemented by subclasses.")


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
