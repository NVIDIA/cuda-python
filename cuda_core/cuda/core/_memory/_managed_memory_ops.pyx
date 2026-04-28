# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cpython.mem cimport PyMem_Free, PyMem_Malloc
from libc.stdint cimport uintptr_t

from cuda.bindings cimport cydriver
from cuda.core._memory._buffer cimport Buffer, _init_mem_attrs
from cuda.core._resource_handles cimport as_cu
from cuda.core._stream cimport Stream, Stream_accept
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN

from cuda.core._utils.cuda_utils import driver, handle_return
from cuda.core._utils.version import binding_version
from cuda.core._device import Device
from cuda.core._memory._managed_location import Location, _coerce_location


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

cdef int _HOST_NUMA_CURRENT_ID = 0
cdef int _FIRST_PREFETCH_LOCATION_INDEX = 0
cdef size_t _SINGLE_RANGE_COUNT = 1
cdef size_t _SINGLE_PREFETCH_LOCATION_COUNT = 1
cdef unsigned long long _MANAGED_OPERATION_FLAGS = 0

# Lazily cached values for immutable runtime properties.
cdef object _CU_DEVICE_CPU = None
cdef dict _ADVICE_ENUM_TO_ALIAS = None
_V2_BINDINGS = -1
cdef int _DISCARD_PREFETCH_SUPPORTED = -1


cdef object _managed_location_enum(str location_type):
    cdef str attr_name = _MANAGED_LOCATION_TYPE_ATTRS[location_type]
    cdef object result = getattr(driver.CUmemLocationType, attr_name, None)
    if result is None:
        raise RuntimeError(
            f"Managed-memory location type {location_type!r} is not supported by the "
            f"installed cuda.bindings package."
        )
    return result


cdef object _make_managed_location(str location_type, int location_id):
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


cdef tuple _normalize_managed_advice(object advice):
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


cdef object _normalize_managed_location(
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


cdef bint _managed_location_uses_v2_bindings():
    # cuda.bindings 13.x switches these APIs to CUmemLocation-based wrappers.
    global _V2_BINDINGS
    if _V2_BINDINGS < 0:
        _V2_BINDINGS = 1 if binding_version() >= (13, 0) else 0
    return _V2_BINDINGS != 0


cdef object _LEGACY_LOC_DEVICE = None
cdef object _LEGACY_LOC_HOST = None

cdef int _managed_location_to_legacy_device(object location, str what):
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


cdef void _require_managed_buffer(Buffer self, str what):
    _init_mem_attrs(self)
    if not self._mem_attrs.is_managed:
        raise ValueError(f"{what} requires a managed-memory allocation")


# Coerce ``targets`` (single Buffer or sequence) to a tuple[Buffer, ...].
cdef tuple _coerce_buffer_targets(object targets, str what):
    cdef list out
    if isinstance(targets, Buffer):
        return (<Buffer>targets,)
    if isinstance(targets, (list, tuple)):
        if not targets:
            raise ValueError(f"{what}: empty targets sequence")
        out = []
        for t in targets:
            if not isinstance(t, Buffer):
                raise TypeError(
                    f"{what}: each target must be a Buffer, got {type(t).__name__}"
                )
            out.append(t)
        return tuple(out)
    raise TypeError(
        f"{what}: targets must be a Buffer or sequence of Buffer, "
        f"got {type(targets).__name__}"
    )


# Broadcast a single location across ``n`` targets, or coerce a length-N
# sequence elementwise.
cdef tuple _broadcast_locations(object location, Py_ssize_t n, bint allow_none, str what):
    cdef object coerced
    if isinstance(location, (list, tuple)):
        if len(location) != n:
            raise ValueError(
                f"{what}: location length {len(location)} does not match "
                f"targets length {n}"
            )
        return tuple(_coerce_location(loc, allow_none=allow_none) for loc in location)
    coerced = _coerce_location(location, allow_none=allow_none)
    return tuple([coerced] * n)


IF CUDA_CORE_BUILD_MAJOR >= 13:
    # Convert a Location dataclass to a cydriver.CUmemLocation struct.
    cdef inline cydriver.CUmemLocation _to_cumemlocation(object loc):
        cdef cydriver.CUmemLocation out
        cdef str kind = loc.kind
        if kind == "device":
            out.type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            out.id = <int>loc.id
        elif kind == "host":
            out.type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST
            out.id = 0
        elif kind == "host_numa":
            out.type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
            out.id = <int>loc.id
        else:  # host_numa_current
            out.type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT
            out.id = 0
        return out
ELSE:
    # CUDA 12 cuMemPrefetchAsync takes a device ordinal (-1 = host).
    cdef inline int _to_legacy_device(object loc) except? -2:
        cdef str kind = loc.kind
        if kind == "device":
            return <int>loc.id
        if kind == "host":
            return -1
        raise RuntimeError(
            f"location_type={kind!r} requires a CUDA 13 build of cuda.core"
        )


cdef void _require_managed_discard_prefetch_support(str what):
    global _DISCARD_PREFETCH_SUPPORTED
    if _DISCARD_PREFETCH_SUPPORTED < 0:
        _DISCARD_PREFETCH_SUPPORTED = 1 if hasattr(driver, "cuMemDiscardAndPrefetchBatchAsync") else 0
    if not _DISCARD_PREFETCH_SUPPORTED:
        raise RuntimeError(
            f"{what} requires cuda.bindings support for cuMemDiscardAndPrefetchBatchAsync"
        )


def discard(
    targets,
    *,
    options=None,
    stream,
):
    """Discard one or more managed-memory ranges.

    Parameters
    ----------
    targets : :class:`Buffer` | Sequence[:class:`Buffer`]
        One or more managed allocations to discard. Their resident pages
        are released without prefetching new contents; subsequent access
        is satisfied by lazy migration.
    options : None
        Reserved for future per-call flags. Must be ``None``.
    stream : :class:`~_stream.Stream` | :class:`~graph.GraphBuilder`
        Stream for the asynchronous discard (keyword-only).

    Raises
    ------
    NotImplementedError
        On a CUDA 12 build of ``cuda.core``. Discard requires CUDA 13+.
    """
    if options is not None:
        raise TypeError(
            f"discard options must be None (reserved); got {type(options).__name__}"
        )
    cdef tuple bufs = _coerce_buffer_targets(targets, "discard")
    cdef Py_ssize_t n = len(bufs)
    cdef Stream s = Stream_accept(stream)

    cdef Buffer buf
    for buf in bufs:
        _require_managed_buffer(buf, "discard")

    _do_batch_discard(bufs, s)


cdef void _do_batch_discard(tuple bufs, Stream s):
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        cdef Py_ssize_t n = len(bufs)
        cdef cydriver.CUstream hstream = as_cu(s._h_stream)
        cdef cydriver.CUdeviceptr* ptrs = <cydriver.CUdeviceptr*>PyMem_Malloc(
            n * sizeof(cydriver.CUdeviceptr)
        )
        cdef size_t* sizes = <size_t*>PyMem_Malloc(n * sizeof(size_t))
        if not (ptrs and sizes):
            PyMem_Free(ptrs)
            PyMem_Free(sizes)
            raise MemoryError()
        cdef Buffer buf
        cdef Py_ssize_t i
        try:
            for i in range(n):
                buf = <Buffer>bufs[i]
                ptrs[i] = as_cu(buf._h_ptr)
                sizes[i] = buf._size
            with nogil:
                HANDLE_RETURN(cydriver.cuMemDiscardBatchAsync(
                    ptrs, sizes, <size_t>n, 0, hstream,
                ))
        finally:
            PyMem_Free(ptrs)
            PyMem_Free(sizes)
    ELSE:
        raise NotImplementedError(
            "discard requires a CUDA 13 build of cuda.core"
        )


def advise(
    target: Buffer,
    advice: driver.CUmem_advise | str,
    location: Device | int | None = None,
    *,
    location_type: str | None = None,
):
    """Apply managed-memory advice to an allocation range.

    Parameters
    ----------
    target : :class:`Buffer`
        Managed allocation to operate on.
    advice : :obj:`~driver.CUmem_advise` | str
        Managed-memory advice to apply. String aliases such as
        ``"set_read_mostly"``, ``"set_preferred_location"``, and
        ``"set_accessed_by"`` are accepted.
    location : :obj:`~_device.Device` | int | None, optional
        Target location. When ``location_type`` is ``None``, values are
        interpreted as a device ordinal, ``-1`` for host, or ``None`` for
        advice values that ignore location.
    location_type : str | None, optional
        Explicit location kind. Supported values are ``"device"``, ``"host"``,
        ``"host_numa"``, and ``"host_numa_current"``.
    """
    if not isinstance(target, Buffer):
        raise TypeError(f"advise target must be a Buffer, got {type(target).__name__}")
    cdef Buffer buf = <Buffer>target
    _require_managed_buffer(buf, "advise")
    cdef str advice_name
    cdef object ptr = buf.handle
    cdef size_t nbytes = buf._size

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
    targets,
    location=None,
    *,
    options=None,
    stream,
):
    """Prefetch one or more managed-memory ranges to a target location.

    Parameters
    ----------
    targets : :class:`Buffer` | Sequence[:class:`Buffer`]
        One or more managed allocations to operate on.
    location : :class:`Location` | :obj:`~_device.Device` | int | Sequence[...]
        Target location(s). A single location applies to all targets; a
        sequence must match ``len(targets)``. ``Device`` and ``int`` values
        are coerced to :class:`Location` (``-1`` maps to host).
    options : None
        Reserved for future per-call flags. Must be ``None``.
    stream : :class:`~_stream.Stream` | :class:`~graph.GraphBuilder`
        Stream for the asynchronous prefetch (keyword-only).

    Raises
    ------
    NotImplementedError
        If ``len(targets) > 1`` on a CUDA 12 build of ``cuda.core``.
    """
    if options is not None:
        raise TypeError(
            f"prefetch options must be None (reserved); got {type(options).__name__}"
        )
    cdef tuple bufs = _coerce_buffer_targets(targets, "prefetch")
    cdef Py_ssize_t n = len(bufs)
    cdef tuple locs = _broadcast_locations(location, n, False, "prefetch")
    cdef Stream s = Stream_accept(stream)

    cdef Buffer buf
    for buf in bufs:
        _require_managed_buffer(buf, "prefetch")

    if n == 1:
        _do_single_prefetch(<Buffer>bufs[0], locs[0], s)
    else:
        _do_batch_prefetch(bufs, locs, s)


cdef void _do_single_prefetch(Buffer buf, object loc, Stream s):
    cdef cydriver.CUdeviceptr cu_ptr = as_cu(buf._h_ptr)
    cdef size_t nbytes = buf._size
    cdef cydriver.CUstream hstream = as_cu(s._h_stream)
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        cdef cydriver.CUmemLocation cu_loc = _to_cumemlocation(loc)
        with nogil:
            HANDLE_RETURN(cydriver.cuMemPrefetchAsync(cu_ptr, nbytes, cu_loc, 0, hstream))
    ELSE:
        cdef int dev_int = _to_legacy_device(loc)
        with nogil:
            HANDLE_RETURN(cydriver.cuMemPrefetchAsync(cu_ptr, nbytes, dev_int, hstream))


cdef void _do_batch_prefetch(tuple bufs, tuple locs, Stream s):
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        cdef Py_ssize_t n = len(bufs)
        cdef cydriver.CUstream hstream = as_cu(s._h_stream)
        cdef cydriver.CUdeviceptr* ptrs = <cydriver.CUdeviceptr*>PyMem_Malloc(
            n * sizeof(cydriver.CUdeviceptr)
        )
        cdef size_t* sizes = <size_t*>PyMem_Malloc(n * sizeof(size_t))
        cdef cydriver.CUmemLocation* loc_arr = <cydriver.CUmemLocation*>PyMem_Malloc(
            n * sizeof(cydriver.CUmemLocation)
        )
        cdef size_t* loc_indices = <size_t*>PyMem_Malloc(n * sizeof(size_t))
        if not (ptrs and sizes and loc_arr and loc_indices):
            PyMem_Free(ptrs)
            PyMem_Free(sizes)
            PyMem_Free(loc_arr)
            PyMem_Free(loc_indices)
            raise MemoryError()
        cdef Buffer buf
        cdef Py_ssize_t i
        try:
            for i in range(n):
                buf = <Buffer>bufs[i]
                ptrs[i] = as_cu(buf._h_ptr)
                sizes[i] = buf._size
                loc_arr[i] = _to_cumemlocation(locs[i])
                loc_indices[i] = <size_t>i
            with nogil:
                HANDLE_RETURN(cydriver.cuMemPrefetchBatchAsync(
                    ptrs, sizes, <size_t>n,
                    loc_arr, loc_indices, <size_t>n,
                    0, hstream,
                ))
        finally:
            PyMem_Free(ptrs)
            PyMem_Free(sizes)
            PyMem_Free(loc_arr)
            PyMem_Free(loc_indices)
    ELSE:
        raise NotImplementedError(
            "batched prefetch requires a CUDA 13 build of cuda.core"
        )


def discard_prefetch(
    target: Buffer,
    location: Device | int | None = None,
    *,
    stream: Stream | GraphBuilder,
    location_type: str | None = None,
):
    """Discard a managed-memory allocation range and prefetch it to a target location.

    Parameters
    ----------
    target : :class:`Buffer`
        Managed allocation to operate on.
    location : :obj:`~_device.Device` | int | None, optional
        Target location. When ``location_type`` is ``None``, values are
        interpreted as a device ordinal, ``-1`` for host, or ``None``.
        A location is required for discard_prefetch.
    stream : :obj:`~_stream.Stream` | :obj:`~_graph.GraphBuilder`
        Keyword argument specifying the stream for the asynchronous operation.
    location_type : str | None, optional
        Explicit location kind. Supported values are ``"device"``, ``"host"``,
        ``"host_numa"``, and ``"host_numa_current"``.
    """
    if not isinstance(target, Buffer):
        raise TypeError(f"discard_prefetch target must be a Buffer, got {type(target).__name__}")
    cdef Buffer buf = <Buffer>target
    _require_managed_buffer(buf, "discard_prefetch")
    _require_managed_discard_prefetch_support("discard_prefetch")
    cdef Stream s = Stream_accept(stream)
    cdef object ptr = buf.handle
    cdef size_t nbytes = buf._size
    cdef object batch_ptr = driver.CUdeviceptr(int(ptr))
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
