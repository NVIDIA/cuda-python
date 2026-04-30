# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

IF CUDA_CORE_BUILD_MAJOR >= 13:
    from cpython.mem cimport PyMem_Free, PyMem_Malloc

from cuda.bindings cimport cydriver
from cuda.core._memory._buffer cimport Buffer
from cuda.core._resource_handles cimport as_cu
from cuda.core._stream cimport Stream, Stream_accept
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN

from cuda.core._utils.cuda_utils import driver
from cuda.core._memory._managed_location import _coerce_location
from cuda.core._memory._managed_memory_options import (
    AdviseOptions,
    DiscardOptions,
    DiscardPrefetchOptions,
    PrefetchOptions,
)


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

# Lazily cached: maps driver.CUmem_advise enum value → string alias.
cdef dict _ADVICE_ENUM_TO_ALIAS = None


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


cdef void _require_managed_buffer(Buffer self, str what):
    # Buffer.is_managed handles both pointer-attribute and memory-resource
    # paths (e.g. pool-allocated managed memory whose pointer attribute
    # does not advertise CU_POINTER_ATTRIBUTE_IS_MANAGED).
    if not self.is_managed:
        raise ValueError(f"{what} requires a managed-memory allocation")


cdef tuple _coerce_buffer_targets(object targets, str what):
    cdef Buffer buf
    cdef list out
    if isinstance(targets, Buffer):
        return (<Buffer>targets,)
    if isinstance(targets, (list, tuple)):
        if not targets:
            raise ValueError(f"{what}: empty targets sequence")
        out = []
        for t in targets:
            buf = <Buffer?>t
            out.append(buf)
        return tuple(out)
    raise TypeError(
        f"{what}: targets must be a Buffer or sequence of Buffer, "
        f"got {type(targets).__name__}"
    )


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
    options : :class:`DiscardOptions`, optional
        Reserved for future per-call flags. ``None`` (default) and
        ``DiscardOptions()`` are equivalent.
    stream : :class:`~_stream.Stream` | :class:`~graph.GraphBuilder`
        Stream for the asynchronous discard (keyword-only).

    Raises
    ------
    NotImplementedError
        On a CUDA 12 build of ``cuda.core``. Discard requires CUDA 13+.
    """
    if options is not None and not isinstance(options, DiscardOptions):
        raise TypeError(
            "discard options must be a DiscardOptions instance or None, "
            f"got {type(options).__name__}"
        )
    cdef tuple bufs = _coerce_buffer_targets(targets, "discard")
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
    targets,
    advice,
    location=None,
    *,
    options=None,
):
    """Apply managed-memory advice to one or more allocation ranges.

    Parameters
    ----------
    targets : :class:`Buffer` | Sequence[:class:`Buffer`]
        One or more managed allocations to advise.
    advice : str | :obj:`~driver.CUmem_advise`
        Managed-memory advice. String aliases (``"set_read_mostly"``,
        ``"unset_read_mostly"``, ``"set_preferred_location"``,
        ``"unset_preferred_location"``, ``"set_accessed_by"``,
        ``"unset_accessed_by"``) and ``CUmem_advise`` enum values are accepted.
    location : :class:`Location` | :obj:`~_device.Device` | int | Sequence[...]
        Target location(s). Required for advice values that consult a
        location; ignored (may be ``None``) for ``set_read_mostly``,
        ``unset_read_mostly``, and ``unset_preferred_location``. A sequence
        must match ``len(targets)``.
    options : :class:`AdviseOptions`, optional
        Reserved for future per-call flags. ``None`` (default) and
        ``AdviseOptions()`` are equivalent.
    """
    if options is not None and not isinstance(options, AdviseOptions):
        raise TypeError(
            "advise options must be an AdviseOptions instance or None, "
            f"got {type(options).__name__}"
        )
    cdef str advice_name
    cdef object advice_value
    advice_name, advice_value = _normalize_managed_advice(advice)
    cdef bint allow_none = advice_name in _MANAGED_ADVICE_IGNORE_LOCATION
    cdef frozenset allowed_kinds = _MANAGED_ADVICE_ALLOWED_LOCTYPES[advice_name]

    cdef tuple bufs = _coerce_buffer_targets(targets, "advise")
    cdef Py_ssize_t n = len(bufs)
    cdef tuple locs = _broadcast_locations(location, n, allow_none, "advise")

    cdef Buffer buf
    cdef object loc
    for buf in bufs:
        _require_managed_buffer(buf, "advise")
    for loc in locs:
        if loc is not None and loc.kind not in allowed_kinds:
            raise ValueError(
                f"advise '{advice_name}' does not support location_type='{loc.kind}'"
            )

    cdef Py_ssize_t i
    for i in range(n):
        _do_single_advise(<Buffer>bufs[i], advice_value, locs[i], allow_none)


cdef void _do_single_advise(Buffer buf, object advice_value, object loc, bint allow_none):
    cdef cydriver.CUdeviceptr cu_ptr = as_cu(buf._h_ptr)
    cdef size_t nbytes = buf._size
    cdef cydriver.CUmem_advise advice_enum = <cydriver.CUmem_advise>(<int>int(advice_value))
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        cdef cydriver.CUmemLocation cu_loc
        if loc is None:
            # Driver ignores location for read_mostly / unset_preferred_location
            # advice values but still validates the CUmemLocation; pass a
            # host placeholder.
            cu_loc.type = cydriver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST
            cu_loc.id = 0
        else:
            cu_loc = _to_cumemlocation(loc)
        with nogil:
            HANDLE_RETURN(cydriver.cuMemAdvise(cu_ptr, nbytes, advice_enum, cu_loc))
    ELSE:
        cdef int dev_int = -1 if loc is None else _to_legacy_device(loc)
        with nogil:
            HANDLE_RETURN(cydriver.cuMemAdvise(cu_ptr, nbytes, advice_enum, dev_int))


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
    options : :class:`PrefetchOptions`, optional
        Reserved for future per-call flags. ``None`` (default) and
        ``PrefetchOptions()`` are equivalent.
    stream : :class:`~_stream.Stream` | :class:`~graph.GraphBuilder`
        Stream for the asynchronous prefetch (keyword-only).

    Raises
    ------
    NotImplementedError
        If ``len(targets) > 1`` on a CUDA 12 build of ``cuda.core``.
    """
    if options is not None and not isinstance(options, PrefetchOptions):
        raise TypeError(
            "prefetch options must be a PrefetchOptions instance or None, "
            f"got {type(options).__name__}"
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
    targets,
    location=None,
    *,
    options=None,
    stream,
):
    """Discard one or more managed-memory ranges and prefetch them to a target location.

    Parameters
    ----------
    targets : :class:`Buffer` | Sequence[:class:`Buffer`]
        One or more managed allocations to discard and re-prefetch.
    location : :class:`Location` | :obj:`~_device.Device` | int | Sequence[...]
        Target location(s). A single location applies to all targets;
        a sequence must match ``len(targets)``.
    options : :class:`DiscardPrefetchOptions`, optional
        Reserved for future per-call flags. ``None`` (default) and
        ``DiscardPrefetchOptions()`` are equivalent.
    stream : :class:`~_stream.Stream` | :class:`~graph.GraphBuilder`
        Stream for the asynchronous operation (keyword-only).

    Raises
    ------
    NotImplementedError
        On a CUDA 12 build of ``cuda.core``. Discard-and-prefetch
        requires CUDA 13+.
    """
    if options is not None and not isinstance(options, DiscardPrefetchOptions):
        raise TypeError(
            "discard_prefetch options must be a DiscardPrefetchOptions "
            f"instance or None, got {type(options).__name__}"
        )
    cdef tuple bufs = _coerce_buffer_targets(targets, "discard_prefetch")
    cdef Py_ssize_t n = len(bufs)
    cdef tuple locs = _broadcast_locations(location, n, False, "discard_prefetch")
    cdef Stream s = Stream_accept(stream)

    cdef Buffer buf
    for buf in bufs:
        _require_managed_buffer(buf, "discard_prefetch")

    _do_batch_discard_prefetch(bufs, locs, s)


cdef void _do_batch_discard_prefetch(tuple bufs, tuple locs, Stream s):
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
                HANDLE_RETURN(cydriver.cuMemDiscardAndPrefetchBatchAsync(
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
            "discard_prefetch requires a CUDA 13 build of cuda.core"
        )
