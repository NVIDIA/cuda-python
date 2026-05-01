# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

IF CUDA_CORE_BUILD_MAJOR >= 13:
    from libcpp.vector cimport vector

from cuda.bindings cimport cydriver
from cuda.core._memory._buffer cimport Buffer
from cuda.core._resource_handles cimport as_cu
from cuda.core._stream cimport Stream, Stream_accept
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN

from cuda.core._utils.cuda_utils import driver
from cuda.core._memory._managed_location import _coerce_location


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

# Reverse lookup: enum value → alias. Built once at module load.
cdef dict _ADVICE_ENUM_TO_ALIAS = {
    getattr(driver.CUmem_advise, attr_name): alias
    for alias, attr_name in _MANAGED_ADVICE_ALIASES.items()
    if hasattr(driver.CUmem_advise, attr_name)
}


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


cdef tuple _coerce_batch_buffers(object buffers, str what):
    """Coerce ``buffers`` to a tuple[Buffer, ...]; rejects a single Buffer.

    For single-buffer operations, use the corresponding ManagedBuffer
    instance method instead.
    """
    cdef Buffer buf
    cdef list out
    if isinstance(buffers, Buffer):
        raise TypeError(
            f"{what}: pass a sequence of Buffers; for a single buffer use "
            f"the ManagedBuffer instance method"
        )
    if isinstance(buffers, Sequence):
        if not buffers:
            raise ValueError(f"{what}: empty buffers sequence")
        out = []
        for t in buffers:
            buf = <Buffer?>t
            out.append(buf)
        return tuple(out)
    raise TypeError(
        f"{what}: buffers must be a sequence of Buffer, "
        f"got {type(buffers).__name__}"
    )


cdef tuple _broadcast_locations(object location, Py_ssize_t n, bint allow_none, str what):
    cdef object coerced
    if isinstance(location, Sequence):
        if len(location) != n:
            raise ValueError(
                f"{what}: location length {len(location)} does not match "
                f"targets length {n}"
            )
        return tuple(_coerce_location(loc, allow_none=allow_none) for loc in location)
    coerced = _coerce_location(location, allow_none=allow_none)
    return tuple([coerced] * n)


IF CUDA_CORE_BUILD_MAJOR >= 13:
    # Convert a _LocSpec dataclass to a cydriver.CUmemLocation struct.
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


def discard_batch(buffers, *, stream):
    """Discard a batch of managed-memory ranges.

    Requires CUDA 13+. For a single buffer, use
    :meth:`ManagedBuffer.discard` instead.

    Parameters
    ----------
    buffers : Sequence[:class:`Buffer`]
        Two or more managed allocations to discard. Resident pages are
        released without prefetching new contents; subsequent access is
        satisfied by lazy migration.
    stream : :class:`~_stream.Stream` | :class:`~graph.GraphBuilder`
        Stream for the asynchronous discard (keyword-only).

    Raises
    ------
    NotImplementedError
        On a CUDA 12 build of ``cuda.core``.
    """
    cdef tuple bufs = _coerce_batch_buffers(buffers, "discard_batch")
    cdef Stream s = Stream_accept(stream)

    cdef Buffer buf
    for buf in bufs:
        _require_managed_buffer(buf, "discard_batch")

    _do_batch_discard(bufs, s)


def _do_single_discard_py(Buffer buf, stream):
    """Internal: single-buffer discard for ManagedBuffer.discard()."""
    _require_managed_buffer(buf, "discard")
    cdef Stream s = Stream_accept(stream)
    # No single-range cuMemDiscard exists; route through the batched call
    # with count=1.
    cdef tuple bufs = (buf,)
    _do_batch_discard(bufs, s)


cdef void _do_batch_discard(tuple bufs, Stream s):
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        cdef Py_ssize_t n = len(bufs)
        cdef cydriver.CUstream hstream = as_cu(s._h_stream)
        cdef vector[cydriver.CUdeviceptr] ptrs
        cdef vector[size_t] sizes
        ptrs.resize(n)
        sizes.resize(n)
        cdef Buffer buf
        cdef Py_ssize_t i
        for i in range(n):
            buf = <Buffer>bufs[i]
            ptrs[i] = as_cu(buf._h_ptr)
            sizes[i] = buf._size
        with nogil:
            HANDLE_RETURN(cydriver.cuMemDiscardBatchAsync(
                ptrs.data(), sizes.data(), <size_t>n, 0, hstream,
            ))
    ELSE:
        raise NotImplementedError(
            "discard requires a CUDA 13 build of cuda.core"
        )


def _advise_one(Buffer buf, advice, location):
    """Internal: apply managed-memory advice to a single buffer.

    Used by :class:`ManagedBuffer` property setters. Not part of the
    public API.
    """
    cdef str advice_name
    cdef object advice_value
    advice_name, advice_value = _normalize_managed_advice(advice)
    cdef bint allow_none = advice_name in _MANAGED_ADVICE_IGNORE_LOCATION
    cdef frozenset allowed_kinds = _MANAGED_ADVICE_ALLOWED_LOCTYPES[advice_name]
    cdef object loc = _coerce_location(location, allow_none=allow_none)
    if loc is not None and loc.kind not in allowed_kinds:
        raise ValueError(
            f"advise '{advice_name}' does not support location_type='{loc.kind}'"
        )
    _require_managed_buffer(buf, "advise")
    _do_single_advise(buf, advice_value, loc, allow_none)


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


def prefetch_batch(buffers, locations, *, stream):
    """Prefetch a batch of managed-memory ranges to target locations.

    Requires CUDA 13+. For a single buffer, use
    :meth:`ManagedBuffer.prefetch` instead.

    Parameters
    ----------
    buffers : Sequence[:class:`Buffer`]
        Two or more managed allocations to operate on.
    locations : :class:`~cuda.core.Device` | :class:`~cuda.core.Host` | Sequence[...]
        Target location(s). A single location applies to all buffers; a
        sequence must match ``len(buffers)``.
    stream : :class:`~_stream.Stream` | :class:`~graph.GraphBuilder`
        Stream for the asynchronous prefetch (keyword-only).

    Raises
    ------
    NotImplementedError
        On a CUDA 12 build of ``cuda.core``.
    """
    cdef tuple bufs = _coerce_batch_buffers(buffers, "prefetch_batch")
    cdef Py_ssize_t n = len(bufs)
    cdef tuple locs = _broadcast_locations(locations, n, False, "prefetch_batch")
    cdef Stream s = Stream_accept(stream)

    cdef Buffer buf
    for buf in bufs:
        _require_managed_buffer(buf, "prefetch_batch")

    _do_batch_prefetch(bufs, locs, s)


def _do_single_prefetch_py(Buffer buf, location, stream):
    """Internal: single-buffer prefetch for ManagedBuffer.prefetch().

    Uses cuMemPrefetchAsync (works on CUDA 12 and 13).
    """
    _require_managed_buffer(buf, "prefetch")
    cdef object loc = _coerce_location(location, allow_none=False)
    cdef Stream s = Stream_accept(stream)
    _do_single_prefetch(buf, loc, s)


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
        cdef vector[cydriver.CUdeviceptr] ptrs
        cdef vector[size_t] sizes
        cdef vector[cydriver.CUmemLocation] loc_arr
        cdef vector[size_t] loc_indices
        ptrs.resize(n)
        sizes.resize(n)
        loc_arr.resize(n)
        loc_indices.resize(n)
        cdef Buffer buf
        cdef Py_ssize_t i
        for i in range(n):
            buf = <Buffer>bufs[i]
            ptrs[i] = as_cu(buf._h_ptr)
            sizes[i] = buf._size
            loc_arr[i] = _to_cumemlocation(locs[i])
            loc_indices[i] = <size_t>i
        with nogil:
            HANDLE_RETURN(cydriver.cuMemPrefetchBatchAsync(
                ptrs.data(), sizes.data(), <size_t>n,
                loc_arr.data(), loc_indices.data(), <size_t>n,
                0, hstream,
            ))
    ELSE:
        raise NotImplementedError(
            "batched prefetch requires a CUDA 13 build of cuda.core"
        )


def discard_prefetch_batch(buffers, locations, *, stream):
    """Discard a batch of managed-memory ranges and prefetch them to target locations.

    Requires CUDA 13+. For a single buffer, use
    :meth:`ManagedBuffer.discard_prefetch` instead.

    Parameters
    ----------
    buffers : Sequence[:class:`Buffer`]
        Two or more managed allocations to discard and re-prefetch.
    locations : :class:`~cuda.core.Device` | :class:`~cuda.core.Host` | Sequence[...]
        Target location(s). A single location applies to all buffers;
        a sequence must match ``len(buffers)``.
    stream : :class:`~_stream.Stream` | :class:`~graph.GraphBuilder`
        Stream for the asynchronous operation (keyword-only).

    Raises
    ------
    NotImplementedError
        On a CUDA 12 build of ``cuda.core``.
    """
    cdef tuple bufs = _coerce_batch_buffers(buffers, "discard_prefetch_batch")
    cdef Py_ssize_t n = len(bufs)
    cdef tuple locs = _broadcast_locations(locations, n, False, "discard_prefetch_batch")
    cdef Stream s = Stream_accept(stream)

    cdef Buffer buf
    for buf in bufs:
        _require_managed_buffer(buf, "discard_prefetch_batch")

    _do_batch_discard_prefetch(bufs, locs, s)


def _do_single_discard_prefetch_py(Buffer buf, location, stream):
    """Internal: single-buffer discard+prefetch for
    ManagedBuffer.discard_prefetch()."""
    _require_managed_buffer(buf, "discard_prefetch")
    cdef object loc = _coerce_location(location, allow_none=False)
    cdef Stream s = Stream_accept(stream)
    cdef tuple bufs = (buf,)
    cdef tuple locs = (loc,)
    _do_batch_discard_prefetch(bufs, locs, s)


cdef void _do_batch_discard_prefetch(tuple bufs, tuple locs, Stream s):
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        cdef Py_ssize_t n = len(bufs)
        cdef cydriver.CUstream hstream = as_cu(s._h_stream)
        cdef vector[cydriver.CUdeviceptr] ptrs
        cdef vector[size_t] sizes
        cdef vector[cydriver.CUmemLocation] loc_arr
        cdef vector[size_t] loc_indices
        ptrs.resize(n)
        sizes.resize(n)
        loc_arr.resize(n)
        loc_indices.resize(n)
        cdef Buffer buf
        cdef Py_ssize_t i
        for i in range(n):
            buf = <Buffer>bufs[i]
            ptrs[i] = as_cu(buf._h_ptr)
            sizes[i] = buf._size
            loc_arr[i] = _to_cumemlocation(locs[i])
            loc_indices[i] = <size_t>i
        with nogil:
            HANDLE_RETURN(cydriver.cuMemDiscardAndPrefetchBatchAsync(
                ptrs.data(), sizes.data(), <size_t>n,
                loc_arr.data(), loc_indices.data(), <size_t>n,
                0, hstream,
            ))
    ELSE:
        raise NotImplementedError(
            "discard_prefetch requires a CUDA 13 build of cuda.core"
        )
