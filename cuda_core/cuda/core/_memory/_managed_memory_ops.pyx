# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cuda.core._memory._buffer cimport Buffer, _init_mem_attrs
from cuda.core._stream cimport Stream, Stream_accept

from cuda.core._utils.cuda_utils import driver, get_binding_version, handle_return
from cuda.core._device import Device


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
        _V2_BINDINGS = 1 if get_binding_version() >= (13, 0) else 0
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


cdef void _require_managed_discard_prefetch_support(str what):
    global _DISCARD_PREFETCH_SUPPORTED
    if _DISCARD_PREFETCH_SUPPORTED < 0:
        _DISCARD_PREFETCH_SUPPORTED = 1 if hasattr(driver, "cuMemDiscardAndPrefetchBatchAsync") else 0
    if not _DISCARD_PREFETCH_SUPPORTED:
        raise RuntimeError(
            f"{what} requires cuda.bindings support for cuMemDiscardAndPrefetchBatchAsync"
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
    target: Buffer,
    location: Device | int | None = None,
    *,
    stream: Stream | GraphBuilder,
    location_type: str | None = None,
):
    """Prefetch a managed-memory allocation range to a target location.

    Parameters
    ----------
    target : :class:`Buffer`
        Managed allocation to operate on.
    location : :obj:`~_device.Device` | int | None, optional
        Target location. When ``location_type`` is ``None``, values are
        interpreted as a device ordinal, ``-1`` for host, or ``None``.
        A location is required for prefetch.
    stream : :obj:`~_stream.Stream` | :obj:`~_graph.GraphBuilder`
        Keyword argument specifying the stream for the asynchronous prefetch.
    location_type : str | None, optional
        Explicit location kind. Supported values are ``"device"``, ``"host"``,
        ``"host_numa"``, and ``"host_numa_current"``.
    """
    if not isinstance(target, Buffer):
        raise TypeError(f"prefetch target must be a Buffer, got {type(target).__name__}")
    cdef Buffer buf = <Buffer>target
    _require_managed_buffer(buf, "prefetch")
    cdef Stream s = Stream_accept(stream)
    cdef object ptr = buf.handle
    cdef size_t nbytes = buf._size

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
