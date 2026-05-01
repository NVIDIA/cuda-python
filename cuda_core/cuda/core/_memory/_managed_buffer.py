# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from cuda.core._device import Device
from cuda.core._host import Host
from cuda.core._memory._buffer import Buffer
from cuda.core._memory._managed_memory_ops import (
    _advise_one,
    _do_single_discard_prefetch_py,
    _do_single_discard_py,
    _do_single_prefetch_py,
)
from cuda.core._utils.cuda_utils import driver, handle_return

if TYPE_CHECKING:
    from cuda.core._memory._buffer import MemoryResource
    from cuda.core._stream import Stream
    from cuda.core.graph import GraphBuilder


_INT_SIZE = 4

# Enum aliases — referenced once per property write, so cache the lookup.
_ADV = driver.CUmem_advise
_SET_READ_MOSTLY = _ADV.CU_MEM_ADVISE_SET_READ_MOSTLY
_UNSET_READ_MOSTLY = _ADV.CU_MEM_ADVISE_UNSET_READ_MOSTLY
_SET_PREFERRED = _ADV.CU_MEM_ADVISE_SET_PREFERRED_LOCATION
_UNSET_PREFERRED = _ADV.CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION
_SET_ACCESSED_BY = _ADV.CU_MEM_ADVISE_SET_ACCESSED_BY
_UNSET_ACCESSED_BY = _ADV.CU_MEM_ADVISE_UNSET_ACCESSED_BY

_RANGE = driver.CUmem_range_attribute
_ATTR_READ_MOSTLY = _RANGE.CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY
_ATTR_PREFERRED = _RANGE.CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION
_ATTR_ACCESSED_BY = _RANGE.CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY


def _get_int_attr(buf: Buffer, attribute) -> int:
    return handle_return(driver.cuMemRangeGetAttribute(_INT_SIZE, attribute, buf.handle, buf.size))


def _query_accessed_by(buf: Buffer) -> list[Device | Host]:
    """Read the live ``CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY`` list.

    Driver fills an int32 array: device id, ``-1`` = host, ``-2`` = empty.
    Sized to ``cuDeviceGetCount() + 1`` (every visible device plus host).
    """
    num_devices = handle_return(driver.cuDeviceGetCount())
    n = num_devices + 1
    raw = handle_return(driver.cuMemRangeGetAttribute(n * _INT_SIZE, _ATTR_ACCESSED_BY, buf.handle, buf.size))
    return [Host() if v == -1 else Device(v) for v in raw if v != -2]


class AccessedBySet:
    """Live driver-backed view of ``set_accessed_by`` advice for a managed buffer.

    Reads (``__contains__``, ``__iter__``, ``len(...)``) call
    ``cuMemRangeGetAttribute``; writes (``add``, ``discard``) call
    ``cuMemAdvise``. There is no in-memory mirror, so the view always
    reflects the current driver state.

    Note
    ----
    The driver returns integer device ordinals (``-1`` for host); host
    NUMA distinctions applied via ``Host(numa_id=...)`` collapse to a
    generic ``Host()`` when iterating this set.
    """

    __slots__ = ("_buf",)

    def __init__(self, buf: ManagedBuffer):
        self._buf = buf

    def __contains__(self, location) -> bool:
        return location in _query_accessed_by(self._buf)

    def __iter__(self):
        return iter(_query_accessed_by(self._buf))

    def __len__(self) -> int:
        return len(_query_accessed_by(self._buf))

    def __eq__(self, other) -> bool:
        if isinstance(other, AccessedBySet):
            return set(_query_accessed_by(self._buf)) == set(_query_accessed_by(other._buf))
        if isinstance(other, (set, frozenset)):
            return set(_query_accessed_by(self._buf)) == other
        return NotImplemented

    def __repr__(self) -> str:
        return f"AccessedBySet({set(_query_accessed_by(self._buf))!r})"

    def add(self, location: Device | Host) -> None:
        """Apply ``set_accessed_by`` advice for ``location``."""
        _advise_one(self._buf, _SET_ACCESSED_BY, location)

    def discard(self, location: Device | Host) -> None:
        """Apply ``unset_accessed_by`` advice for ``location``."""
        _advise_one(self._buf, _UNSET_ACCESSED_BY, location)


class ManagedBuffer(Buffer):
    """Managed (unified) memory buffer with a property-style advice API.

    Returned by :meth:`ManagedMemoryResource.allocate`, or wrap an
    existing managed-memory pointer with :meth:`ManagedBuffer.from_handle`.

    Examples
    --------
    >>> buf = mr.allocate(size)
    >>> buf.read_mostly = True
    >>> buf.preferred_location = Device(0)
    >>> buf.accessed_by.add(Device(1))
    >>> buf.prefetch(Device(0), stream=stream)

    Note
    ----
    The legacy ``cuMemRangeGetAttribute`` query path returns integer
    device ordinals, so ``Host(numa_id=...)`` collapses to ``Host()``
    on read-back. Setters preserve full NUMA information when issuing
    advice.
    """

    @classmethod
    def from_handle(
        cls,
        ptr,
        size: int,
        mr: MemoryResource | None = None,
        owner: object | None = None,
    ) -> ManagedBuffer:
        """Wrap an existing managed-memory pointer in a :class:`ManagedBuffer`.

        Use this when you have an externally-allocated managed pointer
        and want the property-style advice API (:attr:`read_mostly`,
        :attr:`preferred_location`, :attr:`accessed_by`).

        Parameters
        ----------
        ptr : :obj:`~_memory.DevicePointerT`
            Pointer to a managed allocation.
        size : int
            Allocation size in bytes.
        mr : :obj:`~_memory.MemoryResource`, optional
            Memory resource that owns ``ptr``. When provided, its
            ``deallocate`` is called when the buffer is closed.
        owner : object, optional
            An object that keeps the underlying allocation alive.
            ``owner`` and ``mr`` cannot both be specified.
        """
        return cls._init(ptr, size, mr=mr, owner=owner)

    @property
    def read_mostly(self) -> bool:
        """Whether ``set_read_mostly`` advice is currently applied."""
        return _get_int_attr(self, _ATTR_READ_MOSTLY) != 0

    @read_mostly.setter
    def read_mostly(self, value: bool) -> None:
        _advise_one(self, _SET_READ_MOSTLY if value else _UNSET_READ_MOSTLY, None)

    @property
    def preferred_location(self) -> Device | Host | None:
        """Currently applied ``set_preferred_location`` target, or ``None``.

        .. note::
           The legacy ``CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION`` carries
           only a device ordinal (or ``-1`` for host) and cannot represent
           a specific NUMA node. As a result, ``Host(numa_id=N)`` set via
           the setter currently round-trips back as ``Host()``. The CUDA 13
           driver added ``..._PREFERRED_LOCATION_TYPE`` / ``..._ID`` for
           full ``CUmemLocation`` round-trip, but ``cuda.bindings`` does
           not yet expose these via ``cuMemRangeGetAttribute``; once it
           does, this getter will be upgraded.
        """
        loc_id = _get_int_attr(self, _ATTR_PREFERRED)
        if loc_id == -2:
            return None
        if loc_id == -1:
            return Host()
        return Device(loc_id)

    @preferred_location.setter
    def preferred_location(self, value: Device | Host | None) -> None:
        if value is None:
            _advise_one(self, _UNSET_PREFERRED, None)
        else:
            _advise_one(self, _SET_PREFERRED, value)

    @property
    def accessed_by(self) -> AccessedBySet:
        """Live set-like view of ``set_accessed_by`` locations."""
        return AccessedBySet(self)

    @accessed_by.setter
    def accessed_by(self, locations) -> None:
        # Diff against the current driver state and advise only the deltas.
        current = set(_query_accessed_by(self))
        target = set(locations)
        for loc in current - target:
            _advise_one(self, _UNSET_ACCESSED_BY, loc)
        for loc in target - current:
            _advise_one(self, _SET_ACCESSED_BY, loc)

    def prefetch(self, location: Device | Host, *, stream: Stream | GraphBuilder) -> None:
        """Prefetch this range to ``location`` on ``stream``."""
        _do_single_prefetch_py(self, location, stream)

    def discard(self, *, stream: Stream | GraphBuilder) -> None:
        """Discard this range's resident pages on ``stream`` (CUDA 13+)."""
        _do_single_discard_py(self, stream)

    def discard_prefetch(self, location: Device | Host, *, stream: Stream | GraphBuilder) -> None:
        """Discard this range and prefetch to ``location`` on ``stream`` (CUDA 13+)."""
        _do_single_discard_prefetch_py(self, location, stream)
