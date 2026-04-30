# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from cuda.core._device import Device
from cuda.core._host import Host
from cuda.core._memory._buffer import Buffer
from cuda.core._utils.cuda_utils import driver, handle_return

if TYPE_CHECKING:
    from cuda.core._stream import Stream
    from cuda.core.graph import GraphBuilder


_INT_SIZE = 4


def _get_int_attr(buf: Buffer, attribute) -> int:
    return handle_return(driver.cuMemRangeGetAttribute(_INT_SIZE, attribute, buf.handle, buf.size))


class AccessedBySet:
    """Live driver-backed view of ``set_accessed_by`` advice for a managed buffer.

    Reads (``__contains__``, ``__iter__``, ``len(...)``) call
    ``cuMemRangeGetAttribute``; writes (``add``, ``discard``) call
    ``cuMemAdvise``. There is no in-memory mirror, so the view always
    reflects the current driver state.

    Note
    ----
    The driver's read-back path returns integer device ordinals (``-1`` for
    host); host NUMA distinctions applied via ``Host(numa_id=...)`` are not
    distinguishable from a generic ``Host()`` when iterating this set.
    """

    __slots__ = ("_buf",)

    def __init__(self, buf: ManagedBuffer):
        self._buf = buf

    def _query(self) -> list[Device | Host]:
        # Driver fills the array with device ordinals: device id, -1 = host,
        # -2 = empty slot. Size must accommodate every CUDA-visible device
        # plus a slot for the host. We use cuDeviceGetCount (driver-side) to
        # stay independent of NVML availability.
        num_devices = handle_return(driver.cuDeviceGetCount())
        n = num_devices + 1
        raw = handle_return(
            driver.cuMemRangeGetAttribute(
                n * _INT_SIZE,
                driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY,
                self._buf.handle,
                self._buf.size,
            )
        )
        result: list[Device | Host] = []
        for v in raw:
            if v == -2:  # CU_DEVICE_INVALID — empty slot
                continue
            result.append(Host() if v == -1 else Device(v))
        return result

    def __contains__(self, location) -> bool:
        return location in self._query()

    def __iter__(self):
        return iter(self._query())

    def __len__(self) -> int:
        return len(self._query())

    def __eq__(self, other) -> bool:
        if isinstance(other, AccessedBySet):
            return set(self._query()) == set(other._query())
        if isinstance(other, (set, frozenset)):
            return set(self._query()) == other
        return NotImplemented

    def __repr__(self) -> str:
        return f"AccessedBySet({set(self._query())!r})"

    def add(self, location: Device | Host) -> None:
        """Apply ``set_accessed_by`` advice for ``location``."""
        from cuda.core.utils import advise

        advise(self._buf, "set_accessed_by", location)

    def discard(self, location: Device | Host) -> None:
        """Apply ``unset_accessed_by`` advice for ``location``."""
        from cuda.core.utils import advise

        advise(self._buf, "unset_accessed_by", location)


class ManagedBuffer(Buffer):
    """Managed (unified) memory buffer with a property-style advice API.

    Returned by :meth:`ManagedMemoryResource.allocate`. Wrap an external
    managed-memory pointer with :meth:`ManagedBuffer.from_handle`.

    Examples
    --------
    >>> buf = mr.allocate(size)
    >>> buf.read_mostly = True
    >>> buf.preferred_location = Device(0)
    >>> buf.accessed_by.add(Device(1))
    >>> buf.prefetch(Device(0), stream=stream)

    Note
    ----
    The driver's read-back path for ``preferred_location`` and
    ``accessed_by`` returns integer device ordinals; host NUMA distinctions
    applied via ``Host(numa_id=...)`` collapse to a generic ``Host()`` when
    queried. Setters preserve full NUMA information when issuing advice.
    """

    @classmethod
    def from_handle(
        cls,
        ptr,
        size: int,
        mr=None,
        owner=None,
    ) -> ManagedBuffer:
        """Wrap an existing managed-memory pointer in a :class:`ManagedBuffer`."""
        return cls._init(ptr, size, mr=mr, owner=owner)

    @property
    def read_mostly(self) -> bool:
        """Whether ``set_read_mostly`` advice is currently applied to this range."""
        return _get_int_attr(self, driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY) != 0

    @read_mostly.setter
    def read_mostly(self, value: bool) -> None:
        from cuda.core.utils import advise

        advise(self, "set_read_mostly" if value else "unset_read_mostly")

    @property
    def preferred_location(self) -> Device | Host | None:
        """Currently applied ``set_preferred_location`` target, or ``None`` if unset."""
        # The legacy PREFERRED_LOCATION attribute returns a single int:
        # -2 = invalid (no preferred location), -1 = host, >=0 = device ordinal.
        # NUMA-specific preferences round-trip as a generic Host (CUDA driver
        # limitation of the legacy query path).
        loc_id = _get_int_attr(self, driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION)
        if loc_id == -2:
            return None
        if loc_id == -1:
            return Host()
        return Device(loc_id)

    @preferred_location.setter
    def preferred_location(self, value: Device | Host | None) -> None:
        from cuda.core.utils import advise

        if value is None:
            advise(self, "unset_preferred_location")
        else:
            advise(self, "set_preferred_location", value)

    @property
    def accessed_by(self) -> AccessedBySet:
        """Live set-like view of ``set_accessed_by`` locations."""
        return AccessedBySet(self)

    @accessed_by.setter
    def accessed_by(self, locations) -> None:
        # Diff against the current driver state and advise only the deltas.
        from cuda.core.utils import advise

        current = set(AccessedBySet(self))
        target = set(locations)
        for loc in current - target:
            advise(self, "unset_accessed_by", loc)
        for loc in target - current:
            advise(self, "set_accessed_by", loc)

    def prefetch(self, location: Device | Host | int, *, stream: Stream | GraphBuilder) -> None:
        """Prefetch this range to ``location`` on ``stream``."""
        from cuda.core.utils import prefetch as _prefetch

        _prefetch(self, location, stream=stream)

    def discard(self, *, stream: Stream | GraphBuilder) -> None:
        """Discard this range's resident pages on ``stream`` (CUDA 13+)."""
        from cuda.core.utils import discard as _discard

        _discard(self, stream=stream)

    def discard_prefetch(self, location: Device | Host | int, *, stream: Stream | GraphBuilder) -> None:
        """Discard this range and prefetch to ``location`` on ``stream`` (CUDA 13+)."""
        from cuda.core.utils import discard_prefetch as _discard_prefetch

        _discard_prefetch(self, location, stream=stream)
