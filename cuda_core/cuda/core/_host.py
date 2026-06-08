# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from typing import ClassVar


class Host:
    """Host (CPU) location for managed-memory operations.

    Use one of the following forms:

    * ``Host()`` — generic host (any NUMA node).
    * ``Host(numa_id=N)`` — specific NUMA node ``N``.
    * ``Host.numa_current()`` or ``Host(is_numa_current=True)`` — NUMA node
      of the calling thread. ``numa_id`` and ``is_numa_current`` are
      mutually exclusive.

    ``Host`` is the symmetric counterpart of :class:`~cuda.core.Device`
    for managed-memory `prefetch`, `advise`, and `discard_prefetch`
    targets. Pass either a ``Device`` or a ``Host`` to those operations
    and to ``ManagedBuffer.preferred_location`` / ``accessed_by``.

    ``Host`` is a singleton class, mirroring :class:`~cuda.core.Device`:
    constructor calls with the same arguments return the same instance,
    so ``Host() is Host()`` and ``Host(numa_id=1) is Host(numa_id=1)``.
    ``Host.numa_current()`` returns its own singleton, distinct from
    ``Host()`` because it represents a thread-relative location rather
    than a fixed one.
    """

    __slots__ = ("__weakref__", "_is_numa_current", "_numa_id")

    _numa_id: int | None
    _is_numa_current: bool

    # Singleton cache keyed by (numa_id, is_numa_current).
    _instances: ClassVar[dict[tuple[int | None, bool], Host]] = {}
    _instances_lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls, numa_id: int | None = None, *, is_numa_current: bool = False) -> Host:
        if is_numa_current and numa_id is not None:
            raise ValueError("numa_id and is_numa_current are mutually exclusive")
        if numa_id is not None and (isinstance(numa_id, bool) or not isinstance(numa_id, int) or numa_id < 0):
            raise ValueError(f"numa_id must be a non-negative int, got {numa_id!r}")
        return cls._get_or_create(numa_id, is_numa_current)

    @classmethod
    def _get_or_create(cls, numa_id: int | None, is_numa_current: bool) -> Host:
        key = (numa_id, is_numa_current)
        cache = cls._instances
        inst = cache.get(key)
        if inst is not None:
            return inst
        with cls._instances_lock:
            inst = cache.get(key)
            if inst is None:
                inst = object.__new__(cls)
                inst._numa_id = numa_id
                inst._is_numa_current = is_numa_current
                cache[key] = inst
            return inst

    @property
    def numa_id(self) -> int | None:
        """NUMA node ID, or ``None`` if not pinned to a specific NUMA node."""
        return self._numa_id

    @property
    def is_numa_current(self) -> bool:
        """``True`` if this ``Host`` represents the calling thread's NUMA node (constructed via :meth:`numa_current`)."""
        return self._is_numa_current

    @classmethod
    def numa_current(cls) -> Host:
        """Construct a ``Host`` referring to the calling thread's NUMA node."""
        return cls(is_numa_current=True)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Host):
            return NotImplemented
        return self is other

    def __hash__(self) -> int:
        return hash((Host, self._numa_id, self._is_numa_current))

    def __reduce__(self):
        if self._is_numa_current:
            return (_reconstruct_numa_current, ())
        return (Host, (self._numa_id,))

    def __repr__(self) -> str:
        if self.is_numa_current:
            return "Host.numa_current()"
        if self.numa_id is None:
            return "Host()"
        return f"Host(numa_id={self.numa_id})"


def _reconstruct_numa_current() -> Host:
    return Host.numa_current()
