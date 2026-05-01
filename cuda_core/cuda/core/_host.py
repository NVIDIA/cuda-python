# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


class Host:
    """Host (CPU) location for managed-memory operations.

    Use one of the three forms:

    * ``Host()`` — generic host (any NUMA node).
    * ``Host(numa_id=N)`` — specific NUMA node ``N``.
    * ``Host.numa_current()`` — NUMA node of the calling thread.

    ``Host`` is the symmetric counterpart of :class:`~cuda.core.Device`
    for managed-memory `prefetch`, `advise`, and `discard_prefetch`
    targets. Pass either a ``Device`` or a ``Host`` to those operations
    and to ``ManagedBuffer.preferred_location`` / ``accessed_by``.
    """

    __slots__ = ("_is_numa_current", "_numa_id")

    def __init__(self, numa_id: int | None = None) -> None:
        if numa_id is not None and (not isinstance(numa_id, int) or numa_id < 0):
            raise ValueError(f"numa_id must be a non-negative int, got {numa_id!r}")
        object.__setattr__(self, "_numa_id", numa_id)
        object.__setattr__(self, "_is_numa_current", False)

    @property
    def numa_id(self) -> int | None:
        return self._numa_id

    @property
    def is_numa_current(self) -> bool:
        return self._is_numa_current

    @classmethod
    def numa_current(cls) -> Host:
        """Construct a ``Host`` referring to the calling thread's NUMA node."""
        h = cls()
        object.__setattr__(h, "_is_numa_current", True)
        return h

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable; cannot set {name!r}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Host):
            return NotImplemented
        return self._numa_id == other._numa_id and self._is_numa_current == other._is_numa_current

    def __hash__(self) -> int:
        return hash((Host, self._numa_id, self._is_numa_current))

    def __repr__(self) -> str:
        if self.is_numa_current:
            return "Host.numa_current()"
        if self.numa_id is None:
            return "Host()"
        return f"Host(numa_id={self.numa_id})"
