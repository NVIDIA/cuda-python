# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
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

    numa_id: int | None = None
    is_numa_current: bool = False

    def __post_init__(self) -> None:
        if self.is_numa_current and self.numa_id is not None:
            raise ValueError("Host.numa_current() cannot have an explicit numa_id")
        if self.numa_id is not None and (not isinstance(self.numa_id, int) or self.numa_id < 0):
            raise ValueError(f"numa_id must be a non-negative int, got {self.numa_id!r}")

    @classmethod
    def numa_current(cls) -> Host:
        """Construct a ``Host`` referring to the calling thread's NUMA node."""
        return cls(is_numa_current=True)

    def __repr__(self) -> str:
        if self.is_numa_current:
            return "Host.numa_current()"
        if self.numa_id is None:
            return "Host()"
        return f"Host(numa_id={self.numa_id})"
