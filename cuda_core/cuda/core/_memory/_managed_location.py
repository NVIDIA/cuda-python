# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

_LocationKind = Literal["device", "host", "host_numa", "host_numa_current"]


@dataclass(frozen=True)
class _LocSpec:
    """Internal location record produced by :func:`_coerce_location`.

    Carries the discriminator (``kind``) and the integer payload (``id``)
    that the Cython layer in ``_managed_memory_ops.pyx`` consumes when
    building ``CUmemLocation`` structs (CUDA 13+) or legacy device
    ordinals (CUDA 12).
    """

    kind: _LocationKind
    id: int = 0


def _coerce_location(value, *, allow_none: bool = False) -> _LocSpec | None:
    """Coerce :class:`Device` / :class:`Host` / int / ``None`` to ``_LocSpec``.

    Maps int ``-1`` to host and other non-negative ints to that device
    ordinal. ``Host()``, ``Host(numa_id=N)``, and ``Host.numa_current()``
    map to the corresponding NUMA-aware kinds.
    """
    # Local imports to avoid import cycles (Device pulls in CUDA init).
    from cuda.core._device import Device
    from cuda.core._host import Host

    if isinstance(value, _LocSpec):
        return value
    if isinstance(value, Device):
        return _LocSpec(kind="device", id=value.device_id)
    if isinstance(value, Host):
        if value.is_numa_current:
            return _LocSpec(kind="host_numa_current")
        if value.numa_id is not None:
            return _LocSpec(kind="host_numa", id=value.numa_id)
        return _LocSpec(kind="host")
    if value is None:
        if allow_none:
            return None
        raise ValueError("location is required")
    raise TypeError(f"location must be a Device, Host, or None; got {type(value).__name__}")
