# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from cuda.core._utils.version import binding_version, driver_version

if TYPE_CHECKING:
    from cuda.core._device import Device
    from cuda.core._host import Host

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


def _reject_numa_host_on_cuda12(spec: _LocSpec) -> None:
    """Reject NUMA-host kinds on CUDA 12 builds at the public boundary.

    The CUDA 12 ``cuMemPrefetchAsync`` / ``cuMemAdvise`` ABI takes a
    plain device ordinal (``-1`` for host), so it cannot represent a
    specific host NUMA node. Rather than letting the operation fail
    deep inside the Cython layer with ``RuntimeError``, raise a
    ``TypeError`` at the call boundary with actionable wording.
    """
    # The host-NUMA kinds map to CU_MEM_LOCATION_TYPE_HOST_NUMA{,_CURRENT},
    # both added in CUDA 13. Require both bindings and the runtime driver to
    # be 13.0+; bindings-only is insufficient (PR #2054 / #2064 precedent).
    if binding_version() >= (13, 0, 0) and driver_version() >= (13, 0, 0):
        return
    if spec.kind in ("host_numa", "host_numa_current"):
        raise TypeError(
            "Host(numa_id=...) / Host.numa_current() require both cuda-bindings 13.0+ "
            "and a CUDA 13+ runtime driver; use Host() instead"
        )


def _coerce_location(value: Device | Host | None, *, allow_none: bool = False) -> _LocSpec | None:
    """Coerce :class:`Device` / :class:`Host` / ``None`` to ``_LocSpec``.

    ``Host()``, ``Host(numa_id=N)``, and ``Host.numa_current()`` map to
    the corresponding NUMA-aware kinds. On a CUDA 12 build of
    ``cuda.core``, NUMA-host inputs are rejected with ``TypeError``
    because the legacy ABI cannot represent them.
    """
    # Local imports to avoid import cycles (Device pulls in CUDA init).
    from cuda.core._device import Device
    from cuda.core._host import Host

    if isinstance(value, _LocSpec):
        _reject_numa_host_on_cuda12(value)
        return value
    if isinstance(value, Device):
        return _LocSpec(kind="device", id=value.device_id)
    if isinstance(value, Host):
        if value.is_numa_current:
            spec = _LocSpec(kind="host_numa_current")
            _reject_numa_host_on_cuda12(spec)
            return spec
        if value.numa_id is not None:
            spec = _LocSpec(kind="host_numa", id=value.numa_id)
            _reject_numa_host_on_cuda12(spec)
            return spec
        return _LocSpec(kind="host")
    if value is None:
        if allow_none:
            return None
        raise ValueError("location is required")
    raise TypeError(f"location must be a Device, Host, or None; got {type(value).__name__}")
