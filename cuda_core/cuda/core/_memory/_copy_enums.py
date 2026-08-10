# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING

from cuda.core._utils.cuda_utils import driver
from cuda.core._utils.pycompat import StrEnum
from cuda.core._utils.version import binding_version

if TYPE_CHECKING:
    from cuda.core._device import Device
    from cuda.core._host import Host


__all__ = ["CopyOptions", "MemcpyOverlapMode", "MemcpySrcAccessOrder"]


class MemcpySrcAccessOrder(StrEnum):
    """Source access order hint for batched memcpy operations.

    Maps to ``CUmemcpySrcAccessOrder``. The ``INVALID`` and ``MAX``
    sentinel values from the driver enum are excluded from the public
    Python surface.
    """

    STREAM = "stream"
    DURING_API_CALL = "during_api_call"
    ANY = "any"


class MemcpyOverlapMode(StrEnum):
    """Overlap mode hint for batched memcpy operations.

    Maps to ``CUmemcpyFlags``. Renamed from "flags" to "overlap_mode"
    for clarity; the only non-default flag is CE/compute overlap
    (Tegra).
    """

    DEFAULT = "default"
    PREFER_OVERLAP_WITH_COMPUTE = "prefer_overlap_with_compute"


@dataclasses.dataclass(frozen=True)
class CopyOptions:
    """Attribute bundle for a single copy within a batched memcpy.

    Parameters
    ----------
    src_access_order : :class:`MemcpySrcAccessOrder` or str
        Hint describing how the source will be accessed.
        Default is ``"stream"`` (stream-ordered access).
    src_location_hint : :class:`cuda.core.Device` | :class:`cuda.core.Host` | None
        Hint for the source memory location. ``None`` means no hint.
    dst_location_hint : :class:`cuda.core.Device` | :class:`cuda.core.Host` | None
        Hint for the destination memory location. ``None`` means no hint.
    overlap_mode : :class:`MemcpyOverlapMode` or str
        Hint for copy-engine / compute overlap.  Only meaningful on
        integrated (Tegra) GPUs; on discrete GPUs the driver silently
        ignores it and a :class:`UserWarning` is emitted.
        Default is ``"default"``.
    """

    src_access_order: MemcpySrcAccessOrder | str = "stream"
    src_location_hint: Device | Host | None = None
    dst_location_hint: Device | Host | None = None
    overlap_mode: MemcpyOverlapMode | str = "default"

    def __post_init__(self):
        # Validate enum fields while still in __init__ (frozen dataclass).
        # Use __setattr__ because fields are frozen.
        if isinstance(self.src_access_order, str):
            try:
                object.__setattr__(
                    self,
                    "src_access_order",
                    MemcpySrcAccessOrder(self.src_access_order),
                )
            except ValueError as exc:
                raise ValueError(f"invalid src_access_order: {self.src_access_order!r}") from exc
        if isinstance(self.overlap_mode, str):
            try:
                object.__setattr__(
                    self,
                    "overlap_mode",
                    MemcpyOverlapMode(self.overlap_mode),
                )
            except ValueError as exc:
                raise ValueError(f"invalid overlap_mode: {self.overlap_mode!r}") from exc

    def _to_driver_enum(self) -> int:
        """Return the driver CUmemcpySrcAccessOrder value."""
        if not _SRC_ACCESS_ORDER_TO_DRIVER:
            raise NotImplementedError(_CUDA13_REQUIRED)
        return _SRC_ACCESS_ORDER_TO_DRIVER[MemcpySrcAccessOrder(self.src_access_order)]

    def _to_driver_flags(self) -> int:
        """Return the driver CUmemcpyFlags value."""
        if not _OVERLAP_MODE_TO_DRIVER:
            raise NotImplementedError(_CUDA13_REQUIRED)
        return _OVERLAP_MODE_TO_DRIVER[MemcpyOverlapMode(self.overlap_mode)]


_CUDA13_REQUIRED = "copy attributes require a CUDA 13 build of cuda-bindings"

# CUmemcpySrcAccessOrder and CUmemcpyFlags are CUDA 13 additions, so these
# maps are empty on a CUDA 12 build. Nothing reaches them there: copy_batch
# refuses non-default CopyOptions when the batched entry point is absent.
#
# Keyed by ``str``: under ``python_version = "3.10"`` mypy resolves StrEnum to
# the unstubbed backports shim and so infers the members as plain ``str``.
# StrEnum members are ``str`` instances, so this holds on every version. The
# values are wrapped in ``int()`` because the driver enums are untyped.
_SRC_ACCESS_ORDER_TO_DRIVER: dict[str, int]
_OVERLAP_MODE_TO_DRIVER: dict[str, int]

if binding_version() >= (13, 0, 0):
    _src_order = driver.CUmemcpySrcAccessOrder
    _flags = driver.CUmemcpyFlags
    _SRC_ACCESS_ORDER_TO_DRIVER = {
        MemcpySrcAccessOrder.STREAM: int(_src_order.CU_MEMCPY_SRC_ACCESS_ORDER_STREAM),
        MemcpySrcAccessOrder.DURING_API_CALL: int(_src_order.CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL),
        MemcpySrcAccessOrder.ANY: int(_src_order.CU_MEMCPY_SRC_ACCESS_ORDER_ANY),
    }
    _OVERLAP_MODE_TO_DRIVER = {
        MemcpyOverlapMode.DEFAULT: int(_flags.CU_MEMCPY_FLAG_DEFAULT),
        MemcpyOverlapMode.PREFER_OVERLAP_WITH_COMPUTE: int(_flags.CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE),
    }
    del _src_order, _flags
else:
    _SRC_ACCESS_ORDER_TO_DRIVER = {}
    _OVERLAP_MODE_TO_DRIVER = {}


def _attr_run_starts(attrs: Sequence[CopyOptions]) -> list[int]:
    """Return the start index of each maximal run of equal attributes.

    This mirrors the ``attrsIdxs`` indirection that ``cuMemcpyBatchAsync``
    expects: ``attrs[k]`` applies to the copies in
    ``[starts[k], starts[k + 1])``. Collapsing equal neighbours means a
    broadcast attribute is passed to the driver once (``numAttrs == 1``)
    rather than repeated per copy.
    """
    starts: list[int] = []
    prev: CopyOptions | None = None
    for i, attr in enumerate(attrs):
        if i == 0 or attr != prev:
            starts.append(i)
        prev = attr
    return starts
