# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from cuda.core._device import Device
from cuda.core._host import Host
from cuda.core._utils.cuda_utils import driver
from cuda.core._utils.pycompat import StrEnum
from cuda.core._utils.version import binding_version

__all__ = ["CopyOptions", "MemcpyOverlapMode", "MemcpySrcAccessOrder"]


class MemcpySrcAccessOrder(StrEnum):
    """Source access order hint for batched memcpy operations.

    Maps to ``CUmemcpySrcAccessOrder``.

    ``STREAM``
        Source reads follow stream order. Earlier stream work may still be
        accessing the source when the copy is enqueued.
    ``DURING_API_CALL``
        The driver may read the source out of stream order, but all reads
        are complete before :func:`copy_batch` returns. No earlier stream
        work may be accessing the source at the time of the call.
    ``ANY``
        The driver may read the source after the call returns. The caller
        must keep the source unchanged until the copy completes in stream
        order. No earlier stream work may be accessing the source.
    """

    STREAM = "stream"
    DURING_API_CALL = "during_api_call"
    ANY = "any"


class MemcpyOverlapMode(StrEnum):
    """Overlap mode hint for batched memcpy operations.

    Maps to ``CUmemcpyFlags``.

    ``DEFAULT``
        No overlap preference; the driver uses its default scheduling.
    ``PREFER_OVERLAP_WITH_COMPUTE``
        Hint that the copy should preferably overlap with concurrent
        compute work. This is advisory and may be ignored depending on
        the platform and copy parameters.
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
        Hint for the source memory location. Honored only for managed
        memory on devices with concurrent managed access and for
        system-allocated pageable memory on devices with pageable memory
        access; ignored for all other memory types. Does not prefetch
        memory and does not set persistent memory advice.
        ``None`` means no hint.
    dst_location_hint : :class:`cuda.core.Device` | :class:`cuda.core.Host` | None
        Hint for the destination memory location. Same semantics and
        restrictions as ``src_location_hint``. ``None`` means no hint.
    overlap_mode : :class:`MemcpyOverlapMode` or str
        Hint requesting that the copy overlap with concurrent compute work.
        This is advisory; it has an effect only on devices that support it.
        Default is ``"default"``.
    """

    src_access_order: MemcpySrcAccessOrder | str = "stream"
    src_location_hint: Device | Host | None = None
    dst_location_hint: Device | Host | None = None
    overlap_mode: MemcpyOverlapMode | str = "default"

    def __post_init__(self):
        # Frozen, unlike the other *Options dataclasses in cuda.core, because
        # the batched-API contract agreed in NVIDIA/cuda-python#1775 specifies
        # immutable per-call options:
        # https://github.com/NVIDIA/cuda-python/pull/1775#issuecomment-4355502334
        #
        # Normalizing str -> StrEnum therefore has to go through
        # object.__setattr__; a plain assignment would raise
        # FrozenInstanceError. Done here rather than at use so that a typo
        # fails at construction and the field always holds the enum.
        if not isinstance(self.src_access_order, MemcpySrcAccessOrder):
            try:
                object.__setattr__(
                    self,
                    "src_access_order",
                    MemcpySrcAccessOrder(self.src_access_order),
                )
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid src_access_order: {self.src_access_order!r}") from exc
        if not isinstance(self.overlap_mode, MemcpyOverlapMode):
            try:
                object.__setattr__(
                    self,
                    "overlap_mode",
                    MemcpyOverlapMode(self.overlap_mode),
                )
            except (ValueError, TypeError) as exc:
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


_CUDA13_REQUIRED = "copy attributes require cuda.bindings 13.0 or newer"

# CUmemcpySrcAccessOrder and CUmemcpyFlags are exposed by cuda.bindings 13.0+,
# so these maps are empty when it is older. Nothing reaches them there:
# copy_batch refuses non-default CopyOptions when the batched entry point is
# unavailable.
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


def _reject_unsupported_during_api_call(
    src_access_order: MemcpySrcAccessOrder, requirement: str, *, index: int | None = None
) -> None:
    """Raise if ``src_access_order`` is DURING_API_CALL but the native attributes
    path (``cuMemcpyWithAttributesAsync`` / ``cuMemcpyBatchAsync``) is unavailable.

    STREAM and ANY never promise access sooner than stream order, so a plain
    ``cuMemcpyAsync`` fallback satisfies them; DURING_API_CALL specifically
    promises all source reads complete before the call returns, which
    ``cuMemcpyAsync`` cannot provide (it reads the source in stream order
    only). Silently downgrading that guarantee would let a caller reuse or
    overwrite the source buffer before the real, stream-ordered read
    happens: a silent data race, not a missed optimization. ``requirement``
    names what the native path needs and why it is unavailable here;
    ``index`` identifies the offending copy within a batch.

    Internal, but deliberately importable: shared between the per-buffer and
    batched fallback paths so both raise identically, and directly testable
    without needing an actual old driver/bindings install.
    """
    if src_access_order != MemcpySrcAccessOrder.DURING_API_CALL:
        return
    where = f" at index {index}" if index is not None else ""
    raise RuntimeError(
        f"src_access_order=DURING_API_CALL{where} requires {requirement}. A "
        "plain cuMemcpyAsync fallback reads the source in stream order only, "
        "which would silently violate the guarantee that all source reads "
        "complete before the call returns, letting the caller reuse the "
        "source buffer before the real (stream-ordered) read happens. Use "
        "src_access_order=STREAM or ANY, or omit options, if that works for "
        "your use case."
    )


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
