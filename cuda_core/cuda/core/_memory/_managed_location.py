# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

_VALID_KINDS = ("device", "host", "host_numa", "host_numa_current")
LocationKind = Literal["device", "host", "host_numa", "host_numa_current"]


@dataclass(frozen=True)
class Location:
    """Typed managed-memory location.

    Use the classmethod constructors (``device``, ``host``, ``host_numa``,
    ``host_numa_current``) rather than constructing directly.
    """

    kind: LocationKind
    id: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"kind must be one of {_VALID_KINDS!r}, got {self.kind!r}")
        if self.kind == "device":
            if not isinstance(self.id, int) or self.id < 0:
                raise ValueError("device id must be >= 0")
        elif self.kind == "host_numa":
            if not isinstance(self.id, int) or self.id < 0:
                raise ValueError("host_numa id must be >= 0")
        elif self.kind in ("host", "host_numa_current"):
            if self.id is not None:
                raise ValueError(f"{self.kind} location must have id=None")

    @classmethod
    def device(cls, device_id: int) -> "Location":
        return cls(kind="device", id=device_id)

    @classmethod
    def host(cls) -> "Location":
        return cls(kind="host", id=None)

    @classmethod
    def host_numa(cls, numa_id: int) -> "Location":
        return cls(kind="host_numa", id=numa_id)

    @classmethod
    def host_numa_current(cls) -> "Location":
        return cls(kind="host_numa_current", id=None)
