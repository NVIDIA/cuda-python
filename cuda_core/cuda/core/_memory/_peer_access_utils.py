# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class PeerAccessPlan:
    """Normalized peer-access target state and the driver updates it requires."""

    target_ids: tuple[int, ...]
    to_add: tuple[int, ...]
    to_remove: tuple[int, ...]


def normalize_peer_access_targets(
    owner_device_id: int,
    requested_devices: Iterable[object],
    *,
    resolve_device_id: Callable[[object], int],
) -> tuple[int, ...]:
    """Return sorted, unique peer device IDs, excluding the owner device."""

    target_ids = {resolve_device_id(device) for device in requested_devices}
    target_ids.discard(owner_device_id)
    return tuple(sorted(target_ids))


def plan_peer_access_update(
    owner_device_id: int,
    current_peer_ids: Iterable[int],
    requested_devices: Iterable[object],
    *,
    resolve_device_id: Callable[[object], int],
    can_access_peer: Callable[[int], bool],
) -> PeerAccessPlan:
    """Compute the peer-access target state and add/remove deltas."""

    target_ids = normalize_peer_access_targets(
        owner_device_id,
        requested_devices,
        resolve_device_id=resolve_device_id,
    )
    bad = tuple(dev_id for dev_id in target_ids if not can_access_peer(dev_id))
    if bad:
        bad_ids = ", ".join(str(dev_id) for dev_id in bad)
        raise ValueError(f"Device {owner_device_id} cannot access peer(s): {bad_ids}")

    current_ids = set(current_peer_ids)
    target_id_set = set(target_ids)
    return PeerAccessPlan(
        target_ids=target_ids,
        to_add=tuple(sorted(target_id_set - current_ids)),
        to_remove=tuple(sorted(current_ids - target_id_set)),
    )
