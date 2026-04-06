# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cuda.core._memory._peer_access_utils import PeerAccessPlan, plan_peer_access_update


@dataclass(frozen=True)
class DummyDevice:
    device_id: int


def _resolve_device_id(device) -> int:
    if isinstance(device, DummyDevice):
        return device.device_id
    return int(device)


def test_plan_peer_access_update_normalizes_requests():
    plan = plan_peer_access_update(
        owner_device_id=1,
        current_peer_ids=(),
        requested_devices=[1, DummyDevice(3), 2, DummyDevice(2), 3],
        resolve_device_id=_resolve_device_id,
        can_access_peer=lambda _device_id: True,
    )

    assert plan == PeerAccessPlan(
        target_ids=(2, 3),
        to_add=(2, 3),
        to_remove=(),
    )


def test_plan_peer_access_update_rejects_inaccessible_peers():
    with pytest.raises(ValueError, match=r"Device 0 cannot access peer\(s\): 2, 4"):
        plan_peer_access_update(
            owner_device_id=0,
            current_peer_ids=(1,),
            requested_devices=[4, 0, DummyDevice(2), 1],
            resolve_device_id=_resolve_device_id,
            can_access_peer=lambda device_id: device_id == 1,
        )


def test_plan_peer_access_update_covers_all_state_transitions():
    states = [(), (1,), (2,), (1, 2)]
    for current_state in states:
        for requested_state in states:
            plan = plan_peer_access_update(
                owner_device_id=0,
                current_peer_ids=current_state,
                requested_devices=requested_state,
                resolve_device_id=_resolve_device_id,
                can_access_peer=lambda device_id: device_id in {1, 2},
            )

            assert plan == PeerAccessPlan(
                target_ids=requested_state,
                to_add=tuple(sorted(set(requested_state) - set(current_state))),
                to_remove=tuple(sorted(set(current_state) - set(requested_state))),
            )
