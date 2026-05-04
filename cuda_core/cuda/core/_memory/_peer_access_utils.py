# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Iterable, MutableSet
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuda.core._device import Device
    from cuda.core._memory._device_memory_resource import DeviceMemoryResource


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


def _resolve_peer_device_id(value: Device | int) -> int:
    """Coerce ``Device | int`` into a device-ordinal int."""
    from cuda.core._device import Device

    return Device(value).device_id


class PeerAccessibleBySetProxy(MutableSet):
    """Live driver-backed view of the peer devices granted access to a memory pool.

    Reads (``__contains__``, ``__iter__``, ``len(...)``) call ``cuMemPoolGetAccess``;
    writes (``add``, ``discard``, and bulk ops) call ``cuMemPoolSetAccess``. There
    is no in-memory mirror, so the view always reflects the current driver state
    and stays consistent across multiple wrappers around the same pool.

    Iteration yields :class:`~cuda.core.Device` objects. ``add``, ``discard``, and
    ``__contains__`` accept either a :class:`~cuda.core.Device` or a device-ordinal
    ``int``; the owner device is silently ignored when supplied.

    All bulk operations (``update``, ``|=``, ``&=``, ``-=``, ``^=``, ``clear``)
    issue exactly one ``cuMemPoolSetAccess`` call. This matters: peer-access
    transitions can take seconds per pool because every existing memory mapping
    is updated, so coalescing into a single driver call lets the toolkit handle
    the mappings in parallel.
    """

    __slots__ = ("_mr",)

    def __init__(self, mr: DeviceMemoryResource):
        self._mr = mr

    @classmethod
    def _from_iterable(cls, it):
        # Binary set operators (&, |, -, ^) collect their result through
        # _from_iterable. Returning a plain set lets the user reason about
        # the result independently of any pool's driver state.
        return set(it)

    # --- abstract MutableSet methods ---

    def __contains__(self, value) -> bool:
        try:
            dev_id = _resolve_peer_device_id(value)
        except (TypeError, ValueError):
            return False
        if dev_id == self._mr._dev_id:
            return False
        return self._mr._peer_access_includes(dev_id)

    def __iter__(self):
        from cuda.core._device import Device

        return iter(Device(dev_id) for dev_id in self._mr._query_peer_access_ids())

    def __len__(self) -> int:
        return len(self._mr._query_peer_access_ids())

    def add(self, value: Device | int) -> None:
        """Grant peer access from ``value`` to allocations in this pool."""
        self._apply([value], ())

    def discard(self, value: Device | int) -> None:
        """Revoke peer access from ``value`` to allocations in this pool."""
        try:
            dev_id = _resolve_peer_device_id(value)
        except (TypeError, ValueError):
            return
        self._apply((), [dev_id])

    # --- bulk overrides: one driver call per op ---

    def clear(self) -> None:
        """Revoke all peer access in a single driver call."""
        self._apply((), self._mr._query_peer_access_ids())

    def update(self, *others) -> None:
        """Grant peer access to every device in ``others`` in one driver call."""
        to_add = []
        for other in others:
            to_add.extend(other)
        if to_add:
            self._apply(to_add, ())

    def difference_update(self, *others) -> None:
        """Revoke peer access for every device in ``others`` in one driver call."""
        revoke_ids: set[int] = set()
        for other in others:
            for value in other:
                try:
                    revoke_ids.add(_resolve_peer_device_id(value))
                except (TypeError, ValueError):
                    continue
        current = set(self._mr._query_peer_access_ids())
        to_remove = revoke_ids & current
        if to_remove:
            self._apply((), to_remove)

    def intersection_update(self, *others) -> None:
        """Restrict peer access to the intersection in a single driver call."""
        keep_ids: set[int] | None = None
        for other in others:
            ids: set[int] = set()
            for value in other:
                try:
                    ids.add(_resolve_peer_device_id(value))
                except (TypeError, ValueError):
                    continue
            keep_ids = ids if keep_ids is None else keep_ids & ids
        if keep_ids is None:
            return  # ``set.intersection_update()`` with no args is a no-op
        current = set(self._mr._query_peer_access_ids())
        to_remove = current - keep_ids
        if to_remove:
            self._apply((), to_remove)

    def symmetric_difference_update(self, other) -> None:
        """Toggle peer access for every device in ``other`` in one driver call."""
        toggle_ids: set[int] = set()
        for value in other:
            try:
                toggle_ids.add(_resolve_peer_device_id(value))
            except (TypeError, ValueError):
                continue
        current = set(self._mr._query_peer_access_ids())
        to_add = toggle_ids - current
        to_remove = toggle_ids & current
        if to_add or to_remove:
            self._apply(to_add, to_remove)

    def __ior__(self, other):
        self.update(other)
        return self

    def __iand__(self, other):
        self.intersection_update(other)
        return self

    def __isub__(self, other):
        if other is self:
            self.clear()
        else:
            self.difference_update(other)
        return self

    def __ixor__(self, other):
        self.symmetric_difference_update(other)
        return self

    def __repr__(self) -> str:
        return f"PeerAccessibleBySetProxy({set(self)!r})"

    # --- internal: route every write through one batched driver call ---

    def _apply(self, additions: Iterable[object], removals: Iterable[object]) -> None:
        """Compute the diff and issue a single ``cuMemPoolSetAccess``.

        ``additions`` and ``removals`` are user-supplied (``Device | int``);
        only the owner device is filtered out. Adds are validated through
        :meth:`Device.can_access_peer` via :func:`plan_peer_access_update`;
        removals bypass that check (revoking is always permitted).
        """
        from cuda.core._device import Device

        owner_id = self._mr._dev_id
        owner = Device(owner_id)
        current = self._mr._query_peer_access_ids()

        # Plan additions through the existing helper (validates can_access_peer).
        plan = plan_peer_access_update(
            owner_device_id=owner_id,
            current_peer_ids=current,
            # union of (current set + requested adds) so the planner emits
            # exactly the to_add deltas for these additions, no removals.
            requested_devices=[*current, *additions],
            resolve_device_id=_resolve_peer_device_id,
            can_access_peer=owner.can_access_peer,
        )
        to_add = plan.to_add

        # Removals: resolve, drop owner and unknowns, intersect with current.
        current_set = set(current)
        revoke_ids: set[int] = set()
        for value in removals:
            try:
                dev_id = _resolve_peer_device_id(value)
            except (TypeError, ValueError):
                continue
            if dev_id == owner_id:
                continue
            if dev_id in current_set:
                revoke_ids.add(dev_id)
        to_remove = tuple(sorted(revoke_ids))

        if not to_add and not to_remove:
            return
        self._mr._apply_peer_access_diff(to_add, to_remove)
