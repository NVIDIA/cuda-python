# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.buffers import PatternGen, compare_buffer_to_constant, make_scratch_buffer
from helpers.collection_interface_testers import assert_mutable_set_interface

from cuda.core import Device, DeviceMemoryResource, DeviceMemoryResourceOptions, system
from cuda.core._memory import _peer_access_utils
from cuda.core._memory._peer_access_utils import PeerAccessibleBySetProxy
from cuda.core._utils.cuda_utils import CUDAError

NBYTES = 1024


def test_peer_access_basic(mempool_device_x2):
    """Basic tests for dmr.peer_accessible_by."""
    dev0, dev1 = mempool_device_x2
    zero_on_dev0 = make_scratch_buffer(dev0, 0, NBYTES)
    one_on_dev0 = make_scratch_buffer(dev0, 1, NBYTES)
    stream_on_dev0 = dev0.create_stream()
    # Use owned pool to ensure clean initial state (no stale peer access).
    dmr_on_dev1 = DeviceMemoryResource(dev1, DeviceMemoryResourceOptions())
    buf_on_dev1 = dmr_on_dev1.allocate(NBYTES)

    # No access at first.
    assert 0 not in dmr_on_dev1.peer_accessible_by
    with pytest.raises(CUDAError, match="CUDA_ERROR_INVALID_VALUE"):
        one_on_dev0.copy_to(buf_on_dev1, stream=stream_on_dev0)

    with pytest.raises(CUDAError, match="CUDA_ERROR_INVALID_VALUE"):
        zero_on_dev0.copy_from(buf_on_dev1, stream=stream_on_dev0)

    # Allow access to device 1's allocations from device 0.
    dmr_on_dev1.peer_accessible_by = [dev0]
    assert 0 in dmr_on_dev1.peer_accessible_by
    compare_buffer_to_constant(zero_on_dev0, 0)
    one_on_dev0.copy_to(buf_on_dev1, stream=stream_on_dev0)
    zero_on_dev0.copy_from(buf_on_dev1, stream=stream_on_dev0)
    stream_on_dev0.sync()
    compare_buffer_to_constant(zero_on_dev0, 1)

    # Revoke access
    dmr_on_dev1.peer_accessible_by = []
    assert 0 not in dmr_on_dev1.peer_accessible_by
    with pytest.raises(CUDAError, match="CUDA_ERROR_INVALID_VALUE"):
        one_on_dev0.copy_to(buf_on_dev1, stream=stream_on_dev0)

    with pytest.raises(CUDAError, match="CUDA_ERROR_INVALID_VALUE"):
        zero_on_dev0.copy_from(buf_on_dev1, stream=stream_on_dev0)


def test_peer_access_transitions(mempool_device_x3):
    """Advanced tests for dmr.peer_accessible_by."""

    # Check all transitions between peer access states. The implementation
    # performs transactions that add or remove access as needed. This test
    # ensures that that is working as expected.

    # Doing everything from the point-of-view of device 0, there are four
    # access states:
    #
    #     [(), (1,), (2,), (1, 2)]
    #
    # and 4^2-4 = 12 non-identity transitions.

    devs = mempool_device_x3  # Three devices

    # Allocate per-device resources.
    streams = [dev.create_stream() for dev in devs]
    pgens = [PatternGen(devs[i], NBYTES, streams[i]) for i in range(3)]
    # Use owned pools (with options) to ensure clean initial state.
    # Default pools are shared and may have stale peer access from prior tests.
    dmrs = [DeviceMemoryResource(dev, DeviceMemoryResourceOptions()) for dev in devs]
    bufs = [dmr.allocate(NBYTES) for dmr in dmrs]

    def verify_state(state, pattern_seed):
        """
        Verify an access state from the POV of device 0. E.g., (1,) means
        device 1 has access but device 2 does not.
        """
        # Populate device 0's buffer with a new pattern.
        devs[0].set_current()
        pgens[0].fill_buffer(bufs[0], seed=pattern_seed)
        streams[0].sync()

        for peer in [1, 2]:
            devs[peer].set_current()
            if peer in state:
                # Peer device has access to 0's allocation
                bufs[peer].copy_from(bufs[0], stream=streams[peer])
                # Check the result on the peer device.
                pgens[peer].verify_buffer(bufs[peer], seed=pattern_seed)
            else:
                # Peer device has no access to 0's allocation
                with pytest.raises(CUDAError, match="CUDA_ERROR_INVALID_VALUE"):
                    bufs[peer].copy_from(bufs[0], stream=streams[peer])

    # For each transition, set the access state before and after, checking for
    # the expected peer access capabilities at each stop.
    pattern_seed = 0
    states = [(), (1,), (2,), (1, 2)]
    transitions = [(s0, s1) for s0 in states for s1 in states if s0 != s1]
    for init_state, final_state in transitions:
        dmrs[0].peer_accessible_by = init_state
        assert dmrs[0].peer_accessible_by == {Device(i) for i in init_state}
        verify_state(init_state, pattern_seed)
        pattern_seed += 1

        dmrs[0].peer_accessible_by = final_state
        assert dmrs[0].peer_accessible_by == {Device(i) for i in final_state}
        verify_state(final_state, pattern_seed)
        pattern_seed += 1


def test_peer_access_shared_pool_queries_driver(mempool_device_x2):
    """All pools always query the driver, so wrappers see consistent state."""
    dev0, dev1 = mempool_device_x2

    # Grant peer access via one wrapper; a second wrapper must see it.
    dmr1 = DeviceMemoryResource(dev0)
    dmr1.peer_accessible_by = [dev1]
    dmr2 = DeviceMemoryResource(dev0)
    assert dev1.device_id in dmr2.peer_accessible_by

    # Revoke via dmr2; dmr1 must reflect the change immediately.
    dmr2.peer_accessible_by = []
    assert dmr1.peer_accessible_by == set()

    # Re-grant via dmr1. A fresh wrapper that has never read the
    # property must still query the driver before computing diffs
    # in the setter, so setting [] must discover and revoke the access.
    dmr1.peer_accessible_by = [dev1]
    dmr3 = DeviceMemoryResource(dev0)
    assert dmr1.peer_accessible_by == {dev1}
    assert dmr2.peer_accessible_by == {dev1}
    assert dmr3.peer_accessible_by == {dev1}
    dmr3.peer_accessible_by = []
    assert DeviceMemoryResource(dev0).peer_accessible_by == set()
    assert dmr1.peer_accessible_by == set()
    assert dmr2.peer_accessible_by == set()
    assert dmr3.peer_accessible_by == set()


# ---------------------------------------------------------------------------
# Set-proxy interface coverage
#
# These tests exercise the ``PeerAccessibleBySetProxy`` surface added in
# v1.0.0. They run against ``mempool_device_x2`` because every CI machine has
# at most 2 GPUs, which means at most one valid peer device. The relaxed
# ``support_multi_insert=False`` path on ``assert_mutable_set_interface``
# threads that single insertable element through the full ``MutableSet``
# protocol.
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_dmr_x2(mempool_device_x2):
    """Owned-pool DMR on dev0 + the lone peer device (dev1).

    The owned pool guarantees a clean, empty initial peer-access state so the
    proxy tests are not polluted by other tests sharing a default pool.
    """
    dev0, dev1 = mempool_device_x2
    dmr = DeviceMemoryResource(dev0, DeviceMemoryResourceOptions())
    dmr.peer_accessible_by = []
    return dmr, dev0, dev1


def test_peer_accessible_by_mutable_set_interface(isolated_dmr_x2):
    """Run the full MutableSet protocol against a single-peer driver-backed view.

    On a 2-GPU box the proxy can only ever hold ``{dev1}``. The relaxed helper
    uses ``dev0`` (the owner, which the proxy refuses to insert) as the
    non-member sentinel; this exercises every ``MutableSet`` method while
    respecting the proxy's "insert at most one" hardware reality.
    """
    dmr, dev0, dev1 = isolated_dmr_x2
    assert_mutable_set_interface(
        dmr.peer_accessible_by,
        items=[dev1],
        non_members=[dev0, dev0.device_id],
        support_multi_insert=False,
    )


def test_peer_accessible_by_accepts_int_and_device(isolated_dmr_x2):
    """``add``/``discard``/``__contains__`` accept ``Device`` and ``int`` interchangeably."""
    dmr, dev0, dev1 = isolated_dmr_x2

    dmr.peer_accessible_by.add(dev1.device_id)
    assert dmr.peer_accessible_by == {dev1}
    assert dev1 in dmr.peer_accessible_by
    assert dev1.device_id in dmr.peer_accessible_by

    dmr.peer_accessible_by.discard(dev1)
    assert dmr.peer_accessible_by == set()

    dmr.peer_accessible_by.add(dev1)
    assert dmr.peer_accessible_by == {dev1}
    dmr.peer_accessible_by.discard(dev1.device_id)
    assert dmr.peer_accessible_by == set()


def test_peer_accessible_by_silently_ignores_owner(isolated_dmr_x2):
    """The owner device is silently filtered on every write; ``__contains__`` returns False."""
    dmr, dev0, dev1 = isolated_dmr_x2

    # add/discard on owner is a no-op (no error, no state change)
    dmr.peer_accessible_by.add(dev0)
    dmr.peer_accessible_by.add(dev0.device_id)
    assert dmr.peer_accessible_by == set()
    dmr.peer_accessible_by.discard(dev0)
    dmr.peer_accessible_by.discard(dev0.device_id)
    assert dmr.peer_accessible_by == set()

    # __contains__ on owner is False (matches set semantics, never raises)
    assert dev0 not in dmr.peer_accessible_by
    assert dev0.device_id not in dmr.peer_accessible_by

    # Owner mixed into bulk ops is filtered, the peer is still added/removed
    dmr.peer_accessible_by |= {dev0, dev1}
    assert dmr.peer_accessible_by == {dev1}
    dmr.peer_accessible_by -= {dev0, dev1}
    assert dmr.peer_accessible_by == set()


def test_peer_accessible_by_rejects_invalid_inputs(isolated_dmr_x2):
    """``add`` raises on out-of-range/unsupported inputs; lenient methods do not."""
    dmr, dev0, dev1 = isolated_dmr_x2
    bad_id = system.get_num_devices()  # one past the last valid device ordinal

    # add: validates strictly, propagates errors from Device(bad_id)
    with pytest.raises((ValueError, CUDAError)):
        dmr.peer_accessible_by.add(bad_id)
    # Non-coercible inputs surface whatever Device(value) raises (TypeError or
    # ValueError depending on Cython's int coercion path).
    with pytest.raises((TypeError, ValueError)):
        dmr.peer_accessible_by.add("not-a-device")

    # discard: silently ignores non-coercible values (matches set.discard)
    dmr.peer_accessible_by.discard("not-a-device")
    assert dmr.peer_accessible_by == set()

    # __contains__: returns False on non-coercible values, never raises
    assert "not-a-device" not in dmr.peer_accessible_by

    # remove on a non-member raises KeyError (inherited from MutableSet)
    with pytest.raises(KeyError):
        dmr.peer_accessible_by.remove(dev1)


def test_peer_accessible_by_no_cache_across_proxies(mempool_device_x2):
    """Updates via one wrapper are immediately visible through any other proxy."""
    dev0, dev1 = mempool_device_x2
    dmr_a = DeviceMemoryResource(dev0)
    dmr_b = DeviceMemoryResource(dev0)
    dmr_a.peer_accessible_by = []

    proxy = dmr_a.peer_accessible_by  # acquired before the change below
    dmr_b.peer_accessible_by.add(dev1)
    # The proxy must reflect the new driver state, not a snapshot.
    assert dev1 in proxy
    assert proxy == {dev1}

    dmr_b.peer_accessible_by.clear()
    assert proxy == set()


def test_peer_accessible_by_iteration_order_is_sorted(mempool_device_x2):
    """``__iter__`` yields peers in ascending device-ordinal order."""
    dev0, dev1 = mempool_device_x2
    dmr = DeviceMemoryResource(dev0, DeviceMemoryResourceOptions())
    dmr.peer_accessible_by = [dev1]
    devices = list(dmr.peer_accessible_by)
    ids = [d.device_id for d in devices]
    assert ids == sorted(ids)
    assert all(isinstance(d, Device) for d in devices)


def test_peer_accessible_by_repr(isolated_dmr_x2):
    """``repr`` includes the class name and reflects the live contents."""
    dmr, dev0, dev1 = isolated_dmr_x2
    empty_repr = repr(dmr.peer_accessible_by)
    assert "PeerAccessibleBySetProxy" in empty_repr
    assert "set()" in empty_repr

    dmr.peer_accessible_by.add(dev1)
    populated_repr = repr(dmr.peer_accessible_by)
    assert "PeerAccessibleBySetProxy" in populated_repr
    # Don't pin the exact device repr; just confirm content changed.
    assert populated_repr != empty_repr


def test_peer_accessible_by_returns_proxy_type(isolated_dmr_x2):
    """The getter returns the documented proxy type (anchors the public contract)."""
    dmr, dev0, dev1 = isolated_dmr_x2
    assert isinstance(dmr.peer_accessible_by, PeerAccessibleBySetProxy)


# ---------------------------------------------------------------------------
# Batching contract: every bulk op must issue at most one cuMemPoolSetAccess
#
# Spying via ``monkeypatch.setattr`` on the module-level
# ``_apply_peer_access_diff`` works because the proxy and the property setter
# call it by bare name, which Cython resolves through the module's globals at
# runtime (the wrapper is a plain ``def``, not a ``cdef inline``).
# ---------------------------------------------------------------------------


class _DiffSpy:
    """Counts batched driver calls and forwards each call to the real wrapper.

    Only invocations with non-empty deltas are recorded in :attr:`calls`,
    because those are the ones that translate to actual ``cuMemPoolSetAccess``
    work. The wrapper still gets called with empty deltas in a few places
    (e.g. the property setter when the requested target matches current
    driver state), but the underlying ``cdef inline`` short-circuits before
    issuing the driver call, so those invocations do not count against the
    "one call per bulk op" contract.
    """

    def __init__(self, real):
        self._real = real
        self.calls = []

    def __call__(self, mr, to_add, to_remove):
        recorded_add = tuple(to_add)
        recorded_remove = tuple(to_remove)
        if recorded_add or recorded_remove:
            self.calls.append((recorded_add, recorded_remove))
        self._real(mr, to_add, to_remove)


@pytest.fixture
def diff_spy(monkeypatch):
    spy = _DiffSpy(_peer_access_utils._apply_peer_access_diff)
    monkeypatch.setattr(_peer_access_utils, "_apply_peer_access_diff", spy)
    return spy


def test_peer_accessible_by_setter_batches_one_call(diff_spy, isolated_dmr_x2):
    """``mr.peer_accessible_by = [...]`` issues exactly one driver call (or zero on no-op)."""
    dmr, dev0, dev1 = isolated_dmr_x2
    dmr.peer_accessible_by = [dev1]
    assert len(diff_spy.calls) == 1
    assert dev1.device_id in diff_spy.calls[-1][0]

    # Reassigning the same set is a no-op (zero driver calls).
    diff_spy.calls.clear()
    dmr.peer_accessible_by = [dev1]
    assert diff_spy.calls == []

    # Revoking everything is a single call (one removal).
    dmr.peer_accessible_by = []
    assert len(diff_spy.calls) == 1
    assert dev1.device_id in diff_spy.calls[-1][1]


def test_peer_accessible_by_bulk_ops_batch_one_call(diff_spy, isolated_dmr_x2):
    """``|=``/``&=``/``-=``/``^=``/``update``/``clear`` each issue at most one driver call.

    .. note::
       ``proxy = dmr.peer_accessible_by; proxy |= {...}`` is exactly one driver
       call. Augmented assignment directly on the property
       (``dmr.peer_accessible_by |= {...}``) is two operations because Python
       fetches the proxy, mutates it, and then assigns the (already-mutated)
       proxy back through the setter; that final assignment is a no-op at the
       driver level (deltas come out empty), but it is two trips through the
       proxy/setter machinery. The spy filters out empty-delta calls so the
       invariant under test is "actual driver work" rather than that
       Python-level quirk.
    """
    dmr, dev0, dev1 = isolated_dmr_x2
    proxy = dmr.peer_accessible_by  # use the local-binding form for predictable counting

    proxy |= {dev1}
    assert len(diff_spy.calls) == 1
    diff_spy.calls.clear()

    # &= keeping the lone member: no driver call (no diff).
    proxy &= {dev1}
    assert diff_spy.calls == []

    # &= dropping the lone member: one removal.
    proxy &= {dev0}
    assert len(diff_spy.calls) == 1
    diff_spy.calls.clear()

    # ^= toggling the lone peer in then out: two ops, one call each.
    proxy ^= {dev1}
    assert len(diff_spy.calls) == 1
    proxy ^= {dev1}
    assert len(diff_spy.calls) == 2
    diff_spy.calls.clear()

    # update() with the peer already absent: one add.
    proxy.update([dev1])
    assert len(diff_spy.calls) == 1
    diff_spy.calls.clear()

    # clear() with one member: one removal.
    proxy.clear()
    assert len(diff_spy.calls) == 1
    diff_spy.calls.clear()

    # Already-empty bulk ops are no-ops (nothing to add or remove).
    proxy.clear()
    proxy.difference_update([dev1])
    proxy -= {dev1}
    assert diff_spy.calls == []
