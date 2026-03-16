# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.buffers import PatternGen, compare_buffer_to_constant, make_scratch_buffer

import cuda.core
from cuda.core import DeviceMemoryResource, DeviceMemoryResourceOptions
from cuda.core._utils.cuda_utils import CUDAError

NBYTES = 1024


@pytest.mark.usefixtures("requires_concurrent_managed_access")
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


def test_peer_access_property_x2(mempool_device_x2):
    """The the dmr.peer_accessible_by property (but not its functionality)."""
    # The peer access list is a sorted tuple and always excludes the self
    # device.
    dev0, dev1 = mempool_device_x2
    # Use owned pool to ensure clean initial state (no stale peer access).
    dmr = DeviceMemoryResource(dev0, DeviceMemoryResourceOptions())

    def check(expected):
        assert isinstance(dmr.peer_accessible_by, tuple)
        assert dmr.peer_accessible_by == expected

    # No access to begin with.
    check(expected=())
    # fmt: off
    dmr.peer_accessible_by = (0,)            ; check(expected=())
    dmr.peer_accessible_by = (1,)            ; check(expected=(1,))
    dmr.peer_accessible_by = (0, 1)          ; check(expected=(1,))
    dmr.peer_accessible_by = ()              ; check(expected=())
    dmr.peer_accessible_by = [0, 1]          ; check(expected=(1,))
    dmr.peer_accessible_by = set()           ; check(expected=())
    dmr.peer_accessible_by = [1, 1, 1, 1, 1] ; check(expected=(1,))
    # fmt: on

    with pytest.raises(ValueError, match=r"device_id must be \>\= 0"):
        dmr.peer_accessible_by = [-1]  # device ID out of bounds

    num_devices = len(cuda.core.Device.get_all_devices())

    with pytest.raises(ValueError, match=r"device_id must be within \[0, \d+\)"):
        dmr.peer_accessible_by = [num_devices]  # device ID out of bounds


@pytest.mark.usefixtures("requires_concurrent_managed_access")
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
        assert dmrs[0].peer_accessible_by == init_state
        verify_state(init_state, pattern_seed)
        pattern_seed += 1

        dmrs[0].peer_accessible_by = final_state
        assert dmrs[0].peer_accessible_by == final_state
        verify_state(final_state, pattern_seed)
        pattern_seed += 1


def test_peer_access_shared_pool_queries_driver(mempool_device_x2):
    """Non-owned pools always query the driver for peer access state."""
    dev0, dev1 = mempool_device_x2

    # Grant peer access via one wrapper; a second wrapper must see it.
    dmr1 = DeviceMemoryResource(dev0)
    dmr1.peer_accessible_by = [dev1]
    dmr2 = DeviceMemoryResource(dev0)
    assert dev1.device_id in dmr2.peer_accessible_by

    # Revoke via dmr2; dmr1 must reflect the change immediately.
    dmr2.peer_accessible_by = []
    assert dmr1.peer_accessible_by == ()

    # Re-grant via dmr1. A fresh wrapper that has never read the
    # property must still query the driver before computing diffs
    # in the setter, so setting [] must discover and revoke the access.
    dmr1.peer_accessible_by = [dev1]
    dmr3 = DeviceMemoryResource(dev0)
    assert dmr1.peer_accessible_by == (dev1.device_id,)
    assert dmr2.peer_accessible_by == (dev1.device_id,)
    assert dmr3.peer_accessible_by == (dev1.device_id,)
    dmr3.peer_accessible_by = []
    assert DeviceMemoryResource(dev0).peer_accessible_by == ()
    assert dmr1.peer_accessible_by == ()
    assert dmr2.peer_accessible_by == ()
    assert dmr3.peer_accessible_by == ()
