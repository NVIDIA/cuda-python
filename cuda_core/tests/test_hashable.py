# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for __hash__ implementation in cuda.core classes.

These tests verify multi-type hash behavior and integration patterns across
Stream, Event, Context, and Device objects.
"""

from cuda.core.experimental import Device

# ============================================================================
# Integration Tests
# ============================================================================


def test_hash_type_disambiguation_and_mixed_dict(init_cuda):
    """Test that hash salt (type(self)) prevents collisions between different types
    and that different object types can coexist in dictionaries.

    This test validates that:
    1. Including type(self) in the hash calculation ensures different types with
       potentially similar underlying values (like monotonically increasing handles
       or IDs) produce different hashes and don't collide.
    2. Different object types can be used together in the same dictionary without
       conflicts.
    """
    device = Device(0)
    device.set_current()

    # Create objects of different types
    stream = device.create_stream()
    event = stream.record()
    context = stream.context

    # Test 1: Verify all hashes are unique (no collisions between different types)
    hashes = {hash(device), hash(stream), hash(event), hash(context)}

    assert len(hashes) == 4, (
        f"Hash collision detected! Expected 4 unique hashes, got {len(hashes)}. "
        f"This indicates the type salt is not working correctly."
    )

    # Test 2: Verify all types can coexist in same dict without conflicts
    mixed_cache = {stream: "stream_data", event: "event_data", context: "context_data", device: "device_data"}

    assert len(mixed_cache) == 4, "All object types should coexist in dict"
    assert mixed_cache[stream] == "stream_data"
    assert mixed_cache[event] == "event_data"
    assert mixed_cache[context] == "context_data"
    assert mixed_cache[device] == "device_data"
