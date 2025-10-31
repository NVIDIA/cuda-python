# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for __eq__ and __ne__ implementations in cuda.core classes.

These tests verify multi-type equality behavior across Stream,
Event, Context, and Device objects.
"""

from cuda.core.experimental import Device, Stream

# ============================================================================
# Equality Contract Tests
# ============================================================================


def test_equality_is_not_identity():
    """Test that equality (==) is different from identity (is)."""
    device = Device(0)
    device.set_current()

    # Streams: Different objects can be equal
    s1 = device.create_stream()
    s2 = Stream.from_handle(int(s1.handle))

    assert s1 == s2, "Streams with same handle are equal"
    assert s1 is not s2, "But they are not the same object"

    # Device: Same object due to singleton (special case)
    d1 = Device(0)
    d2 = Device(0)

    assert d1 == d2, "Devices with same ID are equal"
    assert d1 is d2, "And they ARE the same object (singleton)"
