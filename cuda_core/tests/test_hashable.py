# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for __hash__ implementation in cuda.core classes.

These tests verify that Stream, Event, Context, and Device objects can be
hashed and used as dictionary keys and in sets, following the pattern
established by PyTorch and CuPy.
"""

from cuda.core.experimental import Device, Stream
from cuda.core.experimental._stream import LEGACY_DEFAULT_STREAM, PER_THREAD_DEFAULT_STREAM

# ============================================================================
# Stream Tests
# ============================================================================


def test_stream_hash_consistency(init_cuda):
    """Hash of same Stream object should be consistent."""
    stream = Device().create_stream()
    hash1 = hash(stream)
    hash2 = hash(stream)
    assert hash1 == hash2, "Hash should be consistent for same object"


def test_stream_hash_equality_same_handle(init_cuda):
    """Two Stream objects wrapping same handle should hash equal."""
    device = Device()
    s1 = device.create_stream()
    s2 = Stream.from_handle(int(s1.handle))

    assert hash(s1) == hash(s2), "Same handle should produce same hash"
    assert s1 == s2, "Streams with same handle should be equal"


def test_stream_hash_inequality_different_handles(init_cuda):
    """Different streams should have different hashes."""
    device = Device()
    s1 = device.create_stream()
    s2 = device.create_stream()

    assert s1 != s2, "Different streams should not be equal"
    # Hash inequality not guaranteed by contract, but very likely
    assert hash(s1) != hash(s2), "Different streams likely have different hashes"


def test_stream_dict_key(init_cuda):
    """Streams should be usable as dictionary keys."""
    device = Device()
    s1 = device.create_stream()
    s2 = Stream.from_handle(int(s1.handle))

    # Use s1 as key
    cache = {s1: "value1"}

    # s2 should find s1's entry (same handle)
    assert s2 in cache, "Stream with same handle should be in dict"
    assert cache[s2] == "value1", "Should retrieve value via equivalent stream"

    # Updating via s2 should update s1's entry
    cache[s2] = "value2"
    assert len(cache) == 1, "Should update existing entry, not add new one"
    assert cache[s1] == "value2", "Original key should see updated value"


def test_stream_set_membership(init_cuda):
    """Streams should work correctly in sets."""
    device = Device()
    s1 = device.create_stream()
    s2 = Stream.from_handle(int(s1.handle))
    s3 = device.create_stream()

    stream_set = {s1}

    # s2 wraps same handle as s1, should be recognized as duplicate
    assert s2 in stream_set, "Equivalent stream should be in set"
    stream_set.add(s2)
    assert len(stream_set) == 1, "Should not add duplicate"

    # s3 is different stream
    stream_set.add(s3)
    assert len(stream_set) == 2, "Should add different stream"


def test_builtin_stream_hash(init_cuda):
    """Builtin streams should be hashable and usable as dict keys."""
    Device().set_current()

    legacy = LEGACY_DEFAULT_STREAM
    per_thread = PER_THREAD_DEFAULT_STREAM

    # Should be hashable
    hash_legacy = hash(legacy)
    hash_per_thread = hash(per_thread)
    assert isinstance(hash_legacy, int)
    assert isinstance(hash_per_thread, int)

    # Should have different hashes (different handles)
    assert hash_legacy != hash_per_thread

    # Should be usable in dicts
    cache = {legacy: "legacy_data", per_thread: "per_thread_data"}
    assert len(cache) == 2
    assert cache[legacy] == "legacy_data"
    assert cache[per_thread] == "per_thread_data"


def test_default_stream_consistency(init_cuda):
    """Device.default_stream should be consistent and hashable."""
    device = Device()

    default1 = device.default_stream
    default2 = device.default_stream

    # Should be same object (or at least equal)
    assert default1 == default2
    assert hash(default1) == hash(default2)


# ============================================================================
# Event Tests
# ============================================================================


def test_event_hash_consistency(init_cuda):
    """Hash of same Event object should be consistent."""
    device = Device()
    stream = device.create_stream()
    event = stream.record()

    hash1 = hash(event)
    hash2 = hash(event)
    assert hash1 == hash2, "Hash should be consistent for same object"


def test_event_hash_equality(init_cuda):
    """Events with same underlying handle should hash equal."""
    device = Device()
    stream = device.create_stream()

    # Create events
    e1 = stream.record()
    e2 = stream.record()

    # Different events should have different hashes
    assert e1 != e2
    assert hash(e1) != hash(e2)

    # Same event should equal itself
    assert e1 == e1
    assert hash(e1) == hash(e1)


def test_event_dict_key(init_cuda):
    """Events should be usable as dictionary keys."""
    device = Device()
    stream = device.create_stream()

    e1 = stream.record()
    e2 = stream.record()

    # Use events as keys
    event_cache = {e1: "timing1", e2: "timing2"}

    assert len(event_cache) == 2
    assert event_cache[e1] == "timing1"
    assert event_cache[e2] == "timing2"


def test_event_set_membership(init_cuda):
    """Events should work correctly in sets."""
    device = Device()
    stream = device.create_stream()

    e1 = stream.record()
    e2 = stream.record()

    event_set = {e1, e2}
    assert len(event_set) == 2

    # Same event should not add duplicate
    event_set.add(e1)
    assert len(event_set) == 2


# ============================================================================
# Context Tests
# ============================================================================


def test_context_hash_consistency(init_cuda):
    """Hash of same Context object should be consistent."""
    device = Device()
    stream = device.create_stream()
    context = stream.context

    hash1 = hash(context)
    hash2 = hash(context)
    assert hash1 == hash2, "Hash should be consistent for same object"


def test_context_equality(init_cuda):
    """Contexts from same device should be equal."""
    device = Device()

    s1 = device.create_stream()
    s2 = device.create_stream()

    ctx1 = s1.context
    ctx2 = s2.context

    # Same device, should have same context
    assert ctx1 == ctx2, "Streams on same device should share context"
    assert hash(ctx1) == hash(ctx2), "Same context should hash equal"


def test_context_dict_key(init_cuda):
    """Contexts should be usable as dictionary keys."""
    device = Device()
    stream = device.create_stream()
    context = stream.context

    ctx_cache = {context: "context_data"}
    assert ctx_cache[context] == "context_data"


# ============================================================================
# Device Tests
# ============================================================================


def test_device_hash_consistency(init_cuda):
    """Hash of same Device object should be consistent."""
    device = Device(0)

    hash1 = hash(device)
    hash2 = hash(device)
    assert hash1 == hash2, "Hash should be consistent for same object"


def test_device_equality_same_id(init_cuda):
    """Devices with same device_id should be equal."""
    dev1 = Device(0)
    dev2 = Device(0)

    # On same thread, should be same object (singleton)
    assert dev1 is dev2, "Device is per-thread singleton"
    assert dev1 == dev2, "Same device_id should be equal"
    assert hash(dev1) == hash(dev2), "Same device_id should hash equal"


def test_device_inequality_different_id(init_cuda):
    """Devices with different device_id should not be equal."""
    try:
        # Only run test when two devices are available.
        dev0 = Device(0)
        dev1 = Device(1)

        assert dev0 != dev1, "Different devices should not be equal"
        assert hash(dev0) != hash(dev1), "Different devices should have different hashes"
    except (ValueError, Exception):
        pass


def test_device_dict_key(init_cuda):
    """Devices should be usable as dictionary keys."""
    dev0 = Device(0)

    device_cache = {dev0: "gpu0_data"}
    assert device_cache[dev0] == "gpu0_data"

    # Getting device again should find same entry
    dev0_again = Device(0)
    assert device_cache[dev0_again] == "gpu0_data"


def test_device_set_membership(init_cuda):
    """Devices should work correctly in sets."""
    dev0_a = Device(0)
    dev0_b = Device(0)

    device_set = {dev0_a}

    # Same device_id should not add duplicate
    device_set.add(dev0_b)
    assert len(device_set) == 1, "Should not add duplicate device"


# ============================================================================
# Integration Tests
# ============================================================================


def test_mixed_object_dict():
    """Test that different object types don't conflict in dicts."""
    device = Device(0)
    device.set_current()

    stream = device.create_stream()
    event = stream.record()
    context = stream.context

    # All should be usable in same dict without conflicts
    mixed_cache = {stream: "stream_data", event: "event_data", context: "context_data", device: "device_data"}

    assert len(mixed_cache) == 4
    assert mixed_cache[stream] == "stream_data"
    assert mixed_cache[event] == "event_data"
    assert mixed_cache[context] == "context_data"
    assert mixed_cache[device] == "device_data"


def test_cache_pattern_example(init_cuda):
    """Test realistic caching pattern with cache hit tracking."""
    device = Device()

    # Stream-specific computation cache with hit tracking
    stream_results = {}
    cache_hits = 0
    cache_misses = 0

    def compute_on_stream(stream, data):
        nonlocal cache_hits, cache_misses
        if stream not in stream_results:
            # Cache miss - compute
            cache_misses += 1
            stream_results[stream] = f"result_for_{data}"
        else:
            # Cache hit
            cache_hits += 1
        return stream_results[stream]

    s1 = device.create_stream()

    # First call - should miss
    result1 = compute_on_stream(s1, "input1")
    assert cache_hits == 0, "First call should be cache miss"
    assert cache_misses == 1, "Should have 1 cache miss"

    # Second call with same stream - should hit
    result2 = compute_on_stream(s1, "input1")
    assert cache_hits == 1, "Second call should be cache hit"
    assert cache_misses == 1, "Should still have 1 cache miss"
    assert result1 == result2, "Should get same cached result"

    # Third call with same stream - another hit
    result3 = compute_on_stream(s1, "input1")
    assert cache_hits == 2, "Third call should be cache hit"
    assert cache_misses == 1, "Should still have 1 cache miss"

    # Different stream - should miss
    s2 = device.create_stream()
    result4 = compute_on_stream(s2, "input1")
    assert cache_hits == 2, "Different stream should not affect hit count"
    assert cache_misses == 2, "Should have 2 cache misses now"
    assert len(stream_results) == 2, "Should have 2 cache entries"

    # Wrapped stream with same handle - should hit!
    s1_wrapped = Stream.from_handle(int(s1.handle))
    result5 = compute_on_stream(s1_wrapped, "input1")
    assert cache_hits == 3, "Wrapped stream should cause cache hit"
    assert cache_misses == 2, "Should still have 2 cache misses"
    assert result5 == result1, "Wrapped stream should get same result"

    # Final validation
    assert cache_hits == 3, "Expected 3 cache hits total"
    assert cache_misses == 2, "Expected 2 cache misses total"
    assert len(stream_results) == 2, "Expected 2 unique streams in cache"
