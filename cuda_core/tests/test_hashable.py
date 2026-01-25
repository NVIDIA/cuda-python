# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for __hash__ implementation in cuda.core classes.

These tests verify:
1. Multi-type hash behavior and collision prevention
2. Subclassing hash behavior for all core types
3. Hash consistency (same object always returns same hash)
4. Dictionary/set usage patterns
5. Hash/equality contract compliance (if a == b, then hash(a) must equal hash(b))
"""

import pytest
from cuda.core import Device, LaunchConfig, Program
from cuda.core._stream import Stream, StreamOptions

# ============================================================================
# Fixtures for parameterized tests
# ============================================================================


@pytest.fixture
def sample_device(init_cuda):
    return Device()


@pytest.fixture
def sample_stream(sample_device):
    return sample_device.create_stream()


@pytest.fixture
def sample_event(sample_device):
    return sample_device.create_event()


@pytest.fixture
def sample_context(sample_device):
    return sample_device.context


@pytest.fixture
def sample_buffer(sample_device):
    return sample_device.allocate(1024)


@pytest.fixture
def sample_launch_config():
    return LaunchConfig(grid=(1,), block=(1,))


@pytest.fixture
def sample_object_code(init_cuda):
    prog = Program('extern "C" __global__ void test_kernel() {}', "c++")
    return prog.compile("ptx")


@pytest.fixture
def sample_kernel(sample_object_code):
    return sample_object_code.get_kernel("test_kernel")


# All hashable classes
HASHABLE = [
    "sample_device",
    "sample_stream",
    "sample_event",
    "sample_context",
    "sample_buffer",
    "sample_launch_config",
    "sample_object_code",
    "sample_kernel",
]


# ============================================================================
# Parameterized Hash Tests
# ============================================================================


@pytest.mark.parametrize("fixture_name", HASHABLE)
def test_hash_consistency(fixture_name, request):
    """Hash of same object is consistent across calls."""
    obj = request.getfixturevalue(fixture_name)
    assert hash(obj) == hash(obj)


@pytest.mark.parametrize("fixture_name", HASHABLE)
def test_set_membership(fixture_name, request):
    """Objects work correctly in sets."""
    obj = request.getfixturevalue(fixture_name)
    s = {obj}
    assert obj in s
    assert len(s) == 1


@pytest.mark.parametrize("fixture_name", HASHABLE)
def test_dict_key(fixture_name, request):
    """Objects work correctly as dict keys."""
    obj = request.getfixturevalue(fixture_name)
    d = {obj: "value"}
    assert d[obj] == "value"


# ============================================================================
# Integration Tests
# ============================================================================


def test_mixed_type_dict(init_cuda):
    """Test that different object types can coexist in dictionaries.

    Since each CUDA handle type has unique values within its type (handles are
    memory addresses or unique identifiers), hash collisions between different
    types are unlikely in practice.
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


# ============================================================================
# Subclassing Hash Tests
# ============================================================================


def test_device_subclass_hash(init_cuda):
    """Test Device subclass hash behavior.

    Device uses a singleton pattern where Device(0) always returns the same
    cached instance. This means MyDevice(0) returns the original Device instance,
    not a new MyDevice instance.
    """

    class MyDevice(Device):
        pass

    device = Device(0)
    device.set_current()
    my_device = MyDevice(0)

    # Singleton returns same instance, so hashes are identical
    assert device is my_device, "Singleton returns same instance"
    assert device == my_device, "Singleton returns same instance"
    assert hash(device) == hash(my_device), "Same object has same hash"

    # Verify hash consistency
    hash1 = hash(device)
    hash2 = hash(device)
    assert hash1 == hash2, "Hash is consistent across multiple calls"


def test_stream_subclass_hash(init_cuda):
    """Test Stream subclass hash behavior."""

    class MyStream(Stream):
        pass

    device = Device(0)
    device.set_current()

    # Same type, same handle -> same hash
    stream1 = Stream._init(options=StreamOptions(), device_id=device.device_id)
    stream2 = Stream.from_handle(int(stream1.handle))
    assert hash(stream1) == hash(stream2), "Streams wrapping same handle have same hash"

    # Verify hash consistency
    hash1 = hash(stream1)
    hash2 = hash(stream1)
    assert hash1 == hash2, "Hash is consistent across multiple calls"

    # Different type, same handle -> SAME hash (type not included in hash)
    my_stream = MyStream._init(options=StreamOptions(), device_id=device.device_id)
    stream_from_handle = Stream.from_handle(int(my_stream.handle))

    assert type(stream_from_handle) is Stream, "from_handle returns Stream, not subclass"
    assert hash(my_stream) == hash(stream_from_handle), (
        "Same handle produces same hash regardless of type (maintains hash/equality contract)"
    )

    # Verify equality matches hash
    assert my_stream == stream_from_handle, "Equal due to isinstance() and same handle"
    assert hash(my_stream) == hash(stream_from_handle), "Equal objects have equal hashes"

    # Different handles -> different hashes
    my_stream2 = MyStream._init(options=StreamOptions(), device_id=device.device_id)
    assert my_stream != my_stream2, "Different streams are not equal"
    assert hash(my_stream) != hash(my_stream2), "Different streams have different hashes"


def test_event_hash(init_cuda):
    """Test Event hash behavior."""
    device = Device(0)
    device.set_current()

    # Create events using public API
    event1 = device.create_event()
    event2 = device.create_event()

    # Different events (different handles) -> different hashes
    assert hash(event1) != hash(event2), "Different events have different hashes"
    assert event1 != event2, "Different handles means not equal"

    # Verify hash consistency
    hash1 = hash(event1)
    hash2 = hash(event1)
    assert hash1 == hash2, "Hash is consistent across multiple calls"

    # Both should be usable as dict keys
    cache = {event1: "first", event2: "second"}
    assert len(cache) == 2, "Different events are distinct dict keys"
    assert cache[event1] == "first"
    assert cache[event2] == "second"


def test_context_hash(init_cuda):
    """Test Context hash behavior."""
    device = Device(0)
    device.set_current()

    # Get context from different sources
    stream1 = device.create_stream()
    stream2 = device.create_stream()
    context1 = stream1.context
    context2 = stream2.context

    # Same underlying context -> same hash
    assert hash(context1) == hash(context2), "Contexts with same handle have same hash"

    # Verify equality matches hash
    assert context1 == context2, "Contexts with same handle are equal"

    # Verify hash consistency
    hash1 = hash(context1)
    hash2 = hash(context1)
    assert hash1 == hash2, "Hash is consistent across multiple calls"


def test_hash_equality_contract_maintained(init_cuda):
    """Verify that the hash/equality contract is maintained with subclasses.

    This test demonstrates that Stream (and other classes) now properly maintain
    Python's invariant: if a == b, then hash(a) must equal hash(b)

    The fix: removed type(self) from __hash__ while keeping isinstance() in __eq__,
    allowing cross-type equality with consistent hashing.
    """

    device = Device(0)
    device.set_current()

    # Test Stream: two references to same handle
    stream1 = device.create_stream()
    stream2 = Stream.from_handle(int(stream1.handle))

    assert stream1 == stream2, "Equal due to same handle"
    assert hash(stream1) == hash(stream2), "Equal objects have equal hashes"

    # Test Context: contexts from same device share same underlying context
    ctx1 = device.context
    ctx2 = device.create_stream().context

    assert ctx1 == ctx2, "Equal contexts with same handle"
    assert hash(ctx1) == hash(ctx2), "Equal objects have equal hashes"

    # Test that different handles still produce different hashes
    stream3 = device.create_stream()
    assert stream1 != stream3, "Different handles means not equal"
    assert hash(stream1) != hash(stream3), "Different objects have different hashes"
