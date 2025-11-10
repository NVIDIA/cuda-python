# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from cuda.core.experimental import Device, Stream, StreamOptions
from cuda.core.experimental._event import Event
from cuda.core.experimental._stream import LEGACY_DEFAULT_STREAM, PER_THREAD_DEFAULT_STREAM
from cuda.core.experimental._utils.cuda_utils import driver


def test_stream_init_disabled():
    with pytest.raises(RuntimeError, match=r"^Stream objects cannot be instantiated directly\."):
        Stream()  # Reject at front door.


def test_stream_init_with_options(init_cuda):
    stream = Device().create_stream(options=StreamOptions(nonblocking=True, priority=0))
    assert stream.is_nonblocking is True
    assert stream.priority == 0


def test_stream_handle(init_cuda):
    stream = Device().create_stream(options=StreamOptions())
    assert isinstance(stream.handle, driver.CUstream)


def test_stream_is_nonblocking(init_cuda):
    stream = Device().create_stream(options=StreamOptions(nonblocking=True))
    assert stream.is_nonblocking is True


def test_stream_priority(init_cuda):
    stream = Device().create_stream(options=StreamOptions(priority=0))
    assert stream.priority == 0
    stream = Device().create_stream(options=StreamOptions(priority=-1))
    assert stream.priority == -1
    with pytest.raises(ValueError):
        stream = Device().create_stream(options=StreamOptions(priority=1))


def test_stream_sync(init_cuda):
    stream = Device().create_stream(options=StreamOptions())
    stream.sync()  # Should not raise any exceptions


def test_stream_record(init_cuda):
    stream = Device().create_stream(options=StreamOptions())
    event = stream.record()
    assert isinstance(event, Event)


def test_stream_wait_event(init_cuda):
    s1 = Device().create_stream()
    s2 = Device().create_stream()
    e1 = s1.record()
    s2.wait(e1)  # Should not raise any exceptions
    s2.sync()


def test_stream_wait_invalid_event(init_cuda):
    stream = Device().create_stream(options=StreamOptions())
    with pytest.raises(ValueError):
        stream.wait(event_or_stream="invalid_event")


def test_stream_device(init_cuda):
    stream = Device().create_stream(options=StreamOptions())
    device = stream.device
    assert isinstance(device, Device)


def test_stream_context(init_cuda):
    stream = Device().create_stream(options=StreamOptions())
    context = stream.context
    assert context is not None
    assert context._handle is not None


def test_stream_from_foreign_stream(init_cuda):
    device = Device()
    other_stream = device.create_stream(options=StreamOptions())
    stream = device.create_stream(obj=other_stream)
    # Now that __eq__ is implemented (issue #664), we can compare directly
    assert other_stream == stream
    assert hash(other_stream) == hash(stream)
    device = stream.device
    assert isinstance(device, Device)
    context = stream.context
    assert context is not None


def test_stream_from_handle():
    stream = Stream.from_handle(0)
    assert isinstance(stream, Stream)


def test_legacy_default_stream():
    assert isinstance(LEGACY_DEFAULT_STREAM, Stream)


def test_per_thread_default_stream():
    assert isinstance(PER_THREAD_DEFAULT_STREAM, Stream)


def test_stream_subclassing(init_cuda):
    class MyStream(Stream):
        pass

    dev = Device()
    dev.set_current()
    stream = MyStream._init(options=StreamOptions(), device_id=dev.device_id)
    assert isinstance(stream, MyStream)


def test_stream_legacy_default_subclassing():
    class MyStream(Stream):
        pass

    stream = MyStream._legacy_default()
    assert isinstance(stream, MyStream)


def test_stream_per_thread_default_subclassing():
    class MyStream(Stream):
        pass

    stream = MyStream._per_thread_default()
    assert isinstance(stream, MyStream)


# ============================================================================
# Stream Equality Tests
# ============================================================================


def test_stream_equality_same_handle(init_cuda):
    """Two Stream objects wrapping same handle should be equal."""
    device = Device()
    s1 = device.create_stream()
    s2 = Stream.from_handle(int(s1.handle))

    assert s1 == s2, "Equal streams should be equal"


def test_stream_inequality_different_handles(init_cuda):
    """Different streams should not be equal."""
    device = Device()
    s1 = device.create_stream()
    s2 = device.create_stream()

    assert s1 != s2, "Different streams should not be equal"


def test_stream_equality_reflexive(init_cuda):
    """Stream should equal itself (reflexive property)."""
    stream = Device().create_stream()
    assert stream == stream, "Stream should equal itself"


def test_stream_equality_symmetric(init_cuda):
    """If s1 == s2, then s2 == s1 (symmetric property)."""
    device = Device()
    s1 = device.create_stream()
    s2 = Stream.from_handle(int(s1.handle))

    assert s1 == s2, "Equality should be symmetric"
    assert s2 == s1, "Equality should be symmetric"


def test_stream_type_safety(init_cuda):
    """Comparing Stream with wrong type should return False."""
    stream = Device().create_stream()

    # These should not raise exceptions
    assert (stream == "not a stream") is False
    assert (stream == 123) is False
    assert (stream is None) is False
    assert (stream == Device()) is False


def test_stream_not_equal_operator(init_cuda):
    """Test != operator works correctly for streams."""
    device = Device()
    s1 = device.create_stream()
    s2 = device.create_stream()
    s3 = Stream.from_handle(int(s1.handle))

    assert s1 != s2, "Different streams should be not-equal"
    assert s1 == s3, "Same handle streams should be equal"


# ============================================================================
# Stream Hash Tests
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
