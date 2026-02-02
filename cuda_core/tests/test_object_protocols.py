# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for Python object protocols (__eq__, __hash__, __weakref__, __repr__).

This module tests that core cuda.core classes properly implement standard Python
object protocols for identity, hashing, weak references, and string representation.
"""

import itertools
import re
import weakref

import pytest
from cuda.core import Buffer, Device, Kernel, LaunchConfig, Program, Stream, system

# =============================================================================
# Fixtures - Primary samples
# =============================================================================


@pytest.fixture
def sample_device(init_cuda):
    """A sample Device object."""
    return Device(0)


@pytest.fixture
def sample_stream(sample_device):
    """A sample Stream object."""
    return sample_device.create_stream()


@pytest.fixture
def sample_event(sample_device):
    """A sample Event object."""
    return sample_device.create_event()


@pytest.fixture
def sample_context(sample_device):
    """A sample Context object."""
    return sample_device.context


@pytest.fixture
def sample_buffer(sample_device):
    """A sample Buffer object."""
    return sample_device.allocate(64)


@pytest.fixture
def sample_launch_config():
    """A sample LaunchConfig object."""
    return LaunchConfig(grid=(1,), block=(1,))


@pytest.fixture
def sample_object_code(init_cuda):
    """A sample ObjectCode object."""
    prog = Program('extern "C" __global__ void test_kernel() {}', "c++")
    return prog.compile("cubin")


@pytest.fixture
def sample_kernel(sample_object_code):
    """A sample Kernel object."""
    return sample_object_code.get_kernel("test_kernel")


# =============================================================================
# Fixtures - Alternate samples (for inequality testing)
# =============================================================================


@pytest.fixture
def sample_device_alt(init_cuda):
    """An alternate Device object (requires multi-GPU)."""
    if system.get_num_devices() < 2:
        pytest.skip("requires multi-GPU")
    device_alt = Device(1)
    device_alt.set_current()
    Device(0).set_current()
    return device_alt


@pytest.fixture
def sample_stream_alt(sample_device):
    """An alternate Stream object."""
    return sample_device.create_stream()


@pytest.fixture
def sample_event_alt(sample_device):
    """An alternate Event object."""
    return sample_device.create_event()


@pytest.fixture
def sample_context_alt(sample_device_alt):
    """An alternate Context object (requires multi-GPU)."""
    return sample_device_alt.context


@pytest.fixture
def sample_buffer_alt(sample_device):
    """An alternate Buffer object."""
    return sample_device.allocate(1024)


@pytest.fixture
def sample_launch_config_alt():
    """An alternate LaunchConfig object."""
    return LaunchConfig(grid=(2,), block=(2,))


@pytest.fixture
def sample_object_code_alt(init_cuda):
    """An alternate ObjectCode object."""
    prog = Program('extern "C" __global__ void test_kernel_alt() {}', "c++")
    return prog.compile("cubin")


@pytest.fixture
def sample_kernel_alt(sample_object_code_alt):
    """An alternate Kernel object."""
    return sample_object_code_alt.get_kernel("test_kernel_alt")


# =============================================================================
# Type groupings
# =============================================================================

# All types that should support weak references
API_TYPES = [
    "sample_device",
    "sample_stream",
    "sample_event",
    "sample_context",
    "sample_buffer",
    "sample_launch_config",
    "sample_object_code",
    "sample_kernel",
]

# Pairs of distinct objects of the same type (for inequality testing)
# Device and Context pairs require multi-GPU and will skip on single-GPU machines
SAME_TYPE_PAIRS = [
    ("sample_device", "sample_device_alt"),
    ("sample_stream", "sample_stream_alt"),
    ("sample_event", "sample_event_alt"),
    ("sample_context", "sample_context_alt"),
    ("sample_buffer", "sample_buffer_alt"),
    ("sample_launch_config", "sample_launch_config_alt"),
    ("sample_object_code", "sample_object_code_alt"),
    ("sample_kernel", "sample_kernel_alt"),
]

# Types with public from_handle methods and how to create a copy
FROM_HANDLE_COPIES = [
    ("sample_stream", lambda s: Stream.from_handle(int(s.handle))),
    ("sample_buffer", lambda b: Buffer.from_handle(b.handle, b.size)),
    ("sample_kernel", lambda k: Kernel.from_handle(int(k.handle))),
]

# Pairs of (fixture_name, regex_pattern) for repr format validation
REPR_PATTERNS = [
    ("sample_device", r"<Device \d+ \(.+\)>"),
    ("sample_stream", r"<Stream handle=0x[0-9a-f]+ context=0x[0-9a-f]+>"),
    ("sample_event", r"<Event handle=0x[0-9a-f]+>"),
    ("sample_context", r"<Context handle=0x[0-9a-f]+ device=\d+>"),
    ("sample_buffer", r"<Buffer ptr=0x[0-9a-f]+ size=\d+>"),
    (
        "sample_launch_config",
        r"LaunchConfig\(grid=\(\d+, \d+, \d+\), cluster=.+, block=\(\d+, \d+, \d+\), "
        r"shmem_size=\d+, cooperative_launch=(?:True|False)\)",
    ),
    ("sample_object_code", r"<ObjectCode handle=0x[0-9a-f]+ code_type='.+'>"),
    ("sample_kernel", r"<Kernel handle=0x[0-9a-f]+>"),
]


# =============================================================================
# Weak reference tests
# =============================================================================


@pytest.mark.parametrize("fixture_name", API_TYPES)
def test_weakref_supported(fixture_name, request):
    """Object supports weak references."""
    obj = request.getfixturevalue(fixture_name)
    ref = weakref.ref(obj)
    assert ref() is obj


# =============================================================================
# Hash tests
# =============================================================================


@pytest.mark.parametrize("fixture_name", API_TYPES)
def test_hash_consistency(fixture_name, request):
    """Hash is consistent across multiple calls."""
    obj = request.getfixturevalue(fixture_name)
    assert hash(obj) == hash(obj)


@pytest.mark.parametrize("fixture_name", API_TYPES)
def test_hash_not_small(fixture_name, request):
    """Hash should not be a small number (guards against returning IDs or indices)."""
    # Heuristic test guarding against poor hashes likely to collide, e.g.,
    # hash(device.device_id).
    obj = request.getfixturevalue(fixture_name)
    h = hash(obj)
    assert abs(h) >= 10, f"hash {h} is suspiciously small"


@pytest.mark.parametrize("a_name,b_name", SAME_TYPE_PAIRS)
def test_hash_distinct_same_type(a_name, b_name, request):
    """Distinct objects of the same type have different hashes."""
    # As a practical matter, the chance of collision is extremely low or even
    # zero; failure here likely indicates a bug.
    obj_a = request.getfixturevalue(a_name)
    obj_b = request.getfixturevalue(b_name)
    assert hash(obj_a) != hash(obj_b), f"{a_name} and {b_name} have same hash but different handles"


@pytest.mark.parametrize("a_name,b_name", itertools.combinations(API_TYPES, 2))
def test_hash_distinct_cross_type(a_name, b_name, request):
    """Distinct objects of different types have different hashes."""
    obj_a = request.getfixturevalue(a_name)
    obj_b = request.getfixturevalue(b_name)
    assert hash(obj_a) != hash(obj_b), f"{a_name} and {b_name} have same hash"


# =============================================================================
# Equality tests
# =============================================================================


@pytest.mark.parametrize("fixture_name", API_TYPES)
def test_equality_basic(fixture_name, request):
    """Object equality: reflexive, not equal to None or other types."""
    obj = request.getfixturevalue(fixture_name)
    assert obj == obj, "reflexive equality failed"
    assert obj != None, "should not equal None"  # noqa: E711
    assert obj != "string", "should not equal unrelated type"
    if hasattr(obj, "handle"):
        assert obj != obj.handle, "should not equal its own handle"


@pytest.mark.parametrize("a_name,b_name", itertools.combinations(API_TYPES, 2))
def test_no_cross_type_equality(a_name, b_name, request):
    """No two distinct objects of different types should compare equal."""
    obj_a = request.getfixturevalue(a_name)
    obj_b = request.getfixturevalue(b_name)
    assert obj_a != obj_b, f"{a_name} == {b_name} but they are distinct objects"


@pytest.mark.parametrize("a_name,b_name", SAME_TYPE_PAIRS)
def test_same_type_inequality(a_name, b_name, request):
    """Two distinct objects of the same type should not compare equal."""
    obj_a = request.getfixturevalue(a_name)
    obj_b = request.getfixturevalue(b_name)
    assert obj_a is not obj_b, f"{a_name} and {b_name} are the same object"
    assert obj_a != obj_b, f"{a_name} == {b_name} but they have different handles"


@pytest.mark.parametrize("fixture_name,copy_fn", FROM_HANDLE_COPIES)
def test_equality_same_handle(fixture_name, copy_fn, request):
    """Two wrappers around the same handle should compare equal."""
    obj = request.getfixturevalue(fixture_name)
    obj2 = copy_fn(obj)
    assert obj == obj2, f"wrapper equality failed for {type(obj).__name__}"
    assert hash(obj) == hash(obj2), f"hash equality failed for {type(obj).__name__}"


# =============================================================================
# Collection usage tests
# =============================================================================


@pytest.mark.parametrize("fixture_name", API_TYPES)
def test_usable_as_dict_key(fixture_name, request):
    """Object can be used as a dictionary key."""
    obj = request.getfixturevalue(fixture_name)
    d = {obj: "value"}
    assert d[obj] == "value"
    assert obj in d


@pytest.mark.parametrize("fixture_name", API_TYPES)
def test_usable_in_set(fixture_name, request):
    """Object can be added to a set."""
    obj = request.getfixturevalue(fixture_name)
    s = {obj}
    assert obj in s


@pytest.mark.parametrize("fixture_name", API_TYPES)
def test_usable_in_weak_value_dict(fixture_name, request):
    """Object can be used as a WeakValueDictionary value."""
    obj = request.getfixturevalue(fixture_name)
    wvd = weakref.WeakValueDictionary()
    wvd["key"] = obj
    assert wvd["key"] is obj


@pytest.mark.parametrize("fixture_name", API_TYPES)
def test_usable_in_weak_key_dict(fixture_name, request):
    """Object can be used as a WeakKeyDictionary key."""
    obj = request.getfixturevalue(fixture_name)
    wkd = weakref.WeakKeyDictionary()
    wkd[obj] = "value"
    assert wkd[obj] == "value"


@pytest.mark.parametrize("fixture_name", API_TYPES)
def test_usable_in_weak_set(fixture_name, request):
    """Object can be added to a WeakSet."""
    obj = request.getfixturevalue(fixture_name)
    ws = weakref.WeakSet()
    ws.add(obj)
    assert obj in ws


# =============================================================================
# Repr tests
# =============================================================================


@pytest.mark.parametrize("fixture_name,pattern", REPR_PATTERNS)
def test_repr_format(fixture_name, pattern, request):
    """repr() returns a properly formatted string."""
    obj = request.getfixturevalue(fixture_name)
    result = repr(obj)
    assert re.fullmatch(pattern, result), f"repr {result!r} does not match {pattern!r}"
