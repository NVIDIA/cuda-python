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
from helpers.graph_kernels import compile_common_kernels

from cuda.core import Buffer, Device, Kernel, LaunchConfig, Program, Stream, system
from cuda.core._graph._graphdef import GraphDef
from cuda.core._program import _can_load_generated_ptx

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
def sample_kernel(sample_object_code_cubin):
    """A sample Kernel object."""
    return sample_object_code_cubin.get_kernel("test_kernel")


# =============================================================================
# Fixtures - ObjectCode variations (by code_type)
# =============================================================================


@pytest.fixture
def sample_object_code_cubin(init_cuda):
    """An ObjectCode compiled to cubin."""
    prog = Program('extern "C" __global__ void test_kernel() {}', "c++")
    return prog.compile("cubin")


@pytest.fixture
def sample_object_code_ptx(init_cuda):
    """An ObjectCode compiled to PTX."""
    if not _can_load_generated_ptx():
        pytest.skip("PTX version too new for current driver")
    prog = Program('extern "C" __global__ void test_kernel() {}', "c++")
    return prog.compile("ptx")


@pytest.fixture
def sample_object_code_ltoir(init_cuda):
    """An ObjectCode compiled to LTOIR."""
    prog = Program('extern "C" __global__ void test_kernel() {}', "c++")
    return prog.compile("ltoir")


# =============================================================================
# Fixtures - Program variations (by backend)
# =============================================================================


@pytest.fixture
def sample_program_nvrtc(init_cuda):
    """A Program using NVRTC backend (C++ code)."""
    return Program('extern "C" __global__ void k() {}', "c++")


@pytest.fixture
def sample_program_ptx(init_cuda):
    """A Program using linker backend (PTX code)."""
    # First compile C++ to PTX, then create a Program from PTX
    if not _can_load_generated_ptx():
        pytest.skip("PTX version too new for current driver")
    prog = Program('extern "C" __global__ void k() {}', "c++")
    obj = prog.compile("ptx")
    ptx_code = obj.code.decode() if isinstance(obj.code, bytes) else obj.code
    return Program(ptx_code, "ptx")


@pytest.fixture
def sample_program_nvvm(init_cuda):
    """A Program using NVVM backend (NVVM IR code)."""
    # Minimal NVVM IR that declares a kernel
    # fmt: off
    nvvm_ir = (
        'target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-'
        'i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"\n'
        'target triple = "nvptx64-nvidia-cuda"\n'
        "\n"
        "define void @test_kernel() {\n"
        "  ret void\n"
        "}\n"
        "\n"
        "!nvvm.annotations = !{!0}\n"
        '!0 = !{void ()* @test_kernel, !"kernel", i32 1}\n'
    )
    # fmt: on
    try:
        return Program(nvvm_ir, "nvvm")
    except RuntimeError as e:
        if "NVVM" in str(e):
            pytest.skip("NVVM not available")
        raise


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
# Fixtures - Graph types (GraphDef and Node)
# =============================================================================

ALLOC_SIZE = 1024


@pytest.fixture
def sample_graphdef(init_cuda):
    """A sample GraphDef."""
    return GraphDef()


@pytest.fixture
def sample_graphdef_alt(init_cuda):
    """An alternate GraphDef (for inequality testing)."""
    return GraphDef()


@pytest.fixture
def sample_root_node(sample_graphdef):
    """An entry Node (virtual, NULL handle)."""
    return sample_graphdef._entry


@pytest.fixture
def sample_root_node_alt(sample_graphdef_alt):
    """An alternate entry Node from different graph."""
    return sample_graphdef_alt._entry


@pytest.fixture
def sample_empty_node(sample_graphdef):
    """An EmptyNode created by merging two branches."""
    a = sample_graphdef.alloc(ALLOC_SIZE)
    b = sample_graphdef.alloc(ALLOC_SIZE)
    return sample_graphdef.join(a, b)


@pytest.fixture
def sample_empty_node_alt(sample_graphdef):
    """An alternate EmptyNode from same graph."""
    c = sample_graphdef.alloc(ALLOC_SIZE)
    d = sample_graphdef.alloc(ALLOC_SIZE)
    return sample_graphdef.join(c, d)


@pytest.fixture
def sample_alloc_node(sample_graphdef):
    """An AllocNode."""
    return sample_graphdef.alloc(ALLOC_SIZE)


@pytest.fixture
def sample_alloc_node_alt(sample_graphdef):
    """An alternate AllocNode from same graph."""
    return sample_graphdef.alloc(ALLOC_SIZE)


@pytest.fixture
def sample_kernel_node(sample_graphdef, init_cuda):
    """A KernelNode."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)
    return sample_graphdef.launch(config, kernel)


@pytest.fixture
def sample_kernel_node_alt(sample_graphdef, init_cuda):
    """An alternate KernelNode from same graph."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)
    return sample_graphdef.launch(config, kernel)


@pytest.fixture
def sample_free_node(sample_graphdef):
    """A FreeNode."""
    alloc = sample_graphdef.alloc(ALLOC_SIZE)
    return alloc.free(alloc.dptr)


@pytest.fixture
def sample_free_node_alt(sample_graphdef):
    """An alternate FreeNode from same graph."""
    alloc = sample_graphdef.alloc(ALLOC_SIZE)
    return alloc.free(alloc.dptr)


@pytest.fixture
def sample_memset_node(sample_graphdef):
    """A MemsetNode."""
    alloc = sample_graphdef.alloc(ALLOC_SIZE)
    return alloc.memset(alloc.dptr, 0, ALLOC_SIZE)


@pytest.fixture
def sample_memset_node_alt(sample_graphdef):
    """An alternate MemsetNode from same graph."""
    alloc = sample_graphdef.alloc(ALLOC_SIZE)
    return alloc.memset(alloc.dptr, 0, ALLOC_SIZE)


@pytest.fixture
def sample_event_record_node(sample_graphdef, sample_device):
    """An EventRecordNode."""
    event = sample_device.create_event()
    return sample_graphdef.record_event(event)


@pytest.fixture
def sample_event_record_node_alt(sample_graphdef, sample_device):
    """An alternate EventRecordNode from same graph."""
    event = sample_device.create_event()
    return sample_graphdef.record_event(event)


@pytest.fixture
def sample_event_wait_node(sample_graphdef, sample_device):
    """An EventWaitNode."""
    event = sample_device.create_event()
    return sample_graphdef.wait_event(event)


@pytest.fixture
def sample_event_wait_node_alt(sample_graphdef, sample_device):
    """An alternate EventWaitNode from same graph."""
    event = sample_device.create_event()
    return sample_graphdef.wait_event(event)


# =============================================================================
# Type groupings
# =============================================================================

# Types with __hash__ support
HASH_TYPES = [
    "sample_device",
    "sample_stream",
    "sample_event",
    "sample_context",
    "sample_buffer",
    "sample_launch_config",
    "sample_object_code_cubin",
    "sample_kernel",
    "sample_graphdef",
    "sample_root_node",
    "sample_empty_node",
    "sample_alloc_node",
    "sample_kernel_node",
    "sample_free_node",
    "sample_memset_node",
    "sample_event_record_node",
    "sample_event_wait_node",
]

# Types with __eq__ support
EQ_TYPES = [
    "sample_device",
    "sample_stream",
    "sample_event",
    "sample_context",
    "sample_buffer",
    "sample_launch_config",
    "sample_object_code_cubin",
    "sample_kernel",
    "sample_graphdef",
    "sample_root_node",
    "sample_empty_node",
    "sample_alloc_node",
    "sample_kernel_node",
    "sample_free_node",
    "sample_memset_node",
    "sample_event_record_node",
    "sample_event_wait_node",
]

# Types with __weakref__ support
WEAKREF_TYPES = [
    "sample_device",
    "sample_stream",
    "sample_event",
    "sample_context",
    "sample_buffer",
    "sample_launch_config",
    "sample_object_code_cubin",
    "sample_kernel",
    "sample_program_nvrtc",
    "sample_graphdef",
    "sample_root_node",
    "sample_empty_node",
    "sample_alloc_node",
    "sample_kernel_node",
    "sample_free_node",
    "sample_memset_node",
    "sample_event_record_node",
    "sample_event_wait_node",
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
    ("sample_object_code_cubin", "sample_object_code_alt"),
    ("sample_kernel", "sample_kernel_alt"),
    ("sample_graphdef", "sample_graphdef_alt"),
    ("sample_root_node", "sample_root_node_alt"),
    ("sample_empty_node", "sample_empty_node_alt"),
    ("sample_alloc_node", "sample_alloc_node_alt"),
    ("sample_kernel_node", "sample_kernel_node_alt"),
    ("sample_free_node", "sample_free_node_alt"),
    ("sample_memset_node", "sample_memset_node_alt"),
    ("sample_event_record_node", "sample_event_record_node_alt"),
    ("sample_event_wait_node", "sample_event_wait_node_alt"),
]

# Types with public from_handle methods and how to create a copy
FROM_HANDLE_COPIES = [
    ("sample_stream", lambda s: Stream.from_handle(int(s.handle))),
    ("sample_buffer", lambda b: Buffer.from_handle(b.handle, b.size)),
    ("sample_kernel", lambda k: Kernel.from_handle(int(k.handle))),
]

# Derived type groupings for collection tests
DICT_KEY_TYPES = sorted(set(HASH_TYPES) & set(EQ_TYPES))
WEAK_KEY_TYPES = sorted(set(HASH_TYPES) & set(EQ_TYPES) & set(WEAKREF_TYPES))

# Pairs of (fixture_name, regex_pattern) for repr format validation
REPR_PATTERNS = [
    # Core types
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
    ("sample_kernel", r"<Kernel handle=0x[0-9a-f]+>"),
    # ObjectCode variations (by code_type)
    ("sample_object_code_cubin", r"<ObjectCode handle=0x[0-9a-f]+ code_type='cubin'>"),
    ("sample_object_code_ptx", r"<ObjectCode handle=0x[0-9a-f]+ code_type='ptx'>"),
    ("sample_object_code_ltoir", r"<ObjectCode handle=0x[0-9a-f]+ code_type='ltoir'>"),
    # Program variations (by backend)
    ("sample_program_nvrtc", r"<Program backend='NVRTC'>"),
    ("sample_program_ptx", r"<Program backend='(nvJitLink|driver)'>"),
    ("sample_program_nvvm", r"<Program backend='NVVM'>"),
    # Graph types
    ("sample_graphdef", r"<GraphDef handle=0x[0-9a-f]+>"),
    ("sample_root_node", r"<Node entry>"),
    ("sample_empty_node", r"<EmptyNode with \d+ preds?>"),
    ("sample_alloc_node", r"<AllocNode dptr=0x[0-9a-f]+ size=\d+>"),
    ("sample_kernel_node", r"<KernelNode grid=\(\d+, \d+, \d+\) block=\(\d+, \d+, \d+\)>"),
    ("sample_free_node", r"<FreeNode dptr=0x[0-9a-f]+>"),
    ("sample_memset_node", r"<MemsetNode dptr=0x[0-9a-f]+ value=\d+ elem=\d+>"),
    ("sample_event_record_node", r"<EventRecordNode event=0x[0-9a-f]+>"),
    ("sample_event_wait_node", r"<EventWaitNode event=0x[0-9a-f]+>"),
]


# =============================================================================
# Weak reference tests
# =============================================================================


@pytest.mark.parametrize("fixture_name", WEAKREF_TYPES)
def test_weakref_supported(fixture_name, request):
    """Object supports weak references."""
    obj = request.getfixturevalue(fixture_name)
    ref = weakref.ref(obj)
    assert ref() is obj


# =============================================================================
# Hash tests
# =============================================================================


@pytest.mark.parametrize("fixture_name", HASH_TYPES)
def test_hash_consistency(fixture_name, request):
    """Hash is consistent across multiple calls."""
    obj = request.getfixturevalue(fixture_name)
    assert hash(obj) == hash(obj)


@pytest.mark.parametrize("a_name,b_name", SAME_TYPE_PAIRS)
def test_hash_distinct_same_type(a_name, b_name, request):
    """Distinct objects of the same type have different hashes."""
    obj_a = request.getfixturevalue(a_name)
    obj_b = request.getfixturevalue(b_name)
    assert hash(obj_a) != hash(obj_b)  # extremely unlikely


@pytest.mark.parametrize("a_name,b_name", itertools.combinations(HASH_TYPES, 2))
def test_hash_distinct_cross_type(a_name, b_name, request):
    """Distinct objects of different types have different hashes."""
    obj_a = request.getfixturevalue(a_name)
    obj_b = request.getfixturevalue(b_name)
    assert hash(obj_a) != hash(obj_b)  # extremely unlikely


# =============================================================================
# Equality tests
# =============================================================================


@pytest.mark.parametrize("fixture_name", EQ_TYPES)
def test_equality_basic(fixture_name, request):
    """Object equality: reflexive, not equal to None or other types."""
    obj = request.getfixturevalue(fixture_name)
    assert obj == obj
    assert obj is not None
    assert obj != "string"
    if hasattr(obj, "handle"):
        assert obj != obj.handle


@pytest.mark.parametrize("a_name,b_name", itertools.combinations(EQ_TYPES, 2))
def test_no_cross_type_equality(a_name, b_name, request):
    """No two distinct objects of different types should compare equal."""
    obj_a = request.getfixturevalue(a_name)
    obj_b = request.getfixturevalue(b_name)
    assert obj_a != obj_b


@pytest.mark.parametrize("a_name,b_name", SAME_TYPE_PAIRS)
def test_same_type_inequality(a_name, b_name, request):
    """Two distinct objects of the same type should not compare equal."""
    obj_a = request.getfixturevalue(a_name)
    obj_b = request.getfixturevalue(b_name)
    assert obj_a is not obj_b
    assert obj_a != obj_b


@pytest.mark.parametrize("fixture_name,copy_fn", FROM_HANDLE_COPIES)
def test_equality_same_handle(fixture_name, copy_fn, request):
    """Two wrappers around the same handle should compare equal."""
    obj = request.getfixturevalue(fixture_name)
    obj2 = copy_fn(obj)
    assert obj == obj2
    assert hash(obj) == hash(obj2)


# =============================================================================
# Collection usage tests
# =============================================================================


@pytest.mark.parametrize("fixture_name", DICT_KEY_TYPES)
def test_usable_as_dict_key(fixture_name, request):
    """Object can be used as a dictionary key."""
    obj = request.getfixturevalue(fixture_name)
    d = {obj: "value"}
    assert d[obj] == "value"
    assert obj in d


@pytest.mark.parametrize("fixture_name", DICT_KEY_TYPES)
def test_usable_in_set(fixture_name, request):
    """Object can be added to a set."""
    obj = request.getfixturevalue(fixture_name)
    s = {obj}
    assert obj in s


@pytest.mark.parametrize("fixture_name", WEAKREF_TYPES)
def test_usable_in_weak_value_dict(fixture_name, request):
    """Object can be used as a WeakValueDictionary value."""
    obj = request.getfixturevalue(fixture_name)
    wvd = weakref.WeakValueDictionary()
    wvd["key"] = obj
    assert wvd["key"] is obj


@pytest.mark.parametrize("fixture_name", WEAK_KEY_TYPES)
def test_usable_in_weak_key_dict(fixture_name, request):
    """Object can be used as a WeakKeyDictionary key."""
    obj = request.getfixturevalue(fixture_name)
    wkd = weakref.WeakKeyDictionary()
    wkd[obj] = "value"
    assert wkd[obj] == "value"


@pytest.mark.parametrize("fixture_name", WEAK_KEY_TYPES)
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
    assert re.fullmatch(pattern, result)
