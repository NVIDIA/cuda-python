# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for Python object protocols (__eq__, __hash__, __weakref__, __repr__, pickle).

This module tests that core cuda.core classes properly implement standard Python
object protocols for identity, hashing, weak references, string representation,
and serialization.
"""

import itertools
import re
import weakref

import pytest
from helpers.graph_kernels import compile_common_kernels
from helpers.misc import try_create_condition

from conftest import xfail_on_graph_mempool_oom
from cuda.core import (
    Buffer,
    Device,
    DeviceMemoryResource,
    DeviceMemoryResourceOptions,
    Kernel,
    LaunchConfig,
    Program,
    Stream,
    system,
)
from cuda.core._program import _can_load_generated_ptx
from cuda.core.graph import GraphDefinition


def _skip_if_no_mempool():
    if not Device(0).properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")


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
    return sample_device.allocate(64, stream=sample_device.default_stream)


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
    return sample_device.allocate(1024, stream=sample_device.default_stream)


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
# Fixtures - IPC samples (for pickle tests)
# =============================================================================

POOL_SIZE = 2097152


@pytest.fixture
def sample_ipc_buffer_descriptor(ipc_device):
    """An IPCBufferDescriptor."""
    options = DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True)
    mr = DeviceMemoryResource(ipc_device, options=options)
    buf = mr.allocate(64, stream=ipc_device.default_stream)
    descriptor = buf.ipc_descriptor
    buf.close()
    # TODO(seberg): 2026-06: mr close may be unsafe with incomplete `buf.close()`
    ipc_device.sync()
    return descriptor


@pytest.fixture
def sample_ipc_event_descriptor(ipc_device):
    """An IPCEventDescriptor."""
    stream = ipc_device.create_stream()
    e = stream.record(options={"ipc_enabled": True})
    return e.ipc_descriptor


# =============================================================================
# Fixtures - Graph types (GraphDefinition and GraphNode)
# =============================================================================

ALLOC_SIZE = 1024


@pytest.fixture
def sample_graphdef(init_cuda):
    """A sample GraphDefinition."""
    return GraphDefinition()


@pytest.fixture
def sample_graphdef_alt(init_cuda):
    """An alternate GraphDefinition (for inequality testing)."""
    return GraphDefinition()


@pytest.fixture
def sample_root_node(sample_graphdef):
    """An entry GraphNode (virtual, NULL handle)."""
    return sample_graphdef._entry


@pytest.fixture
def sample_root_node_alt(sample_graphdef_alt):
    """An alternate entry GraphNode from different graph."""
    return sample_graphdef_alt._entry


@pytest.fixture
def sample_empty_node(sample_graphdef):
    """An EmptyNode created by merging two branches."""
    _skip_if_no_mempool()
    with xfail_on_graph_mempool_oom():
        a = sample_graphdef.allocate(ALLOC_SIZE)
        b = sample_graphdef.allocate(ALLOC_SIZE)
        return sample_graphdef.join(a, b)


@pytest.fixture
def sample_empty_node_alt(sample_graphdef):
    """An alternate EmptyNode from same graph."""
    _skip_if_no_mempool()
    with xfail_on_graph_mempool_oom():
        c = sample_graphdef.allocate(ALLOC_SIZE)
        d = sample_graphdef.allocate(ALLOC_SIZE)
        return sample_graphdef.join(c, d)


@pytest.fixture
def sample_alloc_node(sample_graphdef):
    """An AllocNode."""
    _skip_if_no_mempool()
    with xfail_on_graph_mempool_oom():
        return sample_graphdef.allocate(ALLOC_SIZE)


@pytest.fixture
def sample_alloc_node_alt(sample_graphdef):
    """An alternate AllocNode from same graph."""
    _skip_if_no_mempool()
    with xfail_on_graph_mempool_oom():
        return sample_graphdef.allocate(ALLOC_SIZE)


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
    _skip_if_no_mempool()
    with xfail_on_graph_mempool_oom():
        alloc = sample_graphdef.allocate(ALLOC_SIZE)
        return alloc.deallocate(alloc.dptr)


@pytest.fixture
def sample_free_node_alt(sample_graphdef):
    """An alternate FreeNode from same graph."""
    _skip_if_no_mempool()
    with xfail_on_graph_mempool_oom():
        alloc = sample_graphdef.allocate(ALLOC_SIZE)
        return alloc.deallocate(alloc.dptr)


@pytest.fixture
def sample_memset_node(sample_graphdef):
    """A MemsetNode."""
    _skip_if_no_mempool()
    with xfail_on_graph_mempool_oom():
        alloc = sample_graphdef.allocate(ALLOC_SIZE)
        return alloc.memset(alloc.dptr, 0, ALLOC_SIZE)


@pytest.fixture
def sample_memset_node_alt(sample_graphdef):
    """An alternate MemsetNode from same graph."""
    _skip_if_no_mempool()
    with xfail_on_graph_mempool_oom():
        alloc = sample_graphdef.allocate(ALLOC_SIZE)
        return alloc.memset(alloc.dptr, 0, ALLOC_SIZE)


@pytest.fixture
def sample_memcpy_node(sample_graphdef):
    """A MemcpyNode."""
    _skip_if_no_mempool()
    with xfail_on_graph_mempool_oom():
        src = sample_graphdef.allocate(ALLOC_SIZE)
        dst = sample_graphdef.allocate(ALLOC_SIZE)
        dep = sample_graphdef.join(src, dst)
        return dep.memcpy(dst.dptr, src.dptr, ALLOC_SIZE)


@pytest.fixture
def sample_memcpy_node_alt(sample_graphdef):
    """An alternate MemcpyNode from same graph."""
    _skip_if_no_mempool()
    with xfail_on_graph_mempool_oom():
        src = sample_graphdef.allocate(ALLOC_SIZE)
        dst = sample_graphdef.allocate(ALLOC_SIZE)
        dep = sample_graphdef.join(src, dst)
        return dep.memcpy(dst.dptr, src.dptr, ALLOC_SIZE)


@pytest.fixture
def sample_child_graph_node(sample_graphdef):
    """A ChildGraphNode."""
    child = GraphDefinition()
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    child.launch(LaunchConfig(grid=1, block=1), kernel)
    return sample_graphdef.embed(child)


@pytest.fixture
def sample_child_graph_node_alt(sample_graphdef):
    """An alternate ChildGraphNode from same graph."""
    child = GraphDefinition()
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    child.launch(LaunchConfig(grid=1, block=1), kernel)
    return sample_graphdef.embed(child)


@pytest.fixture
def sample_event_record_node(sample_graphdef, sample_device):
    """An EventRecordNode."""
    event = sample_device.create_event()
    return sample_graphdef.record(event)


@pytest.fixture
def sample_event_record_node_alt(sample_graphdef, sample_device):
    """An alternate EventRecordNode from same graph."""
    event = sample_device.create_event()
    return sample_graphdef.record(event)


@pytest.fixture
def sample_event_wait_node(sample_graphdef, sample_device):
    """An EventWaitNode."""
    event = sample_device.create_event()
    return sample_graphdef.wait(event)


@pytest.fixture
def sample_event_wait_node_alt(sample_graphdef, sample_device):
    """An alternate EventWaitNode from same graph."""
    event = sample_device.create_event()
    return sample_graphdef.wait(event)


@pytest.fixture
def sample_host_callback_node(sample_graphdef):
    """A HostCallbackNode."""

    def my_callback():
        pass

    return sample_graphdef.callback(my_callback)


@pytest.fixture
def sample_host_callback_node_alt(sample_graphdef):
    """An alternate HostCallbackNode from same graph."""

    def other_callback():
        pass

    return sample_graphdef.callback(other_callback)


@pytest.fixture
def sample_condition(sample_graphdef):
    """A GraphCondition object."""
    return try_create_condition(sample_graphdef)


@pytest.fixture
def sample_condition_alt(sample_graphdef):
    """An alternate GraphCondition from same graph."""
    return try_create_condition(sample_graphdef)


@pytest.fixture
def sample_if_node(sample_graphdef):
    """An IfNode."""
    condition = try_create_condition(sample_graphdef)
    return sample_graphdef.if_then(condition)


@pytest.fixture
def sample_if_node_alt(sample_graphdef):
    """An alternate IfNode from same graph."""
    condition = try_create_condition(sample_graphdef)
    return sample_graphdef.if_then(condition)


@pytest.fixture
def sample_if_else_node(sample_graphdef):
    """An IfElseNode."""
    condition = try_create_condition(sample_graphdef)
    return sample_graphdef.if_else(condition)


@pytest.fixture
def sample_if_else_node_alt(sample_graphdef):
    """An alternate IfElseNode from same graph."""
    condition = try_create_condition(sample_graphdef)
    return sample_graphdef.if_else(condition)


@pytest.fixture
def sample_while_node(sample_graphdef):
    """A WhileNode."""
    condition = try_create_condition(sample_graphdef)
    return sample_graphdef.while_loop(condition)


@pytest.fixture
def sample_while_node_alt(sample_graphdef):
    """An alternate WhileNode from same graph."""
    condition = try_create_condition(sample_graphdef)
    return sample_graphdef.while_loop(condition)


@pytest.fixture
def sample_switch_node(sample_graphdef):
    """A SwitchNode."""
    condition = try_create_condition(sample_graphdef)
    return sample_graphdef.switch(condition, 3)


@pytest.fixture
def sample_switch_node_alt(sample_graphdef):
    """An alternate SwitchNode from same graph."""
    condition = try_create_condition(sample_graphdef)
    return sample_graphdef.switch(condition, 3)


# Indirect-parametrize helpers: request.getfixturevalue() runs here, in the
# fixture (main thread), so the resolved object is already available when the
# test function runs in a worker thread.


@pytest.fixture
def sample_object(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def sample_object_a(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def sample_object_b(request):
    return request.getfixturevalue(request.param)


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
    "sample_condition",
    "sample_root_node",
    "sample_empty_node",
    "sample_alloc_node",
    "sample_kernel_node",
    "sample_free_node",
    "sample_memset_node",
    "sample_memcpy_node",
    "sample_child_graph_node",
    "sample_event_record_node",
    "sample_event_wait_node",
    "sample_host_callback_node",
    "sample_if_node",
    "sample_if_else_node",
    "sample_while_node",
    "sample_switch_node",
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
    "sample_condition",
    "sample_root_node",
    "sample_empty_node",
    "sample_alloc_node",
    "sample_kernel_node",
    "sample_free_node",
    "sample_memset_node",
    "sample_memcpy_node",
    "sample_child_graph_node",
    "sample_event_record_node",
    "sample_event_wait_node",
    "sample_host_callback_node",
    "sample_if_node",
    "sample_if_else_node",
    "sample_while_node",
    "sample_switch_node",
]

# Types with __weakref__ support
WEAKREF_TYPES = [
    "sample_device",
    "sample_stream",
    "sample_event",
    "sample_context",
    "sample_condition",
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
    "sample_memcpy_node",
    "sample_child_graph_node",
    "sample_event_record_node",
    "sample_event_wait_node",
    "sample_host_callback_node",
    "sample_if_node",
    "sample_if_else_node",
    "sample_while_node",
    "sample_switch_node",
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
    ("sample_condition", "sample_condition_alt"),
    ("sample_root_node", "sample_root_node_alt"),
    ("sample_empty_node", "sample_empty_node_alt"),
    ("sample_alloc_node", "sample_alloc_node_alt"),
    ("sample_kernel_node", "sample_kernel_node_alt"),
    ("sample_free_node", "sample_free_node_alt"),
    ("sample_memset_node", "sample_memset_node_alt"),
    ("sample_memcpy_node", "sample_memcpy_node_alt"),
    ("sample_child_graph_node", "sample_child_graph_node_alt"),
    ("sample_event_record_node", "sample_event_record_node_alt"),
    ("sample_event_wait_node", "sample_event_wait_node_alt"),
    ("sample_host_callback_node", "sample_host_callback_node_alt"),
    ("sample_if_node", "sample_if_node_alt"),
    ("sample_if_else_node", "sample_if_else_node_alt"),
    ("sample_while_node", "sample_while_node_alt"),
    ("sample_switch_node", "sample_switch_node_alt"),
]

# Types with public from_handle methods and how to create a copy
FROM_HANDLE_COPIES = [
    ("sample_stream", lambda s: Stream.from_handle(int(s.handle))),
    ("sample_buffer", lambda b: Buffer.from_handle(b.handle, b.size)),
    ("sample_kernel", lambda k: Kernel.from_handle(int(k.handle))),
]

# Types with __reduce__ support (pickle/cloudpickle).
# Event, Buffer, and memory resources are excluded: Event only supports
# IPC-based serialization via multiprocessing reduction; Buffer and memory
# resource __reduce__ use a cross-process registry that doesn't support
# same-process roundtrips.
PICKLE_TYPES = [
    "sample_device",
    "sample_object_code_cubin",
    "sample_ipc_buffer_descriptor",
    "sample_ipc_event_descriptor",
]

PICKLE_MODULES = ["pickle", "cloudpickle"]

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
        r"shmem_size=\d+, is_cooperative=(?:True|False)\)",
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
    ("sample_graphdef", r"<GraphDefinition handle=0x[0-9a-f]+>"),
    ("sample_condition", r"<GraphCondition handle=0x[0-9a-f]+>"),
    ("sample_root_node", r"<GraphNode entry>"),
    ("sample_empty_node", r"<EmptyNode handle=0x[0-9a-f]+>"),
    ("sample_alloc_node", r"<AllocNode handle=0x[0-9a-f]+ dptr=0x[0-9a-f]+ size=\d+>"),
    ("sample_kernel_node", r"<KernelNode handle=0x[0-9a-f]+ kernel=0x[0-9a-f]+>"),
    ("sample_free_node", r"<FreeNode handle=0x[0-9a-f]+ dptr=0x[0-9a-f]+>"),
    ("sample_memset_node", r"<MemsetNode handle=0x[0-9a-f]+ dptr=0x[0-9a-f]+ value=\d+>"),
    ("sample_memcpy_node", r"<MemcpyNode handle=0x[0-9a-f]+ dst=0x[0-9a-f]+\([DH]\) src=0x[0-9a-f]+\([DH]\) size=\d+>"),
    ("sample_child_graph_node", r"<ChildGraphNode handle=0x[0-9a-f]+ child=0x[0-9a-f]+>"),
    ("sample_event_record_node", r"<EventRecordNode handle=0x[0-9a-f]+ event=0x[0-9a-f]+>"),
    ("sample_event_wait_node", r"<EventWaitNode handle=0x[0-9a-f]+ event=0x[0-9a-f]+>"),
    ("sample_host_callback_node", r"<HostCallbackNode handle=0x[0-9a-f]+ callback=\w+>"),
    ("sample_if_node", r"<IfNode handle=0x[0-9a-f]+ condition=0x[0-9a-f]+>"),
    ("sample_if_else_node", r"<IfElseNode handle=0x[0-9a-f]+ condition=0x[0-9a-f]+>"),
    ("sample_while_node", r"<WhileNode handle=0x[0-9a-f]+ condition=0x[0-9a-f]+>"),
    ("sample_switch_node", r"<SwitchNode handle=0x[0-9a-f]+ condition=0x[0-9a-f]+>"),
]


# =============================================================================
# Weak reference tests
# =============================================================================


@pytest.mark.parametrize("sample_object", WEAKREF_TYPES, indirect=True)
def test_weakref_supported(sample_object):
    """Object supports weak references."""
    ref = weakref.ref(sample_object)
    assert ref() is sample_object


# =============================================================================
# Hash tests
# =============================================================================


@pytest.mark.parametrize("sample_object", HASH_TYPES, indirect=True)
def test_hash_consistency(sample_object):
    """Hash is consistent across multiple calls."""
    assert hash(sample_object) == hash(sample_object)


@pytest.mark.parametrize("sample_object_a,sample_object_b", SAME_TYPE_PAIRS, indirect=True)
def test_hash_distinct_same_type(sample_object_a, sample_object_b):
    """Distinct objects of the same type have different hashes."""
    assert hash(sample_object_a) != hash(sample_object_b)  # extremely unlikely


@pytest.mark.parametrize("sample_object_a,sample_object_b", itertools.combinations(HASH_TYPES, 2), indirect=True)
def test_hash_distinct_cross_type(sample_object_a, sample_object_b):
    """Distinct objects of different types have different hashes."""
    assert hash(sample_object_a) != hash(sample_object_b)  # extremely unlikely


# =============================================================================
# Equality tests
# =============================================================================


@pytest.mark.parametrize("sample_object", EQ_TYPES, indirect=True)
def test_equality_basic(sample_object):
    """Object equality: reflexive, not equal to None or other types."""
    assert sample_object == sample_object
    assert sample_object is not None
    assert sample_object != "string"
    if hasattr(sample_object, "handle"):
        assert sample_object != sample_object.handle


@pytest.mark.parametrize("sample_object_a,sample_object_b", itertools.combinations(EQ_TYPES, 2), indirect=True)
def test_no_cross_type_equality(sample_object_a, sample_object_b):
    """No two distinct objects of different types should compare equal."""
    assert sample_object_a != sample_object_b


@pytest.mark.parametrize("sample_object_a,sample_object_b", SAME_TYPE_PAIRS, indirect=True)
def test_same_type_inequality(sample_object_a, sample_object_b):
    """Two distinct objects of the same type should not compare equal."""
    assert sample_object_a is not sample_object_b
    assert sample_object_a != sample_object_b


@pytest.mark.parametrize("sample_object,copy_fn", FROM_HANDLE_COPIES, indirect=["sample_object"])
def test_equality_same_handle(sample_object, copy_fn):
    """Two wrappers around the same handle should compare equal."""
    obj2 = copy_fn(sample_object)
    assert sample_object == obj2
    assert hash(sample_object) == hash(obj2)


# =============================================================================
# Collection usage tests
# =============================================================================


@pytest.mark.parametrize("sample_object", DICT_KEY_TYPES, indirect=True)
def test_usable_as_dict_key(sample_object):
    """Object can be used as a dictionary key."""
    d = {sample_object: "value"}
    assert d[sample_object] == "value"
    assert sample_object in d


@pytest.mark.parametrize("sample_object", DICT_KEY_TYPES, indirect=True)
def test_usable_in_set(sample_object):
    """Object can be added to a set."""
    s = {sample_object}
    assert sample_object in s


@pytest.mark.parametrize("sample_object", WEAKREF_TYPES, indirect=True)
def test_usable_in_weak_value_dict(sample_object):
    """Object can be used as a WeakValueDictionary value."""
    wvd = weakref.WeakValueDictionary()
    wvd["key"] = sample_object
    assert wvd["key"] is sample_object


@pytest.mark.parametrize("sample_object", WEAK_KEY_TYPES, indirect=True)
def test_usable_in_weak_key_dict(sample_object):
    """Object can be used as a WeakKeyDictionary key."""
    wkd = weakref.WeakKeyDictionary()
    wkd[sample_object] = "value"
    assert wkd[sample_object] == "value"


@pytest.mark.parametrize("sample_object", WEAK_KEY_TYPES, indirect=True)
def test_usable_in_weak_set(sample_object):
    """Object can be added to a WeakSet."""
    ws = weakref.WeakSet()
    ws.add(sample_object)
    assert sample_object in ws


# =============================================================================
# Repr tests
# =============================================================================


@pytest.mark.parametrize("sample_object,pattern", REPR_PATTERNS, indirect=["sample_object"])
def test_repr_format(sample_object, pattern):
    """repr() returns a properly formatted string."""
    assert re.fullmatch(pattern, repr(sample_object))


# =============================================================================
# Pickle tests
# =============================================================================


@pytest.mark.parametrize("pickle_module", PICKLE_MODULES)
@pytest.mark.parametrize("sample_object", PICKLE_TYPES, indirect=True)
def test_pickle_roundtrip(sample_object, pickle_module):
    """Object survives a pickle/cloudpickle roundtrip."""
    mod = pytest.importorskip(pickle_module)
    result = mod.loads(mod.dumps(sample_object))
    assert type(result) is type(sample_object)
