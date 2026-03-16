# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Tests for explicit CUDA graph construction (GraphDef and GraphNode)."""

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest
from helpers.graph_kernels import compile_common_kernels
from helpers.misc import try_create_condition

from cuda.core import Device, LaunchConfig
from cuda.core._graph import GraphCompleteOptions, GraphDebugPrintOptions
from cuda.core._graph._graphdef import (
    AllocNode,
    ChildGraphNode,
    ConditionalNode,
    EmptyNode,
    EventRecordNode,
    EventWaitNode,
    FreeNode,
    GraphAllocOptions,
    GraphDef,
    GraphNode,
    HostCallbackNode,
    IfElseNode,
    IfNode,
    KernelNode,
    MemcpyNode,
    MemsetNode,
    SwitchNode,
    WhileNode,
)

ALLOC_SIZE = 1024


def _skip_if_no_mempool():
    if not Device(0).properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")


def _driver_has_node_get_params():
    from cuda.bindings import driver as drv

    return drv.cuDriverGetVersion()[1] >= 13020


_HAS_NODE_GET_PARAMS = _driver_has_node_get_params()


def _bindings_major_version():
    from cuda.core._utils.cuda_utils import get_binding_version

    return get_binding_version()[0]


_BINDINGS_MAJOR = _bindings_major_version()


# =============================================================================
# GraphSpec — representative graph topologies
# =============================================================================


@dataclass
class GraphSpec:
    """Describes a graph topology with expected structural properties."""

    name: str
    graphdef: GraphDef
    named_nodes: dict = field(default_factory=dict)
    expected_edges: set = field(default_factory=set)
    expected_pred: dict = field(default_factory=dict)
    expected_succ: dict = field(default_factory=dict)


def _build_empty():
    """No nodes, no edges."""
    return GraphSpec("empty", GraphDef())


def _build_single():
    """One alloc node, no edges."""
    g = GraphDef()
    a = g.alloc(ALLOC_SIZE)
    return GraphSpec(
        "single",
        g,
        named_nodes={"a": a},
        expected_edges=set(),
        expected_pred={"a": set()},
        expected_succ={"a": set()},
    )


def _build_chain():
    """Linear chain: a -> b -> c."""
    g = GraphDef()
    a = g.alloc(ALLOC_SIZE)
    b = a.alloc(ALLOC_SIZE)
    c = b.alloc(ALLOC_SIZE)
    return GraphSpec(
        "chain",
        g,
        named_nodes={"a": a, "b": b, "c": c},
        expected_edges={("a", "b"), ("b", "c")},
        expected_pred={"a": set(), "b": {"a"}, "c": {"b"}},
        expected_succ={"a": {"b"}, "b": {"c"}, "c": set()},
    )


def _build_fan_out():
    """One node feeds three: a -> {b, c, d}."""
    g = GraphDef()
    a = g.alloc(ALLOC_SIZE)
    b = a.alloc(ALLOC_SIZE)
    c = a.alloc(ALLOC_SIZE)
    d = a.alloc(ALLOC_SIZE)
    return GraphSpec(
        "fan_out",
        g,
        named_nodes={"a": a, "b": b, "c": c, "d": d},
        expected_edges={("a", "b"), ("a", "c"), ("a", "d")},
        expected_pred={"a": set(), "b": {"a"}, "c": {"a"}, "d": {"a"}},
        expected_succ={"a": {"b", "c", "d"}, "b": set(), "c": set(), "d": set()},
    )


def _build_fan_in():
    """Three entry nodes merge: {a, b, c} -> d (join)."""
    g = GraphDef()
    a = g.alloc(ALLOC_SIZE)
    b = g.alloc(ALLOC_SIZE)
    c = g.alloc(ALLOC_SIZE)
    d = g.join(a, b, c)
    return GraphSpec(
        "fan_in",
        g,
        named_nodes={"a": a, "b": b, "c": c, "d": d},
        expected_edges={("a", "d"), ("b", "d"), ("c", "d")},
        expected_pred={"a": set(), "b": set(), "c": set(), "d": {"a", "b", "c"}},
        expected_succ={"a": {"d"}, "b": {"d"}, "c": {"d"}, "d": set()},
    )


def _build_diamond():
    """Diamond: a -> {b, c} -> d (join)."""
    g = GraphDef()
    a = g.alloc(ALLOC_SIZE)
    b = a.alloc(ALLOC_SIZE)
    c = a.alloc(ALLOC_SIZE)
    d = b.join(c)
    return GraphSpec(
        "diamond",
        g,
        named_nodes={"a": a, "b": b, "c": c, "d": d},
        expected_edges={("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")},
        expected_pred={"a": set(), "b": {"a"}, "c": {"a"}, "d": {"b", "c"}},
        expected_succ={"a": {"b", "c"}, "b": {"d"}, "c": {"d"}, "d": set()},
    )


def _build_disconnected():
    """Two independent entry nodes: a, b."""
    g = GraphDef()
    a = g.alloc(ALLOC_SIZE)
    b = g.alloc(ALLOC_SIZE)
    return GraphSpec(
        "disconnected",
        g,
        named_nodes={"a": a, "b": b},
        expected_edges=set(),
        expected_pred={"a": set(), "b": set()},
        expected_succ={"a": set(), "b": set()},
    )


_ALL_BUILDERS = [
    pytest.param(_build_empty, id="empty"),
    pytest.param(_build_single, id="single"),
    pytest.param(_build_chain, id="chain"),
    pytest.param(_build_fan_out, id="fan_out"),
    pytest.param(_build_fan_in, id="fan_in"),
    pytest.param(_build_diamond, id="diamond"),
    pytest.param(_build_disconnected, id="disconnected"),
]

_NONEMPTY_BUILDERS = [p for p in _ALL_BUILDERS if p.values[0] is not _build_empty]


@pytest.fixture(params=_ALL_BUILDERS)
def graph_spec(request, init_cuda):
    if request.param is not _build_empty:
        _skip_if_no_mempool()
    return request.param()


@pytest.fixture(params=_NONEMPTY_BUILDERS)
def nonempty_graph_spec(request, init_cuda):
    _skip_if_no_mempool()
    return request.param()


# =============================================================================
# NodeSpec — representative node types
# =============================================================================


@dataclass
class NodeSpec:
    """Describes a node type with expected properties.

    The builder returns (node, expected_attrs) where expected_attrs maps
    property names to expected values. Callable values are treated as
    predicates (e.g., ``lambda v: v != 0``).
    """

    name: str
    expected_class: type
    expected_type_name: str
    builder: Callable[[GraphDef], tuple[GraphNode, dict]]
    reconstructed_class: type | None = None
    needs_mempool: bool = True

    @property
    def roundtrip_class(self):
        """Class expected after reconstruction from the driver."""
        return self.reconstructed_class or self.expected_class


def _build_empty_node(g):
    a = g.alloc(ALLOC_SIZE)
    b = g.alloc(ALLOC_SIZE)
    return g.join(a, b), {}


def _build_kernel_node(g):
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=(2, 3, 1), block=(32, 4, 1), shmem_size=128)
    entry = g.alloc(ALLOC_SIZE)
    node = entry.launch(config, kernel)
    return node, {
        "grid": (2, 3, 1),
        "block": (32, 4, 1),
        "shmem_size": 128,
        "kernel": kernel,
        "config": config,
    }


def _build_alloc_node(g):
    device_id = Device().device_id
    entry = g.alloc(ALLOC_SIZE)
    node = entry.alloc(ALLOC_SIZE)
    return node, {
        "dptr": lambda v: v != 0,
        "bytesize": ALLOC_SIZE,
        "device_id": device_id,
        "memory_type": "device",
        "peer_access": (),
        "options": GraphAllocOptions(device=device_id, memory_type="device"),
    }


def _build_alloc_managed_node(g):
    device_id = Device().device_id
    options = GraphAllocOptions(memory_type="managed")
    entry = g.alloc(ALLOC_SIZE)
    node = entry.alloc(ALLOC_SIZE, options)
    return node, {
        "dptr": lambda v: v != 0,
        "bytesize": ALLOC_SIZE,
        "device_id": device_id,
        "memory_type": "managed",
        "peer_access": (),
        "options": GraphAllocOptions(device=device_id, memory_type="managed"),
    }


def _build_free_node(g):
    alloc = g.alloc(ALLOC_SIZE)
    node = alloc.free(alloc.dptr)
    return node, {
        "dptr": alloc.dptr,
    }


def _build_memset_node(g):
    alloc = g.alloc(ALLOC_SIZE)
    node = alloc.memset(alloc.dptr, 42, ALLOC_SIZE)
    return node, {
        "dptr": alloc.dptr,
        "value": 42,
        "element_size": 1,
        "width": ALLOC_SIZE,
        "height": 1,
        "pitch": 0,
    }


def _build_memset_node_u16(g):
    alloc = g.alloc(ALLOC_SIZE)
    node = alloc.memset(alloc.dptr, b"\xab\xcd", ALLOC_SIZE // 2)
    return node, {
        "dptr": alloc.dptr,
        "value": int.from_bytes(b"\xab\xcd", byteorder="little"),
        "element_size": 2,
        "width": ALLOC_SIZE // 2,
        "height": 1,
        "pitch": 0,
    }


def _build_memset_node_u32(g):
    alloc = g.alloc(ALLOC_SIZE)
    node = alloc.memset(alloc.dptr, b"\x01\x02\x03\x04", ALLOC_SIZE // 4)
    return node, {
        "dptr": alloc.dptr,
        "value": int.from_bytes(b"\x01\x02\x03\x04", byteorder="little"),
        "element_size": 4,
        "width": ALLOC_SIZE // 4,
        "height": 1,
        "pitch": 0,
    }


def _build_memset_node_2d(g):
    rows = 4
    cols = ALLOC_SIZE // rows
    alloc = g.alloc(ALLOC_SIZE)
    node = alloc.memset(alloc.dptr, 0xFF, cols, height=rows, pitch=cols)
    return node, {
        "dptr": alloc.dptr,
        "value": 0xFF,
        "element_size": 1,
        "width": cols,
        "height": rows,
        "pitch": cols,
    }


def _build_event_record_node(g):
    event = Device().create_event()
    entry = g.alloc(ALLOC_SIZE)
    node = entry.record_event(event)
    return node, {
        "event": event,
    }


def _build_event_wait_node(g):
    event = Device().create_event()
    entry = g.alloc(ALLOC_SIZE)
    node = entry.wait_event(event)
    return node, {
        "event": event,
    }


def _build_memcpy_node(g):
    src_alloc = g.alloc(ALLOC_SIZE)
    dst_alloc = g.alloc(ALLOC_SIZE)
    dep = g.join(src_alloc, dst_alloc)
    node = dep.memcpy(dst_alloc.dptr, src_alloc.dptr, ALLOC_SIZE)
    return node, {
        "dst": dst_alloc.dptr,
        "src": src_alloc.dptr,
        "size": ALLOC_SIZE,
    }


def _build_host_callback_node(g):
    def my_callback():
        pass

    node = g.callback(my_callback)
    return node, {
        "callback_fn": lambda v: v is my_callback,
    }


def _build_host_callback_cfunc_node(g):
    import ctypes

    CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

    @CALLBACK
    def noop(data):
        pass

    node = g.callback(noop)
    return node, {}


def _build_child_graph_node(g):
    child = GraphDef()
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)
    child.launch(config, kernel)
    child.launch(config, kernel)
    node = g.embed(child)
    return node, {
        "child_graph": lambda v: isinstance(v, GraphDef) and len(v.nodes()) == 2,
    }


def _build_if_cond_node(g):
    condition = try_create_condition(g)
    node = g.if_cond(condition)
    return node, {
        "condition": condition,
        "cond_type": "if",
        "branches": lambda v: isinstance(v, tuple) and len(v) == 1,
        "then": lambda v: isinstance(v, GraphDef),
    }


def _build_if_else_node(g):
    condition = try_create_condition(g)
    node = g.if_else(condition)
    return node, {
        "condition": condition,
        "cond_type": "if",
        "branches": lambda v: isinstance(v, tuple) and len(v) == 2,
        "then": lambda v: isinstance(v, GraphDef),
        "else_": lambda v: isinstance(v, GraphDef),
    }


def _build_while_loop_node(g):
    condition = try_create_condition(g)
    node = g.while_loop(condition)
    return node, {
        "condition": condition,
        "cond_type": "while",
        "branches": lambda v: isinstance(v, tuple) and len(v) == 1,
        "body": lambda v: isinstance(v, GraphDef),
    }


def _build_switch_node(g):
    condition = try_create_condition(g)
    node = g.switch(condition, 3)
    return node, {
        "condition": condition,
        "cond_type": "switch",
        "branches": lambda v: isinstance(v, tuple) and len(v) == 3,
    }


_NODE_SPECS = [
    pytest.param(NodeSpec("empty", EmptyNode, "CU_GRAPH_NODE_TYPE_EMPTY", _build_empty_node), id="empty"),
    pytest.param(NodeSpec("kernel", KernelNode, "CU_GRAPH_NODE_TYPE_KERNEL", _build_kernel_node), id="kernel"),
    pytest.param(NodeSpec("alloc", AllocNode, "CU_GRAPH_NODE_TYPE_MEM_ALLOC", _build_alloc_node), id="alloc"),
    pytest.param(
        NodeSpec("alloc_managed", AllocNode, "CU_GRAPH_NODE_TYPE_MEM_ALLOC", _build_alloc_managed_node),
        id="alloc_managed",
        marks=pytest.mark.skipif(_BINDINGS_MAJOR < 13, reason="managed alloc requires CUDA 13.0+ bindings"),
    ),
    pytest.param(NodeSpec("free", FreeNode, "CU_GRAPH_NODE_TYPE_MEM_FREE", _build_free_node), id="free"),
    pytest.param(NodeSpec("memset", MemsetNode, "CU_GRAPH_NODE_TYPE_MEMSET", _build_memset_node), id="memset"),
    pytest.param(
        NodeSpec("memset_u16", MemsetNode, "CU_GRAPH_NODE_TYPE_MEMSET", _build_memset_node_u16), id="memset_u16"
    ),
    pytest.param(
        NodeSpec("memset_u32", MemsetNode, "CU_GRAPH_NODE_TYPE_MEMSET", _build_memset_node_u32), id="memset_u32"
    ),
    pytest.param(NodeSpec("memset_2d", MemsetNode, "CU_GRAPH_NODE_TYPE_MEMSET", _build_memset_node_2d), id="memset_2d"),
    pytest.param(
        NodeSpec("memcpy", MemcpyNode, "CU_GRAPH_NODE_TYPE_MEMCPY", _build_memcpy_node),
        id="memcpy",
    ),
    pytest.param(
        NodeSpec(
            "child_graph", ChildGraphNode, "CU_GRAPH_NODE_TYPE_GRAPH", _build_child_graph_node, needs_mempool=False
        ),
        id="child_graph",
    ),
    pytest.param(
        NodeSpec(
            "host_callback", HostCallbackNode, "CU_GRAPH_NODE_TYPE_HOST", _build_host_callback_node, needs_mempool=False
        ),
        id="host_callback",
    ),
    pytest.param(
        NodeSpec(
            "host_callback_cfunc",
            HostCallbackNode,
            "CU_GRAPH_NODE_TYPE_HOST",
            _build_host_callback_cfunc_node,
            needs_mempool=False,
        ),
        id="host_callback_cfunc",
    ),
    pytest.param(
        NodeSpec("event_record", EventRecordNode, "CU_GRAPH_NODE_TYPE_EVENT_RECORD", _build_event_record_node),
        id="event_record",
    ),
    pytest.param(
        NodeSpec("event_wait", EventWaitNode, "CU_GRAPH_NODE_TYPE_WAIT_EVENT", _build_event_wait_node),
        id="event_wait",
    ),
    pytest.param(
        NodeSpec(
            "if_cond",
            IfNode,
            "CU_GRAPH_NODE_TYPE_CONDITIONAL",
            _build_if_cond_node,
            reconstructed_class=IfNode if _HAS_NODE_GET_PARAMS else ConditionalNode,
            needs_mempool=False,
        ),
        id="if_cond",
    ),
    pytest.param(
        NodeSpec(
            "if_else",
            IfElseNode,
            "CU_GRAPH_NODE_TYPE_CONDITIONAL",
            _build_if_else_node,
            reconstructed_class=IfElseNode if _HAS_NODE_GET_PARAMS else ConditionalNode,
            needs_mempool=False,
        ),
        id="if_else",
    ),
    pytest.param(
        NodeSpec(
            "while_loop",
            WhileNode,
            "CU_GRAPH_NODE_TYPE_CONDITIONAL",
            _build_while_loop_node,
            reconstructed_class=WhileNode if _HAS_NODE_GET_PARAMS else ConditionalNode,
            needs_mempool=False,
        ),
        id="while_loop",
    ),
    pytest.param(
        NodeSpec(
            "switch",
            SwitchNode,
            "CU_GRAPH_NODE_TYPE_CONDITIONAL",
            _build_switch_node,
            reconstructed_class=SwitchNode if _HAS_NODE_GET_PARAMS else ConditionalNode,
            needs_mempool=False,
        ),
        id="switch",
    ),
]


@pytest.fixture(params=_NODE_SPECS)
def node_spec(request, init_cuda):
    spec = request.param
    if spec.needs_mempool:
        _skip_if_no_mempool()
    g = GraphDef()
    node, expected_attrs = spec.builder(g)
    return spec, g, node, expected_attrs


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_graphdef(init_cuda):
    """A sample GraphDef for standalone tests."""
    return GraphDef()


@pytest.fixture
def dot_file(tmp_path):
    """Temporary DOT file path, cleaned up after test."""
    path = tmp_path / "graph.dot"
    yield path
    path.unlink(missing_ok=True)


# =============================================================================
# Topology tests (parameterized over graph specs)
# =============================================================================


def test_node_count(graph_spec):
    """Graph contains the expected number of nodes."""
    assert len(graph_spec.graphdef.nodes()) == len(graph_spec.named_nodes)


def test_nodes_match(nonempty_graph_spec):
    """nodes() returns exactly the expected nodes."""
    spec = nonempty_graph_spec
    assert set(spec.graphdef.nodes()) == set(spec.named_nodes.values())


def test_edges(graph_spec):
    """edges() returns exactly the expected edges."""
    spec = graph_spec
    node_to_name = {v: k for k, v in spec.named_nodes.items()}
    actual = {(node_to_name[a], node_to_name[b]) for a, b in spec.graphdef.edges()}
    assert actual == spec.expected_edges


def test_pred(nonempty_graph_spec):
    """Each node has the expected predecessors."""
    spec = nonempty_graph_spec
    node_to_name = {v: k for k, v in spec.named_nodes.items()}
    for name, node in spec.named_nodes.items():
        actual = {node_to_name[p] for p in node.pred}
        assert actual == spec.expected_pred[name], f"pred mismatch for node {name}"


def test_succ(nonempty_graph_spec):
    """Each node has the expected successors."""
    spec = nonempty_graph_spec
    node_to_name = {v: k for k, v in spec.named_nodes.items()}
    for name, node in spec.named_nodes.items():
        actual = {node_to_name[s] for s in node.succ}
        assert actual == spec.expected_succ[name], f"succ mismatch for node {name}"


def test_node_graph_property(nonempty_graph_spec):
    """Every node's .graph property returns the parent GraphDef."""
    spec = nonempty_graph_spec
    for name, node in spec.named_nodes.items():
        assert node.graph == spec.graphdef, f"graph mismatch for node {name}"


# =============================================================================
# Node type tests (parameterized over node specs)
# =============================================================================


def test_node_isinstance(node_spec):
    """GraphNode is an instance of the expected subclass."""
    spec, g, node, _ = node_spec
    assert isinstance(node, spec.expected_class)
    assert isinstance(node, GraphNode)


def test_node_type_property(node_spec):
    """Node.type returns the expected CUgraphNodeType."""
    spec, g, node, _ = node_spec
    assert node.type.name == spec.expected_type_name


def test_node_type_preserved_by_nodes(node_spec):
    """Node type is preserved when retrieved via graphdef.nodes()."""
    spec, g, node, _ = node_spec
    all_nodes = g.nodes()
    matched = [n for n in all_nodes if n == node]
    assert len(matched) == 1
    assert isinstance(matched[0], spec.roundtrip_class)


def test_node_type_preserved_by_pred_succ(node_spec):
    """Node type is preserved when retrieved via pred/succ traversal."""
    spec, g, node, _ = node_spec
    for predecessor in node.pred:
        matched = [s for s in predecessor.succ if s == node]
        assert len(matched) == 1
        assert isinstance(matched[0], spec.roundtrip_class)


def test_node_attrs(node_spec):
    """Type-specific attributes have expected values after construction."""
    spec, g, node, expected_attrs = node_spec
    if not expected_attrs:
        pytest.skip("no type-specific attributes")
    for attr, expected in expected_attrs.items():
        actual = getattr(node, attr)
        if callable(expected):
            assert expected(actual), f"{spec.name}.{attr}: check failed (got {actual})"
        else:
            assert actual == expected, f"{spec.name}.{attr}: expected {expected}, got {actual}"


def test_node_attrs_preserved_by_nodes(node_spec):
    """Type-specific attributes survive round-trip through graphdef.nodes()."""
    spec, g, node, expected_attrs = node_spec
    if not expected_attrs:
        pytest.skip("no type-specific attributes")
    if spec.roundtrip_class != spec.expected_class:
        pytest.skip("reconstructed type differs — attrs not preserved")
    retrieved = next(n for n in g.nodes() if n == node)
    for attr in expected_attrs:
        assert getattr(retrieved, attr) == getattr(node, attr), f"{spec.name}.{attr} not preserved by nodes()"


# =============================================================================
# GraphDef basics
# =============================================================================


def test_graphdef_handle_valid(sample_graphdef):
    """GraphDef has a valid non-null handle."""
    assert sample_graphdef.handle is not None
    assert int(sample_graphdef.handle) != 0


def test_graphdef_entry_is_virtual(sample_graphdef):
    """Internal entry node is virtual (no pred/succ, type is None)."""
    entry = sample_graphdef._entry
    assert isinstance(entry, GraphNode)
    assert entry.pred == ()
    assert entry.succ == ()
    assert entry.type is None


# =============================================================================
# Alloc/free API
# =============================================================================


def test_alloc_zero_size_fails(sample_graphdef):
    """Alloc with zero size raises error (CUDA limitation)."""
    _skip_if_no_mempool()
    from cuda.core._utils.cuda_utils import CUDAError

    with pytest.raises(CUDAError):
        sample_graphdef.alloc(0)


def test_free_creates_dependency(sample_graphdef):
    """Free node depends on its predecessor."""
    _skip_if_no_mempool()
    alloc = sample_graphdef.alloc(ALLOC_SIZE)
    free = alloc.free(alloc.dptr)
    assert alloc in free.pred


def test_alloc_free_chain(sample_graphdef):
    """Alloc and free can be chained."""
    _skip_if_no_mempool()
    a1 = sample_graphdef.alloc(ALLOC_SIZE)
    a2 = a1.alloc(ALLOC_SIZE)
    f2 = a2.free(a2.dptr)
    f1 = f2.free(a1.dptr)
    assert a1 in a2.pred
    assert a2 in f2.pred
    assert f2 in f1.pred


# =============================================================================
# Allocation options (error cases, input variants, multi-GPU)
# =============================================================================


def test_alloc_memory_type_invalid(sample_graphdef):
    """Invalid memory type raises ValueError."""
    options = GraphAllocOptions(memory_type="invalid")
    with pytest.raises(ValueError, match="Invalid memory_type"):
        sample_graphdef.alloc(ALLOC_SIZE, options)


@pytest.mark.parametrize(
    "device_spec",
    [
        pytest.param(lambda d: d.device_id, id="device_id"),
        pytest.param(lambda d: d, id="Device_object"),
    ],
)
def test_alloc_device_option(sample_graphdef, device_spec):
    """Device can be specified as int or Device object."""
    _skip_if_no_mempool()
    device = Device()
    options = GraphAllocOptions(device=device_spec(device))
    node = sample_graphdef.alloc(ALLOC_SIZE, options)
    assert node.dptr != 0


def test_alloc_peer_access(mempool_device_x2):
    """AllocNode.peer_access reflects requested peers."""
    d0, d1 = mempool_device_x2
    g = GraphDef()
    options = GraphAllocOptions(device=d0.device_id, peer_access=[d1.device_id])
    node = g.alloc(ALLOC_SIZE, options)
    assert d1.device_id in node.peer_access


# =============================================================================
# Join API
# =============================================================================


@pytest.mark.parametrize("num_branches", [2, 3, 5])
def test_join_merges_branches(sample_graphdef, num_branches):
    """join() with multiple branches creates correct dependencies."""
    _skip_if_no_mempool()
    branches = [sample_graphdef.alloc(ALLOC_SIZE) for _ in range(num_branches)]
    joined = sample_graphdef.join(*branches)
    assert isinstance(joined, EmptyNode)
    assert set(joined.pred) == set(branches)


# =============================================================================
# Kernel launch
# =============================================================================


def test_launch_creates_node(sample_graphdef):
    """launch() creates a KernelNode."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)
    node = sample_graphdef.launch(config, kernel)
    assert isinstance(node, KernelNode)


def test_launch_chain_dependencies(sample_graphdef):
    """Chained launches create correct dependencies."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)
    n1 = sample_graphdef.launch(config, kernel)
    n2 = n1.launch(config, kernel)
    n3 = n2.launch(config, kernel)
    assert n1 in n2.pred
    assert n2 in n3.pred
    assert n1 not in n3.pred


# =============================================================================
# Instantiation and execution
# =============================================================================

_SENTINEL_UPLOAD_STREAM = "USE_TEST_STREAM"

_INSTANTIATE_ONLY_OPTIONS = [
    pytest.param({"no_arg": True}, id="no-options"),
    pytest.param({"options": None}, id="none-options"),
    pytest.param(
        {"options": GraphCompleteOptions(auto_free_on_launch=True, use_node_priority=True)},
        id="all-bool-flags",
    ),
]

_EXECUTE_OPTIONS = [
    pytest.param({}, id="no-options"),
    pytest.param({"options": GraphCompleteOptions(auto_free_on_launch=True)}, id="auto-free"),
    pytest.param({"options": GraphCompleteOptions(use_node_priority=True)}, id="node-priority"),
    pytest.param(
        {"options": GraphCompleteOptions(upload_stream=_SENTINEL_UPLOAD_STREAM)},
        id="upload-stream",
    ),
]


def _instantiate(graphdef, kwargs, stream=None):
    """Call graphdef.instantiate() with the given kwargs, resolving sentinels."""
    if "no_arg" in kwargs:
        return graphdef.instantiate()
    opts = kwargs.get("options")
    if opts is not None and opts.upload_stream == _SENTINEL_UPLOAD_STREAM:
        opts = GraphCompleteOptions(
            auto_free_on_launch=opts.auto_free_on_launch,
            upload_stream=stream,
            device_launch=opts.device_launch,
            use_node_priority=opts.use_node_priority,
        )
    return graphdef.instantiate(options=opts)


def _instantiate_and_upload(graphdef, kwargs, stream):
    """Instantiate and upload, handling upload_stream option."""
    graph = _instantiate(graphdef, kwargs, stream)
    if not (kwargs.get("options") and kwargs["options"].upload_stream):
        graph.upload(stream)
    return graph


@pytest.mark.parametrize("inst_kwargs", _INSTANTIATE_ONLY_OPTIONS)
def test_instantiate_empty_graph(sample_graphdef, inst_kwargs):
    """Empty graph can be instantiated."""
    graph = _instantiate(sample_graphdef, inst_kwargs)
    assert graph is not None


@pytest.mark.parametrize("inst_kwargs", _INSTANTIATE_ONLY_OPTIONS)
def test_instantiate_with_nodes(sample_graphdef, inst_kwargs):
    """Graph with nodes can be instantiated."""
    _skip_if_no_mempool()
    sample_graphdef.alloc(ALLOC_SIZE)
    sample_graphdef.alloc(ALLOC_SIZE)
    graph = _instantiate(sample_graphdef, inst_kwargs)
    assert graph is not None


@pytest.mark.skipif(not Device(0).properties.unified_addressing, reason="requires unified addressing")
def test_instantiate_and_execute_kernel_device_launch(sample_graphdef):
    """Kernel-only graph can be instantiated with device_launch flag."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)
    sample_graphdef.launch(config, kernel)

    opts = GraphCompleteOptions(device_launch=True)
    graph = sample_graphdef.instantiate(options=opts)

    stream = Device().create_stream()
    graph.upload(stream)
    graph.launch(stream)
    stream.sync()


@pytest.mark.parametrize("inst_kwargs", _EXECUTE_OPTIONS)
def test_instantiate_and_execute_kernel(sample_graphdef, inst_kwargs):
    """Graph with kernel can be instantiated and executed."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)
    sample_graphdef.launch(config, kernel)

    stream = Device().create_stream()
    graph = _instantiate_and_upload(sample_graphdef, inst_kwargs, stream)
    graph.launch(stream)
    stream.sync()


@pytest.mark.parametrize("inst_kwargs", _EXECUTE_OPTIONS)
def test_instantiate_and_execute_alloc_free(sample_graphdef, inst_kwargs):
    """Graph with alloc/free can be executed."""
    _skip_if_no_mempool()
    alloc = sample_graphdef.alloc(ALLOC_SIZE)
    alloc.free(alloc.dptr)

    stream = Device().create_stream()
    graph = _instantiate_and_upload(sample_graphdef, inst_kwargs, stream)
    graph.launch(stream)
    stream.sync()


@pytest.mark.parametrize("inst_kwargs", _EXECUTE_OPTIONS)
def test_instantiate_and_execute_memset(sample_graphdef, inst_kwargs):
    """Graph with alloc/memset/free can be executed."""
    _skip_if_no_mempool()
    alloc = sample_graphdef.alloc(ALLOC_SIZE)
    ms = alloc.memset(alloc.dptr, 0xAB, ALLOC_SIZE)
    ms.free(alloc.dptr)

    stream = Device().create_stream()
    graph = _instantiate_and_upload(sample_graphdef, inst_kwargs, stream)
    graph.launch(stream)
    stream.sync()


@pytest.mark.parametrize("inst_kwargs", _EXECUTE_OPTIONS)
def test_instantiate_and_execute_memcpy(sample_graphdef, inst_kwargs):
    """Graph with alloc/memset/memcpy/free can be executed and data is copied."""
    _skip_if_no_mempool()
    import ctypes

    src_alloc = sample_graphdef.alloc(ALLOC_SIZE)
    dst_alloc = sample_graphdef.alloc(ALLOC_SIZE)
    dep = sample_graphdef.join(src_alloc, dst_alloc)
    ms = dep.memset(src_alloc.dptr, 0xAB, ALLOC_SIZE)
    cp = ms.memcpy(dst_alloc.dptr, src_alloc.dptr, ALLOC_SIZE)
    cp.free(src_alloc.dptr)

    stream = Device().create_stream()
    graph = _instantiate_and_upload(sample_graphdef, inst_kwargs, stream)
    graph.launch(stream)
    stream.sync()

    host_buf = (ctypes.c_ubyte * ALLOC_SIZE)()
    from cuda.bindings import driver as drv

    drv.cuMemcpyDtoH(host_buf, dst_alloc.dptr, ALLOC_SIZE)
    assert all(b == 0xAB for b in host_buf)


def test_instantiate_and_execute_child_graph(sample_graphdef):
    """Graph with embedded child graph can be executed."""
    child = GraphDef()
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)
    child.launch(config, kernel)

    sample_graphdef.embed(child)
    graph = sample_graphdef.instantiate()

    stream = Device().create_stream()
    graph.upload(stream)
    graph.launch(stream)
    stream.sync()


def test_instantiate_and_execute_host_callback(sample_graphdef):
    """Graph with host callback can be executed and callback is invoked."""
    results = []

    def my_callback():
        results.append(42)

    sample_graphdef.callback(my_callback)
    graph = sample_graphdef.instantiate()

    stream = Device().create_stream()
    graph.upload(stream)
    graph.launch(stream)
    stream.sync()

    assert results == [42]


def test_instantiate_and_execute_host_callback_cfunc(sample_graphdef):
    """Graph with ctypes function pointer callback can be executed."""
    import ctypes

    CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    called = [False]

    @CALLBACK
    def raw_fn(data):
        called[0] = True

    sample_graphdef.callback(raw_fn)
    graph = sample_graphdef.instantiate()

    stream = Device().create_stream()
    graph.upload(stream)
    graph.launch(stream)
    stream.sync()

    assert called[0]


def test_host_callback_cfunc_with_user_data(sample_graphdef):
    """Host callback with bytes user_data passes data to C function."""
    import ctypes

    CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    result = [0]

    @CALLBACK
    def read_byte(data):
        result[0] = ctypes.cast(data, ctypes.POINTER(ctypes.c_uint8))[0]

    sample_graphdef.callback(read_byte, user_data=bytes([0xAB]))
    graph = sample_graphdef.instantiate()

    stream = Device().create_stream()
    graph.upload(stream)
    graph.launch(stream)
    stream.sync()

    assert result[0] == 0xAB


def test_host_callback_user_data_rejected_for_python_callable(sample_graphdef):
    """user_data is rejected for Python callables."""
    with pytest.raises(ValueError, match="user_data is only supported"):
        sample_graphdef.callback(lambda: None, user_data=b"hello")


def test_instantiate_and_execute_event_record_wait(sample_graphdef):
    """Graph with event record and wait nodes can be executed."""
    event = Device().create_event()
    rec = sample_graphdef.record_event(event)
    rec.wait_event(event)
    graph = sample_graphdef.instantiate()

    stream = Device().create_stream()
    graph.upload(stream)
    graph.launch(stream)
    stream.sync()


# =============================================================================
# Conditional nodes
# =============================================================================


def _skip_unless_cc_90():
    if Device(0).compute_capability < (9, 0):
        pytest.skip("Conditional node execution requires CC >= 9.0 (Hopper)")


def test_instantiate_and_execute_if_cond(sample_graphdef):
    """If-conditional node: body executes only when condition is non-zero."""
    _skip_unless_cc_90()
    _skip_if_no_mempool()
    import ctypes

    from helpers.graph_kernels import compile_conditional_kernels

    condition = sample_graphdef.create_condition(default_value=0)
    mod = compile_conditional_kernels(int)
    set_handle = mod.get_kernel("set_handle")
    add_one = mod.get_kernel("add_one")

    alloc = sample_graphdef.alloc(ctypes.sizeof(ctypes.c_int))
    ms = alloc.memset(alloc.dptr, 0, ctypes.sizeof(ctypes.c_int))
    setter = ms.launch(LaunchConfig(grid=1, block=1), set_handle, condition.handle, 1)
    if_node = setter.if_cond(condition)
    if_node.then.launch(LaunchConfig(grid=1, block=1), add_one, alloc.dptr)

    graph = sample_graphdef.instantiate()
    stream = Device().create_stream()
    graph.upload(stream)
    graph.launch(stream)
    stream.sync()

    result = (ctypes.c_int * 1)()
    from cuda.bindings import driver as drv

    drv.cuMemcpyDtoH(result, alloc.dptr, ctypes.sizeof(ctypes.c_int))
    assert result[0] == 1


def test_instantiate_and_execute_if_else(sample_graphdef):
    """If-else node: then or else branch executes based on condition."""
    _skip_unless_cc_90()
    _skip_if_no_mempool()
    import ctypes

    from helpers.graph_kernels import compile_conditional_kernels

    condition = sample_graphdef.create_condition(default_value=0)
    mod = compile_conditional_kernels(int)
    set_handle = mod.get_kernel("set_handle")
    add_one = mod.get_kernel("add_one")

    alloc = sample_graphdef.alloc(ctypes.sizeof(ctypes.c_int))
    ms = alloc.memset(alloc.dptr, 0, ctypes.sizeof(ctypes.c_int))
    setter = ms.launch(LaunchConfig(grid=1, block=1), set_handle, condition.handle, 0)
    ie_node = setter.if_else(condition)
    ie_node.then.launch(LaunchConfig(grid=1, block=1), add_one, alloc.dptr)
    n1 = ie_node.else_.launch(LaunchConfig(grid=1, block=1), add_one, alloc.dptr)
    n1.launch(LaunchConfig(grid=1, block=1), add_one, alloc.dptr)

    graph = sample_graphdef.instantiate()
    stream = Device().create_stream()
    graph.upload(stream)
    graph.launch(stream)
    stream.sync()

    result = (ctypes.c_int * 1)()
    from cuda.bindings import driver as drv

    drv.cuMemcpyDtoH(result, alloc.dptr, ctypes.sizeof(ctypes.c_int))
    assert result[0] == 2


def test_instantiate_and_execute_switch(sample_graphdef):
    """Switch node: selected branch executes based on condition value."""
    _skip_unless_cc_90()
    _skip_if_no_mempool()
    import ctypes

    from helpers.graph_kernels import compile_conditional_kernels

    condition = sample_graphdef.create_condition(default_value=0)
    mod = compile_conditional_kernels(int)
    set_handle = mod.get_kernel("set_handle")
    add_one = mod.get_kernel("add_one")

    alloc = sample_graphdef.alloc(ctypes.sizeof(ctypes.c_int))
    ms = alloc.memset(alloc.dptr, 0, ctypes.sizeof(ctypes.c_int))
    setter = ms.launch(LaunchConfig(grid=1, block=1), set_handle, condition.handle, 2)
    sw_node = setter.switch(condition, 4)
    for branch in sw_node.branches:
        branch.launch(LaunchConfig(grid=1, block=1), add_one, alloc.dptr)

    graph = sample_graphdef.instantiate()
    stream = Device().create_stream()
    graph.upload(stream)
    graph.launch(stream)
    stream.sync()

    result = (ctypes.c_int * 1)()
    from cuda.bindings import driver as drv

    drv.cuMemcpyDtoH(result, alloc.dptr, ctypes.sizeof(ctypes.c_int))
    assert result[0] == 1


def test_conditional_node_type_preserved_by_nodes(sample_graphdef):
    """Conditional nodes appear as ConditionalNode base when read back from graph."""
    condition = try_create_condition(sample_graphdef)
    if_node = sample_graphdef.if_cond(condition)
    assert isinstance(if_node, IfNode)

    all_nodes = sample_graphdef.nodes()
    matched = [n for n in all_nodes if n == if_node]
    assert len(matched) == 1
    assert isinstance(matched[0], ConditionalNode)


# =============================================================================
# Debug output
# =============================================================================


def test_debug_dot_print_creates_file(sample_graphdef, dot_file):
    """debug_dot_print writes a DOT file."""
    _skip_if_no_mempool()
    sample_graphdef.alloc(ALLOC_SIZE)
    sample_graphdef.debug_dot_print(str(dot_file))
    assert dot_file.exists()
    content = dot_file.read_text()
    assert "digraph" in content


def test_debug_dot_print_with_options(sample_graphdef, dot_file):
    """debug_dot_print accepts GraphDebugPrintOptions."""
    _skip_if_no_mempool()
    sample_graphdef.alloc(ALLOC_SIZE)
    options = GraphDebugPrintOptions(verbose=True, handles=True)
    sample_graphdef.debug_dot_print(str(dot_file), options)
    assert dot_file.exists()


def test_debug_dot_print_invalid_options(sample_graphdef, dot_file):
    """debug_dot_print rejects invalid options type."""
    _skip_if_no_mempool()
    sample_graphdef.alloc(ALLOC_SIZE)
    with pytest.raises(TypeError, match="options must be a GraphDebugPrintOptions"):
        sample_graphdef.debug_dot_print(str(dot_file), "invalid")
