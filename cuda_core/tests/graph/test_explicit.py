# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Tests for explicit CUDA graph construction (GraphDef and Node)."""

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest
from helpers.graph_kernels import compile_common_kernels

from cuda.core import Device, LaunchConfig
from cuda.core._graph import GraphDebugPrintOptions
from cuda.core._graph._graphdef import (
    AllocNode,
    EmptyNode,
    FreeNode,
    GraphAllocOptions,
    GraphDef,
    KernelNode,
    MemsetNode,
    Node,
)

ALLOC_SIZE = 1024


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
    a = g.root.alloc(ALLOC_SIZE)
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
    a = g.root.alloc(ALLOC_SIZE)
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
    a = g.root.alloc(ALLOC_SIZE)
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
    a = g.root.alloc(ALLOC_SIZE)
    b = g.root.alloc(ALLOC_SIZE)
    c = g.root.alloc(ALLOC_SIZE)
    d = a.join(b, c)
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
    a = g.root.alloc(ALLOC_SIZE)
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
    a = g.root.alloc(ALLOC_SIZE)
    b = g.root.alloc(ALLOC_SIZE)
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
    return request.param()


@pytest.fixture(params=_NONEMPTY_BUILDERS)
def nonempty_graph_spec(request, init_cuda):
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
    builder: Callable[[GraphDef], tuple[Node, dict]]


def _build_empty_node(g):
    a = g.root.alloc(ALLOC_SIZE)
    b = g.root.alloc(ALLOC_SIZE)
    return a.join(b), {}


def _build_kernel_node(g):
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=(2, 3, 1), block=(32, 4, 1), shmem_size=128)
    entry = g.root.alloc(ALLOC_SIZE)
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
    entry = g.root.alloc(ALLOC_SIZE)
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
    entry = g.root.alloc(ALLOC_SIZE)
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
    alloc = g.root.alloc(ALLOC_SIZE)
    node = alloc.free(alloc.dptr)
    return node, {
        "dptr": alloc.dptr,
    }


def _build_memset_node(g):
    alloc = g.root.alloc(ALLOC_SIZE)
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
    alloc = g.root.alloc(ALLOC_SIZE)
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
    alloc = g.root.alloc(ALLOC_SIZE)
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
    alloc = g.root.alloc(ALLOC_SIZE)
    node = alloc.memset(alloc.dptr, 0xFF, cols, height=rows, pitch=cols)
    return node, {
        "dptr": alloc.dptr,
        "value": 0xFF,
        "element_size": 1,
        "width": cols,
        "height": rows,
        "pitch": cols,
    }


_NODE_SPECS = [
    pytest.param(NodeSpec("empty", EmptyNode, "CU_GRAPH_NODE_TYPE_EMPTY", _build_empty_node), id="empty"),
    pytest.param(NodeSpec("kernel", KernelNode, "CU_GRAPH_NODE_TYPE_KERNEL", _build_kernel_node), id="kernel"),
    pytest.param(NodeSpec("alloc", AllocNode, "CU_GRAPH_NODE_TYPE_MEM_ALLOC", _build_alloc_node), id="alloc"),
    pytest.param(
        NodeSpec("alloc_managed", AllocNode, "CU_GRAPH_NODE_TYPE_MEM_ALLOC", _build_alloc_managed_node),
        id="alloc_managed",
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
]


@pytest.fixture(params=_NODE_SPECS)
def node_spec(request, init_cuda):
    spec = request.param
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
    """Node is an instance of the expected subclass."""
    spec, g, node, _ = node_spec
    assert isinstance(node, spec.expected_class)
    assert isinstance(node, Node)


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
    assert isinstance(matched[0], spec.expected_class)


def test_node_type_preserved_by_pred_succ(node_spec):
    """Node type is preserved when retrieved via pred/succ traversal."""
    spec, g, node, _ = node_spec
    for predecessor in node.pred:
        matched = [s for s in predecessor.succ if s == node]
        assert len(matched) == 1
        assert isinstance(matched[0], spec.expected_class)


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


def test_graphdef_root_returns_node(sample_graphdef):
    """GraphDef.root returns a Node instance."""
    assert isinstance(sample_graphdef.root, Node)


def test_graphdef_root_is_virtual(sample_graphdef):
    """Root node is virtual (no pred/succ, type is None)."""
    root = sample_graphdef.root
    assert root.pred == ()
    assert root.succ == ()
    assert root.type is None


# =============================================================================
# Alloc/free API
# =============================================================================


def test_alloc_zero_size_fails(sample_graphdef):
    """Alloc with zero size raises error (CUDA limitation)."""
    from cuda.core._utils.cuda_utils import CUDAError

    with pytest.raises(CUDAError):
        sample_graphdef.root.alloc(0)


def test_free_creates_dependency(sample_graphdef):
    """Free node depends on its predecessor."""
    alloc = sample_graphdef.root.alloc(ALLOC_SIZE)
    free = alloc.free(alloc.dptr)
    assert alloc in free.pred


def test_alloc_free_chain(sample_graphdef):
    """Alloc and free can be chained."""
    a1 = sample_graphdef.root.alloc(ALLOC_SIZE)
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
        sample_graphdef.root.alloc(ALLOC_SIZE, options)


@pytest.mark.parametrize(
    "device_spec",
    [
        pytest.param(lambda d: d.device_id, id="device_id"),
        pytest.param(lambda d: d, id="Device_object"),
    ],
)
def test_alloc_device_option(sample_graphdef, device_spec):
    """Device can be specified as int or Device object."""
    device = Device()
    options = GraphAllocOptions(device=device_spec(device))
    node = sample_graphdef.root.alloc(ALLOC_SIZE, options)
    assert node.dptr != 0


def test_alloc_peer_access(mempool_device_x2):
    """AllocNode.peer_access reflects requested peers."""
    d0, d1 = mempool_device_x2
    g = GraphDef()
    options = GraphAllocOptions(device=d0.device_id, peer_access=[d1.device_id])
    node = g.root.alloc(ALLOC_SIZE, options)
    assert d1.device_id in node.peer_access


# =============================================================================
# Join API
# =============================================================================


@pytest.mark.parametrize("num_branches", [2, 3, 5])
def test_join_merges_branches(sample_graphdef, num_branches):
    """join() with multiple branches creates correct dependencies."""
    branches = [sample_graphdef.root.alloc(ALLOC_SIZE) for _ in range(num_branches)]
    joined = branches[0].join(*branches[1:])
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
    node = sample_graphdef.root.launch(config, kernel)
    assert isinstance(node, KernelNode)


def test_launch_chain_dependencies(sample_graphdef):
    """Chained launches create correct dependencies."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)
    n1 = sample_graphdef.root.launch(config, kernel)
    n2 = n1.launch(config, kernel)
    n3 = n2.launch(config, kernel)
    assert n1 in n2.pred
    assert n2 in n3.pred
    assert n1 not in n3.pred


# =============================================================================
# Instantiation and execution
# =============================================================================


def test_instantiate_empty_graph(sample_graphdef):
    """Empty graph can be instantiated."""
    graph = sample_graphdef.instantiate()
    assert graph is not None


def test_instantiate_with_nodes(sample_graphdef):
    """Graph with nodes can be instantiated."""
    sample_graphdef.root.alloc(ALLOC_SIZE)
    sample_graphdef.root.alloc(ALLOC_SIZE)
    graph = sample_graphdef.instantiate()
    assert graph is not None


def test_instantiate_and_execute_kernel(sample_graphdef):
    """Graph with kernel can be instantiated and executed."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)
    sample_graphdef.root.launch(config, kernel)
    graph = sample_graphdef.instantiate()

    stream = Device().create_stream()
    graph.upload(stream)
    graph.launch(stream)
    stream.sync()


def test_instantiate_and_execute_alloc_free(sample_graphdef):
    """Graph with alloc/free can be executed."""
    alloc = sample_graphdef.root.alloc(ALLOC_SIZE)
    alloc.free(alloc.dptr)
    graph = sample_graphdef.instantiate()

    stream = Device().create_stream()
    graph.upload(stream)
    graph.launch(stream)
    stream.sync()


def test_instantiate_and_execute_memset(sample_graphdef):
    """Graph with alloc/memset/free can be executed."""
    alloc = sample_graphdef.root.alloc(ALLOC_SIZE)
    ms = alloc.memset(alloc.dptr, 0xAB, ALLOC_SIZE)
    ms.free(alloc.dptr)
    graph = sample_graphdef.instantiate()

    stream = Device().create_stream()
    graph.upload(stream)
    graph.launch(stream)
    stream.sync()


# =============================================================================
# Debug output
# =============================================================================


def test_debug_dot_print_creates_file(sample_graphdef, dot_file):
    """debug_dot_print writes a DOT file."""
    sample_graphdef.root.alloc(ALLOC_SIZE)
    sample_graphdef.debug_dot_print(str(dot_file))
    assert dot_file.exists()
    content = dot_file.read_text()
    assert "digraph" in content


def test_debug_dot_print_with_options(sample_graphdef, dot_file):
    """debug_dot_print accepts GraphDebugPrintOptions."""
    sample_graphdef.root.alloc(ALLOC_SIZE)
    options = GraphDebugPrintOptions(verbose=True, handles=True)
    sample_graphdef.debug_dot_print(str(dot_file), options)
    assert dot_file.exists()


def test_debug_dot_print_invalid_options(sample_graphdef, dot_file):
    """debug_dot_print rejects invalid options type."""
    sample_graphdef.root.alloc(ALLOC_SIZE)
    with pytest.raises(TypeError, match="options must be a GraphDebugPrintOptions"):
        sample_graphdef.debug_dot_print(str(dot_file), "invalid")
