# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Tests for explicit CUDA graph construction (GraphDef and Node)."""

import itertools
import tempfile
from pathlib import Path

import pytest
from helpers.graph_kernels import compile_common_kernels

from cuda.core import Device, LaunchConfig
from cuda.core._graph import GraphDebugPrintOptions
from cuda.core._graph._graphdef import GraphAllocOptions, GraphDef, Node

ALLOC_SIZE = 1024


# =============================================================================
# Fixtures - Sample objects
# =============================================================================


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
    """A root Node (virtual, NULL handle)."""
    return sample_graphdef.root


@pytest.fixture
def sample_root_node_alt(sample_graphdef_alt):
    """An alternate root Node from different graph."""
    return sample_graphdef_alt.root


@pytest.fixture
def sample_empty_node(sample_graphdef):
    """An empty Node (join node)."""
    return sample_graphdef.root.join()


@pytest.fixture
def sample_empty_node_alt(sample_graphdef):
    """An alternate empty Node from same graph."""
    return sample_graphdef.root.join()


@pytest.fixture
def sample_alloc_node(sample_graphdef):
    """An allocation Node."""
    return sample_graphdef.root.alloc(ALLOC_SIZE)


@pytest.fixture
def sample_alloc_node_alt(sample_graphdef):
    """An alternate allocation Node from same graph."""
    return sample_graphdef.root.alloc(ALLOC_SIZE)


@pytest.fixture
def sample_kernel_node(sample_graphdef, init_cuda):
    """A kernel launch Node."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)
    return sample_graphdef.root.launch(config, kernel)


@pytest.fixture
def dot_file():
    """Temporary DOT file path, cleaned up after test."""
    path = Path(tempfile.mktemp(suffix=".dot"))
    yield path
    path.unlink(missing_ok=True)


# =============================================================================
# Type groupings
# =============================================================================

# All types that support __hash__
HASH_TYPES = [
    "sample_graphdef",
    "sample_root_node",
    "sample_empty_node",
    "sample_alloc_node",
]

# All types that support __eq__
EQ_TYPES = [
    "sample_graphdef",
    "sample_root_node",
    "sample_empty_node",
    "sample_alloc_node",
]

# All types (for repr testing)
ALL_TYPES = [
    "sample_graphdef",
    "sample_root_node",
    "sample_empty_node",
    "sample_alloc_node",
    "sample_kernel_node",
]

# Pairs of distinct objects for inequality testing (a != b)
DISTINCT_PAIRS = [
    ("sample_graphdef", "sample_graphdef_alt"),
    ("sample_root_node", "sample_root_node_alt"),
    ("sample_empty_node", "sample_empty_node_alt"),
    ("sample_alloc_node", "sample_alloc_node_alt"),
]

# Repr patterns
REPR_PATTERNS = [
    ("sample_graphdef", r"<GraphDef handle=0x[0-9a-f]+>"),
    ("sample_root_node", r"<Node root>"),
    ("sample_empty_node", r"<Node handle=0x[0-9a-f]+>"),
    ("sample_alloc_node", r"<Node handle=0x[0-9a-f]+ dptr=0x[0-9a-f]+>"),
    ("sample_kernel_node", r"<Node handle=0x[0-9a-f]+>"),
]


# =============================================================================
# Hash tests
# =============================================================================


@pytest.mark.parametrize("fixture_name", HASH_TYPES)
def test_hash_consistent(fixture_name, request):
    """Hash is consistent across multiple calls."""
    obj = request.getfixturevalue(fixture_name)
    assert hash(obj) == hash(obj)


@pytest.mark.parametrize("a_name,b_name", DISTINCT_PAIRS)
def test_hash_distinct(a_name, b_name, request):
    """Distinct objects have different hashes."""
    obj_a = request.getfixturevalue(a_name)
    obj_b = request.getfixturevalue(b_name)
    assert hash(obj_a) != hash(obj_b)


# =============================================================================
# Equality tests (identity-based)
# =============================================================================


@pytest.mark.parametrize("fixture_name", EQ_TYPES)
def test_equals_self(fixture_name, request):
    """Object equals itself."""
    obj = request.getfixturevalue(fixture_name)
    assert obj == obj


@pytest.mark.parametrize("fixture_name", EQ_TYPES)
def test_not_equal_to_other_types(fixture_name, request):
    """Object not equal to unrelated types."""
    obj = request.getfixturevalue(fixture_name)
    assert obj.__eq__("string") is NotImplemented
    assert obj.__eq__(42) is NotImplemented
    assert obj.__eq__(None) is NotImplemented


@pytest.mark.parametrize("a_name,b_name", DISTINCT_PAIRS)
def test_distinct_objects_not_equal(a_name, b_name, request):
    """Distinct objects of same type are not equal."""
    obj_a = request.getfixturevalue(a_name)
    obj_b = request.getfixturevalue(b_name)
    assert obj_a is not obj_b
    assert obj_a != obj_b


@pytest.mark.parametrize("a_name,b_name", list(itertools.combinations(EQ_TYPES, 2)))
def test_cross_type_equality_by_identity(a_name, b_name, request):
    """Cross-type equality: equal iff same object identity."""
    obj_a = request.getfixturevalue(a_name)
    obj_b = request.getfixturevalue(b_name)
    if obj_a is obj_b:
        assert obj_a == obj_b
    else:
        assert obj_a != obj_b


# =============================================================================
# Collection usage tests
# =============================================================================


@pytest.mark.parametrize("fixture_name", HASH_TYPES)
def test_usable_in_set(fixture_name, request):
    """Object can be added to a set."""
    obj = request.getfixturevalue(fixture_name)
    s = {obj}
    assert obj in s


@pytest.mark.parametrize("fixture_name", HASH_TYPES)
def test_usable_as_dict_key(fixture_name, request):
    """Object can be used as dictionary key."""
    obj = request.getfixturevalue(fixture_name)
    d = {obj: "value"}
    assert d[obj] == "value"


# =============================================================================
# Repr tests
# =============================================================================


@pytest.mark.parametrize("fixture_name,pattern", REPR_PATTERNS)
def test_repr_format(fixture_name, pattern, request):
    """repr() matches expected pattern."""
    import re

    obj = request.getfixturevalue(fixture_name)
    assert re.fullmatch(pattern, repr(obj))


# =============================================================================
# GraphDef-specific tests
# =============================================================================


def test_graphdef_handle_valid(sample_graphdef):
    """GraphDef has a valid non-null handle."""
    assert sample_graphdef.handle is not None
    assert int(sample_graphdef.handle) != 0


def test_graphdef_root_returns_node(sample_graphdef):
    """GraphDef.root returns a Node instance."""
    assert isinstance(sample_graphdef.root, Node)


def test_graphdef_root_is_virtual(sample_graphdef):
    """Root node is virtual (no pred/succ)."""
    root = sample_graphdef.root
    assert root.pred == ()
    assert root.succ == ()


# =============================================================================
# Node property tests
# =============================================================================


def test_node_graph_property(sample_graphdef):
    """Node.graph returns the parent GraphDef."""
    node = sample_graphdef.root.join()
    assert node.graph == sample_graphdef


def test_node_dptr_zero_for_non_alloc(sample_empty_node):
    """Non-alloc nodes have dptr=0."""
    assert sample_empty_node.dptr == 0


def test_node_dptr_nonzero_for_alloc(sample_alloc_node):
    """Alloc nodes have non-zero dptr."""
    assert sample_alloc_node.dptr != 0


# =============================================================================
# Graph building: join
# =============================================================================


def test_join_from_root(sample_graphdef):
    """Join from root creates entry node with no predecessors."""
    node = sample_graphdef.root.join()
    assert isinstance(node, Node)
    assert len(node.pred) == 0


def test_join_single_dependency(sample_graphdef):
    """Join from a node creates dependency."""
    n1 = sample_graphdef.root.join()
    n2 = n1.join()
    assert n1 in n2.pred
    assert len(n2.pred) == 1


@pytest.mark.parametrize("num_deps", [2, 3, 5])
def test_join_multiple_dependencies(sample_graphdef, num_deps):
    """Join N nodes creates node depending on all."""
    nodes = [sample_graphdef.root.join() for _ in range(num_deps)]
    joined = nodes[0].join(*nodes[1:])
    assert set(joined.pred) == set(nodes)


# =============================================================================
# Graph building: alloc/free
# =============================================================================


def test_alloc_returns_valid_dptr(sample_graphdef):
    """Alloc returns node with valid device pointer."""
    node = sample_graphdef.root.alloc(ALLOC_SIZE)
    assert node.dptr != 0


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
    assert free.dptr == 0


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
# Allocation options
# =============================================================================


@pytest.mark.parametrize("memory_type", ["device", "managed"])
def test_alloc_memory_type(sample_graphdef, memory_type):
    """Allocation succeeds for supported memory types."""
    options = GraphAllocOptions(memory_type=memory_type)
    node = sample_graphdef.root.alloc(ALLOC_SIZE, options)
    assert node.dptr != 0


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
    """Allocation with peer access list succeeds."""
    d0, d1 = mempool_device_x2
    g = GraphDef()
    options = GraphAllocOptions(device=d0.device_id, peer_access=[d1.device_id])
    node = g.root.alloc(ALLOC_SIZE, options)
    assert node.dptr != 0


# =============================================================================
# Graph traversal: nodes, edges, pred, succ
# =============================================================================


def test_empty_graph_has_no_nodes(sample_graphdef):
    """Empty graph returns no nodes."""
    assert sample_graphdef.nodes() == ()


def test_empty_graph_has_no_edges(sample_graphdef):
    """Empty graph returns no edges."""
    assert sample_graphdef.edges() == ()


def test_nodes_returns_all_nodes(sample_graphdef):
    """nodes() returns all added nodes."""
    n1 = sample_graphdef.root.join()
    n2 = sample_graphdef.root.join()
    n3 = n1.join(n2)
    nodes = sample_graphdef.nodes()
    assert len(nodes) == 3
    assert set(nodes) == {n1, n2, n3}


def test_edges_returns_dependency_pairs(sample_graphdef):
    """edges() returns (from, to) pairs for all dependencies."""
    n1 = sample_graphdef.root.join()
    n2 = n1.join()
    edges = sample_graphdef.edges()
    assert (n1, n2) in edges


def test_edges_multiple(sample_graphdef):
    """edges() with fan-in topology."""
    n1 = sample_graphdef.root.join()
    n2 = sample_graphdef.root.join()
    n3 = n1.join(n2)
    edges = sample_graphdef.edges()
    assert len(edges) == 2
    assert (n1, n3) in edges
    assert (n2, n3) in edges


@pytest.mark.parametrize("direction", ["pred", "succ"])
def test_traversal_single(sample_graphdef, direction):
    """Single predecessor/successor relationship."""
    n1 = sample_graphdef.root.join()
    n2 = n1.join()
    if direction == "pred":
        assert n1 in n2.pred
        assert len(n2.pred) == 1
    else:
        assert n2 in n1.succ
        assert len(n1.succ) == 1


@pytest.mark.parametrize("direction", ["pred", "succ"])
def test_traversal_multiple(sample_graphdef, direction):
    """Multiple predecessors/successors."""
    if direction == "pred":
        n1 = sample_graphdef.root.join()
        n2 = sample_graphdef.root.join()
        n3 = n1.join(n2)
        assert set(n3.pred) == {n1, n2}
    else:
        n1 = sample_graphdef.root.join()
        n2 = n1.join()
        n3 = n1.join()
        assert set(n1.succ) == {n2, n3}


# =============================================================================
# Kernel launch
# =============================================================================


def test_launch_creates_node(sample_graphdef, init_cuda):
    """launch() creates a kernel node."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)
    node = sample_graphdef.root.launch(config, kernel)
    assert isinstance(node, Node)
    assert node.dptr == 0


def test_launch_chain_dependencies(sample_graphdef, init_cuda):
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
# Graph instantiation and execution
# =============================================================================


def test_instantiate_empty_graph(sample_graphdef):
    """Empty graph can be instantiated."""
    graph = sample_graphdef.instantiate()
    assert graph is not None


def test_instantiate_with_nodes(sample_graphdef):
    """Graph with nodes can be instantiated."""
    sample_graphdef.root.join()
    sample_graphdef.root.join()
    graph = sample_graphdef.instantiate()
    assert graph is not None


def test_instantiate_and_execute_kernel(sample_graphdef, init_cuda):
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


# =============================================================================
# Debug output
# =============================================================================


def test_debug_dot_print_creates_file(sample_graphdef, dot_file):
    """debug_dot_print writes a DOT file."""
    sample_graphdef.root.join()
    sample_graphdef.debug_dot_print(str(dot_file))
    assert dot_file.exists()
    content = dot_file.read_text()
    assert "digraph" in content


def test_debug_dot_print_with_options(sample_graphdef, dot_file):
    """debug_dot_print accepts GraphDebugPrintOptions."""
    sample_graphdef.root.join()
    options = GraphDebugPrintOptions(verbose=True, handles=True)
    sample_graphdef.debug_dot_print(str(dot_file), options)
    assert dot_file.exists()


def test_debug_dot_print_invalid_options(sample_graphdef, dot_file):
    """debug_dot_print rejects invalid options type."""
    sample_graphdef.root.join()
    with pytest.raises(TypeError, match="options must be a GraphDebugPrintOptions"):
        sample_graphdef.debug_dot_print(str(dot_file), "invalid")
