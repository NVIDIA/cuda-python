# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Tests for mutating a graph definition (edge changes, node removal)."""

import numpy as np
import pytest
from helpers.collection_interface_testers import assert_mutable_set_interface
from helpers.graph_kernels import compile_parallel_kernels
from helpers.marks import requires_module

from cuda.core import Device, LaunchConfig, LegacyPinnedMemoryResource
from cuda.core._graph._graph_def import GraphDef, KernelNode, MemsetNode
from cuda.core._utils.cuda_utils import CUDAError


class YRig:
    """Test rigging for graph mutation tests. Constructs a Y-shaped graph with
    two parallel arms joined by a combine node. Modifying the sequence of
    operations along either arm changes the output.

    Topology::

        a0 -- a1 -- a2
                      \
                       j -- r
                      /
              b0 -- b1

    Each a/b node applies ``affine(ptr, m, b)`` to its arm's int accumulator.
    Node r computes result ``combine(R, A, B) = (A << 16) | (B & 0xFFFF)``,
    encoding both arms' results into a single int. j is a joining (empty) node
    preceeding r.

    The affine operation a * m + b is noncommutative, so we can be sure the
    graph has exactly the topology we expect by checking the final value.
    """

    def __init__(self):
        self.A_OPS = [(2, 1), (3, 2), (5, 3)]
        self.B_OPS = [(2, 7), (3, 1)]

        mod = compile_parallel_kernels()
        self.affine = mod.get_kernel("affine")
        self.combine = mod.get_kernel("combine")
        self.config = LaunchConfig(grid=1, block=1)

        self._mr = LegacyPinnedMemoryResource()
        self._buf = self._mr.allocate(3 * 4)
        self._arr = np.from_dlpack(self._buf).view(np.int32)

        self.ptr_a = self._arr[0:].ctypes.data
        self.ptr_b = self._arr[1:].ctypes.data
        self.ptr_r = self._arr[2:].ctypes.data

        self.graph_def = GraphDef()
        self.stream = None

        # Arm A
        self.a = []
        prev = self.graph_def
        for m, b in self.A_OPS:
            prev = prev.launch(self.config, self.affine, self.ptr_a, m, b)
            self.a.append(prev)

        # Arm B
        self.b = []
        prev = self.graph_def
        for m, b in self.B_OPS:
            prev = prev.launch(self.config, self.affine, self.ptr_b, m, b)
            self.b.append(prev)

        # Join and combine
        self.j = self.graph_def.join(self.a[-1], self.b[-1])
        self.r = self.j.launch(self.config, self.combine, self.ptr_r, self.ptr_a, self.ptr_b)

    def run(self):
        if self.stream is None:
            self.stream = Device().create_stream()
        graph = self.graph_def.instantiate()
        self.reset()
        graph.launch(self.stream)
        self.stream.sync()

    def reset(self):
        self._arr[:] = 0

    @property
    def A_out(self):
        return int(self._arr[0])

    @property
    def B_out(self):
        return int(self._arr[1])

    @property
    def R_out(self):
        return int(self._arr[2])

    @property
    def output(self):
        return self.A_out, self.B_out, self.R_out

    @property
    def expected_output(self):
        """Expected (A, B, R) after one run from zero."""

        def apply_affine(val, ops):
            for m, b in ops:
                val = val * m + b
            return val

        a = apply_affine(0, self.A_OPS)
        b = apply_affine(0, self.B_OPS)
        r = (a << 16) | (b & 0xFFFF)
        return (a, b, r)

    @property
    def edges(self):
        return self.graph_def.edges()

    @property
    def initial_edges(self):
        return (
            set(zip(self.a, self.a[1:]))
            | set(zip(self.b, self.b[1:]))
            | {(self.a[-1], self.j), (self.b[-1], self.j), (self.j, self.r)}
        )

    @property
    def nodes(self):
        return self.graph_def.nodes()

    @property
    def initial_nodes(self):
        return set(self.a + self.b + [self.j, self.r])

    def close(self):
        self._buf.close()


@requires_module(np, "2.1")
class TestMutateYRig:
    """Tests that mutate the Y-shaped graph built by YRig."""

    def test_baseline(self, init_cuda):
        """Unmodified graph produces the expected results."""
        rig = YRig()
        rig.run()
        assert rig.output == rig.expected_output
        assert rig.edges == rig.initial_edges
        assert rig.nodes == rig.initial_nodes
        rig.close()

    def test_destroy_a1(self, init_cuda):
        """Destroy a1 (creates a race on arm a). Arm b yields the expected
        value, and the final step is correctly ordered after b completes."""
        rig = YRig()
        rig.a[1].destroy()
        rig.run()
        _, b_exp, _ = rig.expected_output
        assert rig.B_out == b_exp
        assert (rig.R_out & 0xFFFF) == b_exp
        a0, a1, a2 = rig.a
        assert rig.edges == rig.initial_edges - {(a0, a1), (a1, a2)}
        assert rig.nodes == rig.initial_nodes - {a1}
        rig.close()

    def test_destroy_a2(self, init_cuda):
        """Destroy a2, connect a1--r"""
        rig = YRig()
        rig.a[2].destroy()
        rig.a[1].succ.add(rig.r)
        rig.A_OPS.pop()
        rig.run()
        assert rig.output == rig.expected_output
        a0, a1, a2, j, r = rig.a + [rig.j, rig.r]
        assert rig.edges == (rig.initial_edges - {(a1, a2), (a2, j)}) | {(a1, r)}
        assert rig.nodes == rig.initial_nodes - {a2}
        rig.close()

    def test_destroy_joint(self, init_cuda):
        """Remove the joining node and instead add edges directly to r."""
        rig = YRig()
        _, _, a2, _, b1, j, r = rig.a + rig.b + [rig.j, rig.r]
        j.destroy()
        r.pred = {a2, b1}
        rig.run()
        assert rig.output == rig.expected_output
        assert rig.edges == (rig.initial_edges - {(a2, j), (b1, j), (j, r)}) | {(a2, r), (b1, r)}
        assert rig.nodes == rig.initial_nodes - {j}
        rig.close()

    def test_insert_b(self, init_cuda):
        """Insert a node into arm b."""
        rig = YRig()
        coeffs = 5, 3
        b_new = rig.graph_def.launch(rig.config, rig.affine, rig.ptr_b, *coeffs)
        b0, b1 = rig.b
        b0.succ.discard(b1)
        b0.succ.add(b_new)
        b_new.succ.add(b1)
        rig.B_OPS.insert(1, coeffs)
        rig.run()
        assert rig.output == rig.expected_output
        assert rig.edges == (rig.initial_edges - {(b0, b1)}) | {(b0, b_new), (b_new, b1)}
        assert rig.nodes == rig.initial_nodes | {b_new}
        rig.close()


def test_adjacency_set_interface(init_cuda):
    """Exercise every MutableSet method on AdjacencySetProxy."""
    g = GraphDef()
    hub = g.join()
    items = [g.join() for _ in range(5)]
    assert_mutable_set_interface(hub.succ, items)


def test_adjacency_set_pred_direction(init_cuda):
    """Verify that pred works symmetrically with succ."""
    g = GraphDef()
    target = g.join()
    x, y, z = (g.join() for _ in range(3))

    pred = target.pred
    assert pred == set()

    pred.add(x)
    pred.add(y)
    assert pred == {x, y}

    # Verify the edge is visible from the other direction
    assert target in x.succ
    assert target in y.succ
    assert target not in z.succ

    pred.discard(x)
    assert pred == {y}
    assert target not in x.succ


def test_adjacency_set_property_setter(init_cuda):
    """Verify that assigning to node.pred or node.succ replaces all edges."""
    g = GraphDef()
    hub = g.join()
    a, b, c = (g.join() for _ in range(3))

    hub.succ = {a, b}
    assert hub.succ == {a, b}

    hub.succ = {c}
    assert hub.succ == {c}
    assert a not in hub.succ

    hub.succ = set()
    assert hub.succ == set()

    hub.pred = {a, b}
    assert hub.pred == {a, b}

    hub.pred = set()
    assert hub.pred == set()

    hub.pred = set()
    assert hub.pred == set()


def test_destroyed_node(init_cuda):
    """Test that destroy() invalidates a node."""
    mr = LegacyPinnedMemoryResource()
    buf = mr.allocate(4)
    arr = np.from_dlpack(buf).view(np.int32)
    arr[:] = 0
    ptr = arr[0:].ctypes.data

    g = GraphDef()
    a = g.memset(ptr, 0, 4)
    b = a.memset(ptr, 42, 4)

    assert a.is_valid
    assert b.is_valid
    assert b in g.nodes()
    assert (a, b) in g.edges()

    b.destroy()

    assert not b.is_valid
    assert b not in g.nodes()
    assert (a, b) not in g.edges()

    # Python object is invalid but using it does not crash.
    assert isinstance(b, MemsetNode)
    assert b.type is None
    assert b.pred == set()
    assert b.succ == set()
    assert b.handle is None
    assert b.dptr == ptr  # tolerable
    assert b.value == 42  # tolerable
    assert b.width == 4  # tolerable

    # Adding an edge to a destroyed node fails.
    with pytest.raises(CUDAError):
        a.succ.add(b)

    # Repeated destroy succeeds quietly.
    b.destroy()
    assert not b.is_valid


def test_add_wrong_type(init_cuda):
    """Adding a non-GraphNode raises TypeError."""
    g = GraphDef()
    node = g.join()
    with pytest.raises(TypeError, match="expected GraphNode"):
        node.succ.add("not a node")
    with pytest.raises(TypeError, match="expected GraphNode"):
        node.succ.add(42)


def test_cross_graph_edge(init_cuda):
    """Adding an edge to a node from a different graph raises CUDAError."""
    g1 = GraphDef()
    g2 = GraphDef()
    a = g1.join()
    b = g2.join()
    with pytest.raises(CUDAError):
        a.succ.add(b)


def test_self_edge(init_cuda):
    """Adding a self-edge raises CUDAError."""
    g = GraphDef()
    node = g.join()
    with pytest.raises(CUDAError):
        node.succ.add(node)


@requires_module(np, "2.1")
def test_convert_linear_to_fan_in(init_cuda):
    """Chain four computations sequentially, then rewire so all pairs run in
    parallel feeding into a reduce node.

    Initial topology (sequential)::

        memset0 -- launch0 -- memset1 -- launch1 -- memset2 -- launch2 -- memset3 -- launch3

    After rewiring (parallel)::

        memset0 -- launch0 --\
        memset1 -- launch1 ---+-- reduce
        memset2 -- launch2 --/
        memset3 -- launch3 -/
    """
    mod = compile_parallel_kernels()
    affine = mod.get_kernel("affine")
    reduce_kern = mod.get_kernel("reduce")
    config = LaunchConfig(grid=1, block=1)

    mr = LegacyPinnedMemoryResource()
    buf = mr.allocate(5 * 4)
    arr = np.from_dlpack(buf).view(np.int32)
    arr[:] = 0

    values = np.array([10, 20, 30, 40], dtype=np.int32)
    ptrs = [arr[i:].ctypes.data for i in range(5)]

    # Create the initial graph.
    g = GraphDef()
    prev = g
    for i, val in enumerate(values):
        prev = prev.memset(ptrs[i], val, 1)
        prev = prev.launch(config, affine, ptrs[i], 2, 1)
    reduce_node = g.launch(config, reduce_kern, ptrs[4], ptrs[0], 4)

    # Rewire:
    #   - drop preds from memsets
    #   - connect kernel launches to the reduction
    assert len(g.edges()) == 7

    for node in g.nodes():
        if isinstance(node, MemsetNode):
            node.pred.clear()
        elif isinstance(node, KernelNode) and node is not reduce_node:
            node.succ.add(reduce_node)

    assert len(g.edges()) == 8

    stream = Device().create_stream()
    graph = g.instantiate()
    graph.launch(stream)
    stream.sync()
    assert arr[4] == sum(2 * values + 1)

    buf.close()
