# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Tests for resource lifetime management in explicit CUDA graphs.

These tests verify that the RAII mechanism in GraphHandle correctly
prevents dangling references when parent Python objects are deleted
while child/body graph references remain alive.
"""

import gc

import pytest
from helpers.graph_kernels import compile_common_kernels
from helpers.misc import try_create_condition

from cuda.core import LaunchConfig
from cuda.core._graph._graphdef import (
    ChildGraphNode,
    ConditionalNode,
    GraphDef,
)


# =============================================================================
# Conditional body graph lifetime
# =============================================================================


def _make_if(g, cond):
    node = g.if_cond(cond)
    return [node.then]


def _make_if_else(g, cond):
    node = g.if_else(cond)
    return [node.then, node.else_]


def _make_while(g, cond):
    node = g.while_loop(cond)
    return [node.body]


def _make_switch(g, cond):
    node = g.switch(cond, 4)
    return list(node.branches)


_COND_BUILDERS = [
    pytest.param(_make_if, 1, id="if"),
    pytest.param(_make_if_else, 2, id="if_else"),
    pytest.param(_make_while, 1, id="while"),
    pytest.param(_make_switch, 4, id="switch"),
]


@pytest.mark.parametrize("builder, expected_count", _COND_BUILDERS)
def test_branches_survive_parent_deletion(init_cuda, builder, expected_count):
    """All branch graphs remain valid after parent GraphDef is deleted."""
    g = GraphDef()
    condition = try_create_condition(g)
    branches = builder(g, condition)
    assert len(branches) == expected_count

    del g, condition
    gc.collect()

    for branch in branches:
        assert branch.nodes() == ()


@pytest.mark.parametrize("builder, expected_count", _COND_BUILDERS)
def test_branches_usable_after_parent_deletion(init_cuda, builder, expected_count):
    """Nodes can be added to branch graphs after parent GraphDef is deleted."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)

    g = GraphDef()
    condition = try_create_condition(g)
    branches = builder(g, condition)

    del g, condition
    gc.collect()

    for branch in branches:
        branch.launch(config, kernel)
        assert len(branch.nodes()) == 1


def test_reconstructed_body_survives_parent_deletion(init_cuda):
    """Body graph obtained via nodes() reconstruction survives parent deletion."""
    g = GraphDef()
    condition = try_create_condition(g)
    g.while_loop(condition)

    all_nodes = g.nodes()
    cond_nodes = [n for n in all_nodes if isinstance(n, ConditionalNode)]
    assert len(cond_nodes) == 1

    branches = cond_nodes[0].branches
    if not branches:
        pytest.skip("Body reconstruction requires CUDA 13.2+")
    body = branches[0]

    del g, condition, all_nodes, cond_nodes, branches
    gc.collect()

    assert body.nodes() == ()


# =============================================================================
# Child graph (embed) lifetime
# =============================================================================


def test_child_graph_survives_parent_deletion(init_cuda):
    """Embedded child graph remains valid after parent GraphDef is deleted."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)

    child_def = GraphDef()
    child_def.launch(config, kernel)
    child_def.launch(config, kernel)

    g = GraphDef()
    node = g.embed(child_def)
    child_ref = node.child_graph

    del g, node, child_def
    gc.collect()

    assert len(child_ref.nodes()) == 2


def test_nested_child_graph_lifetime(init_cuda):
    """Grandchild graph keeps entire ancestor chain alive."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)

    inner = GraphDef()
    inner.launch(config, kernel)

    middle = GraphDef()
    middle.embed(inner)

    outer = GraphDef()
    outer_node = outer.embed(middle)

    middle_ref = outer_node.child_graph
    middle_nodes = middle_ref.nodes()
    child_node = [n for n in middle_nodes if isinstance(n, ChildGraphNode)][0]
    grandchild = child_node.child_graph

    del outer, outer_node, middle, inner, middle_ref, middle_nodes, child_node
    gc.collect()

    assert len(grandchild.nodes()) == 1
