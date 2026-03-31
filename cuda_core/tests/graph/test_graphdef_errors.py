# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Tests for GraphDef input validation, error handling, and edge cases."""

import ctypes

import pytest
from helpers.graph_kernels import compile_common_kernels
from helpers.misc import try_create_condition

from cuda.core import Device, LaunchConfig
from cuda.core._graph._graphdef import (
    Condition,
    EmptyNode,
    GraphDef,
)
from cuda.core._utils.cuda_utils import CUDAError

SIZEOF_INT = ctypes.sizeof(ctypes.c_int)


def _skip_if_no_mempool():
    if not Device(0).properties.memory_pools_supported:
        pytest.skip("Device does not support mempool operations")


# =============================================================================
# Type validation — wrong types for conditional node methods
# =============================================================================


@pytest.mark.parametrize(
    "method, args",
    [
        pytest.param("if_cond", (42,), id="if_cond_int"),
        pytest.param("if_else", ("not a condition",), id="if_else_str"),
        pytest.param("while_loop", (None,), id="while_loop_none"),
        pytest.param("switch", ([1, 2, 3], 4), id="switch_list"),
    ],
)
def test_conditional_rejects_non_condition(init_cuda, method, args):
    """Conditional node methods reject non-Condition arguments."""
    g = GraphDef()
    with pytest.raises(TypeError, match="Condition"):
        getattr(g, method)(*args)


def test_embed_rejects_non_graphdef(init_cuda):
    """embed() rejects non-GraphDef arguments."""
    g = GraphDef()
    with pytest.raises((TypeError, AttributeError)):
        g.embed("not a graph")


# =============================================================================
# Value validation — invalid parameter values
# =============================================================================


def test_free_null_pointer(init_cuda):
    """free(0) raises a CUDA error."""
    g = GraphDef()
    with pytest.raises(CUDAError):
        g.free(0)


def test_memset_invalid_value_size(init_cuda):
    """memset with 3-byte value (not 1, 2, or 4) raises ValueError."""
    _skip_if_no_mempool()
    g = GraphDef()
    alloc = g.alloc(1024)
    with pytest.raises(ValueError):
        alloc.memset(alloc.dptr, b"\x01\x02\x03", 100)


def test_switch_zero_branches(init_cuda):
    """switch with count=0 raises an error."""
    g = GraphDef()
    condition = try_create_condition(g)
    with pytest.raises(CUDAError):
        g.switch(condition, 0)


# =============================================================================
# Cross-graph misuse
# =============================================================================


def test_condition_from_different_graph(init_cuda):
    """Using a condition created for graph A in graph B raises an error."""
    g1 = GraphDef()
    g2 = GraphDef()
    condition = try_create_condition(g1)
    with pytest.raises(CUDAError):
        g2.if_cond(condition)


# =============================================================================
# Edge cases — valid but unusual usage patterns
# =============================================================================


def test_join_no_extra_nodes(init_cuda):
    """join() from entry with no extra nodes creates a single empty node."""
    g = GraphDef()
    joined = g.join()
    assert isinstance(joined, EmptyNode)
    assert len(g.nodes()) == 1


def test_join_single_predecessor(init_cuda):
    """node.join() with no extra args creates a single-dep empty node."""
    _skip_if_no_mempool()
    g = GraphDef()
    a = g.alloc(1024)
    joined = a.join()
    assert isinstance(joined, EmptyNode)
    assert set(joined.pred) == {a}


def test_multiple_instantiation(init_cuda):
    """Same GraphDef can be instantiated multiple times independently."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    cfg = LaunchConfig(grid=1, block=1)

    g = GraphDef()
    g.launch(cfg, kernel)
    g1 = g.instantiate()
    g2 = g.instantiate()
    assert g1 is not g2


def test_unmatched_alloc_succeeds(init_cuda):
    """Alloc without corresponding free is valid (graph-scoped lifetime)."""
    _skip_if_no_mempool()
    g = GraphDef()
    g.alloc(1024)
    graph = g.instantiate()
    stream = Device().create_stream()
    graph.launch(stream)
    stream.sync()


def test_create_condition_no_default_value(init_cuda):
    """create_condition with no default_value succeeds."""
    g = GraphDef()
    try:
        condition = g.create_condition()
    except CUDAError:
        pytest.skip("Conditional nodes not supported (requires CC >= 9.0)")
    assert isinstance(condition, Condition)


# =============================================================================
# Boundary condition execution — conditional nodes with extreme values
# =============================================================================


def _skip_unless_cc_90():
    if Device(0).compute_capability < (9, 0):
        pytest.skip("Conditional node execution requires CC >= 9.0")


def test_while_loop_zero_iterations(init_cuda):
    """While loop with default_value=0 never executes its body."""
    _skip_unless_cc_90()
    _skip_if_no_mempool()

    mod = compile_common_kernels()
    add_one = mod.get_kernel("add_one")
    cfg = LaunchConfig(grid=1, block=1)

    g = GraphDef()
    condition = g.create_condition(default_value=0)
    alloc = g.alloc(SIZEOF_INT)
    ms = alloc.memset(alloc.dptr, 0, SIZEOF_INT)
    loop = ms.while_loop(condition)
    loop.body.launch(cfg, add_one, alloc.dptr)

    graph = g.instantiate()
    stream = Device().create_stream()
    graph.launch(stream)
    stream.sync()

    result = (ctypes.c_int * 1)()
    from cuda.bindings import driver as drv

    drv.cuMemcpyDtoH(result, alloc.dptr, SIZEOF_INT)
    assert result[0] == 0, "Body should not have executed"


def test_if_cond_false_skips_body(init_cuda):
    """If conditional with default_value=0 does not execute its body."""
    _skip_unless_cc_90()
    _skip_if_no_mempool()

    mod = compile_common_kernels()
    add_one = mod.get_kernel("add_one")
    cfg = LaunchConfig(grid=1, block=1)

    g = GraphDef()
    condition = g.create_condition(default_value=0)
    alloc = g.alloc(SIZEOF_INT)
    ms = alloc.memset(alloc.dptr, 0, SIZEOF_INT)
    if_node = ms.if_cond(condition)
    if_node.then.launch(cfg, add_one, alloc.dptr)

    graph = g.instantiate()
    stream = Device().create_stream()
    graph.launch(stream)
    stream.sync()

    result = (ctypes.c_int * 1)()
    from cuda.bindings import driver as drv

    drv.cuMemcpyDtoH(result, alloc.dptr, SIZEOF_INT)
    assert result[0] == 0, "Body should not have executed"


def test_switch_oob_skips_all_branches(init_cuda):
    """Switch with out-of-range condition value does not execute any branch."""
    _skip_unless_cc_90()
    _skip_if_no_mempool()

    mod = compile_common_kernels()
    add_one = mod.get_kernel("add_one")
    cfg = LaunchConfig(grid=1, block=1)

    g = GraphDef()
    condition = g.create_condition(default_value=99)
    alloc = g.alloc(SIZEOF_INT)
    ms = alloc.memset(alloc.dptr, 0, SIZEOF_INT)
    sw = ms.switch(condition, 3)
    for branch in sw.branches:
        branch.launch(cfg, add_one, alloc.dptr)

    graph = g.instantiate()
    stream = Device().create_stream()
    graph.launch(stream)
    stream.sync()

    result = (ctypes.c_int * 1)()
    from cuda.bindings import driver as drv

    drv.cuMemcpyDtoH(result, alloc.dptr, SIZEOF_INT)
    assert result[0] == 0, "No branch should have executed"
