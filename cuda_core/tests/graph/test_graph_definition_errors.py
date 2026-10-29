# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GraphDefinition input validation, error handling, and edge cases."""

import ctypes

import pytest
from helpers.contexts import current_context_handle, use_context
from helpers.graph_kernels import compile_common_kernels
from helpers.misc import try_create_condition

from cuda.core import ContextOptions, Device, LaunchConfig, SMResourceOptions
from cuda.core._utils.cuda_utils import CUDAError, driver, handle_return
from cuda.core._utils.version import driver_version
from cuda.core.graph import (
    EmptyNode,
    GraphCompleteOptions,
    GraphCondition,
    GraphDefinition,
)

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
        pytest.param("if_then", (42,), id="if_then_int"),
        pytest.param("if_else", ("not a condition",), id="if_else_str"),
        pytest.param("while_loop", (None,), id="while_loop_none"),
        pytest.param("switch", ([1, 2, 3], 4), id="switch_list"),
    ],
)
def test_conditional_rejects_non_condition(init_cuda, method, args):
    """Conditional node methods reject non-GraphCondition arguments."""
    g = GraphDefinition()
    with pytest.raises(TypeError, match="GraphCondition"):
        getattr(g, method)(*args)


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize("none_arg", ["config", "kernel"])
@pytest.mark.parametrize("target", ["GraphDefinition", "GraphNode"])
def test_launch_rejects_none(init_cuda, target, none_arg):
    """launch() on a graph or node rejects None for config and kernel."""
    kernel = compile_common_kernels().get_kernel("add_one")
    g = GraphDefinition()
    if target == "GraphDefinition":
        owner = g
    else:
        owner = g.empty()
    args = {"config": LaunchConfig(grid=1, block=1), "kernel": kernel}
    args[none_arg] = None

    with pytest.raises(TypeError):
        owner.launch(args["config"], args["kernel"])


def test_embed_rejects_non_graphdef(init_cuda):
    """embed() rejects non-GraphDefinition arguments."""
    g = GraphDefinition()
    with pytest.raises((TypeError, AttributeError)):
        g.embed("not a graph")


# =============================================================================
# Value validation — invalid parameter values
# =============================================================================


def test_free_null_pointer(init_cuda):
    """free(0) raises a CUDA error."""
    g = GraphDefinition()
    with pytest.raises(CUDAError):
        g.deallocate(0)


def test_memset_invalid_value_size(init_cuda):
    """memset with 3-byte value (not 1, 2, or 4) raises ValueError."""
    _skip_if_no_mempool()
    g = GraphDefinition()
    alloc = g.allocate(1024)
    with pytest.raises(ValueError):
        alloc.memset(alloc.dptr, b"\x01\x02\x03", 100)


def test_switch_zero_branches(init_cuda):
    """switch with count=0 raises an error."""
    g = GraphDefinition()
    condition = try_create_condition(g)
    with pytest.raises(CUDAError):
        g.switch(condition, 0)


# =============================================================================
# Cross-graph misuse
# =============================================================================


def test_condition_from_different_graph(init_cuda):
    """Using a condition created for graph A in graph B raises an error."""
    g1 = GraphDefinition()
    g2 = GraphDefinition()
    condition = try_create_condition(g1)
    with pytest.raises(CUDAError):
        g2.if_then(condition)


# =============================================================================
# Edge cases — valid but unusual usage patterns
# =============================================================================


def test_empty_node(init_cuda):
    """empty() creates a single entry-point empty node."""
    g = GraphDefinition()
    joined = g.empty()
    assert isinstance(joined, EmptyNode)
    assert len(g.nodes()) == 1


def test_join_single_predecessor(init_cuda):
    """node.join() with no extra args creates a single-dep empty node."""
    _skip_if_no_mempool()
    g = GraphDefinition()
    a = g.allocate(1024)
    joined = a.join()
    assert isinstance(joined, EmptyNode)
    assert set(joined.pred) == {a}


def test_multiple_instantiation(init_cuda):
    """Same GraphDefinition can be instantiated multiple times independently."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    cfg = LaunchConfig(grid=1, block=1)

    g = GraphDefinition()
    g.launch(cfg, kernel)
    g1 = g.instantiate()
    g2 = g.instantiate()
    assert g1 is not g2


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_instantiate_rejects_cycle_and_accepts_repaired_graph(init_cuda):
    """Removing a cycle makes an invalid graph executable again."""
    definition = GraphDefinition()
    called = []
    first = definition.empty()
    last = first.callback(lambda: called.append(True))
    last.succ.add(first)
    stream = init_cuda.create_stream()
    executable = None
    try:
        with pytest.raises(RuntimeError, match="invalid structure, such as cycles"):
            definition.instantiate()
        assert called == []
        last.succ.discard(first)
        executable = definition.instantiate()
        executable.launch(stream)
        stream.sync()
        assert called == [True]
    finally:
        if executable is not None:
            executable.close()
        last.succ.discard(first)
        last.destroy()
        first.destroy()
        stream.close()


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_device_launch_rejects_host_callback(init_cuda):
    """Device-launch graphs reject host callbacks before execution."""
    definition = GraphDefinition()
    called = []
    node = definition.callback(lambda: called.append(True))
    try:
        with pytest.raises(
            RuntimeError, match="device launch failed because the graph contained an unsupported operation"
        ):
            definition.instantiate(GraphCompleteOptions(device_launch=True))
        assert called == []
    finally:
        node.destroy()


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_device_launch_rejects_nodes_from_different_contexts(init_cuda):
    """Device-launch instantiation rejects mixed contexts and accepts a repaired graph."""
    if driver_version() < (12, 4, 0):
        pytest.skip("green contexts require CUDA driver 12.4 or newer")
    if not init_cuda.properties.unified_addressing:
        pytest.skip("device launch requires unified addressing")
    original_context = current_context_handle()
    definition = GraphDefinition()
    green_context = first = foreign = executable = stream = None
    try:
        groups, _ = init_cuda.resources.sm.split(SMResourceOptions(count=None))
        green_context = init_cuda.create_context(ContextOptions(resources=[groups[0]]))
        module = compile_common_kernels()
        kernel = module.get_kernel("empty_kernel")
        config = LaunchConfig(grid=1, block=1)
        first = definition.launch(config, kernel)
        with use_context(init_cuda, green_context):
            foreign = definition.launch(config, kernel)

        # CUkernel handles are context-independent; check each node's resolved context.
        first_params = handle_return(driver.cuGraphKernelNodeGetParams(first.handle))
        foreign_params = handle_return(driver.cuGraphKernelNodeGetParams(foreign.handle))
        assert int(first_params.ctx) == original_context
        assert int(foreign_params.ctx) == int(green_context.handle)
        assert int(foreign_params.ctx) != original_context
        assert current_context_handle() == original_context
        with pytest.raises(RuntimeError, match="nodes belonging to different contexts"):
            definition.instantiate(GraphCompleteOptions(device_launch=True))
        assert current_context_handle() == original_context

        # A single-context graph remains valid after the failed instantiation.
        foreign.destroy()
        foreign = None
        executable = definition.instantiate(GraphCompleteOptions(device_launch=True))
        stream = init_cuda.create_stream()
        executable.launch(stream)
        stream.sync()
        assert current_context_handle() == original_context
    finally:
        if executable is not None:
            executable.close()
        if foreign is not None:
            foreign.destroy()
        if first is not None:
            first.destroy()
        if stream is not None:
            stream.close()
        if green_context is not None:
            green_context.close()


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_instantiate_rejects_unassociated_condition(init_cuda):
    """Instantiation rejects a condition that no conditional node uses."""
    if driver_version() < (12, 8, 0):
        pytest.skip("Unused graph-condition diagnostics require CUDA driver 12.8 or newer")
    definition = GraphDefinition()
    # Deliberately leave this graph-owned condition unassociated with a node.
    definition.create_condition(default_value=1)
    node = definition.empty()
    try:
        with pytest.raises(RuntimeError, match="conditional handles are not associated with conditional builders"):
            definition.instantiate()
    finally:
        node.destroy()


def test_unmatched_alloc_succeeds(init_cuda):
    """Alloc without corresponding free is valid (graph-scoped lifetime)."""
    _skip_if_no_mempool()
    g = GraphDefinition()
    g.allocate(1024)
    graph = g.instantiate()
    stream = Device().create_stream()
    graph.launch(stream)
    stream.sync()


def test_create_condition_no_default_value(init_cuda):
    """create_condition with no default_value succeeds."""
    g = GraphDefinition()
    try:
        condition = g.create_condition()
    except CUDAError:
        pytest.skip("Conditional nodes not supported (requires CC >= 9.0)")
    assert isinstance(condition, GraphCondition)


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

    g = GraphDefinition()
    condition = g.create_condition(default_value=0)
    alloc = g.allocate(SIZEOF_INT)
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


def test_if_then_false_skips_body(init_cuda):
    """If conditional with default_value=0 does not execute its body."""
    _skip_unless_cc_90()
    _skip_if_no_mempool()

    mod = compile_common_kernels()
    add_one = mod.get_kernel("add_one")
    cfg = LaunchConfig(grid=1, block=1)

    g = GraphDefinition()
    condition = g.create_condition(default_value=0)
    alloc = g.allocate(SIZEOF_INT)
    ms = alloc.memset(alloc.dptr, 0, SIZEOF_INT)
    if_node = ms.if_then(condition)
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

    g = GraphDefinition()
    condition = g.create_condition(default_value=99)
    alloc = g.allocate(SIZEOF_INT)
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
