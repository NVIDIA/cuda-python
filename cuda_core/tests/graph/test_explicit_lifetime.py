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

from cuda.core import Device, EventOptions, Kernel, LaunchConfig
from cuda.core._graph._graphdef import (
    ChildGraphNode,
    ConditionalNode,
    GraphDef,
    KernelNode,
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
    child_node = next(n for n in middle_nodes if isinstance(n, ChildGraphNode))
    grandchild = child_node.child_graph

    del outer, outer_node, middle, inner, middle_ref, middle_nodes, child_node
    gc.collect()

    assert len(grandchild.nodes()) == 1


# =============================================================================
# Event lifetime — event nodes should keep the Event alive
# =============================================================================


def test_event_record_node_keeps_event_alive(init_cuda):
    """EventRecordNode should keep the Event alive after original is deleted."""
    dev = Device()
    g = GraphDef()
    alloc = g.alloc(1024)

    event = dev.create_event(EventOptions(enable_timing=False))
    node = alloc.record_event(event)

    del event
    gc.collect()

    retrieved = node.event
    assert retrieved.is_done is True


def test_event_wait_node_keeps_event_alive(init_cuda):
    """EventWaitNode should keep the Event alive after original is deleted."""
    dev = Device()
    g = GraphDef()
    alloc = g.alloc(1024)

    event = dev.create_event(EventOptions(enable_timing=False))
    node = alloc.wait_event(event)

    del event
    gc.collect()

    retrieved = node.event
    assert retrieved.is_done is True


def test_event_record_node_preserves_metadata(init_cuda):
    """Reconstructed EventRecordNode recovers full Event metadata via reverse lookup."""
    dev = Device()
    g = GraphDef()

    event = dev.create_event(EventOptions(enable_timing=True, busy_waited_sync=True))
    node = g.record_event(event)

    reconstructed = node.event
    assert reconstructed.is_timing_disabled is False
    assert reconstructed.is_sync_busy_waited is True
    assert reconstructed.is_ipc_enabled is False
    assert reconstructed.device is not None


def test_event_wait_node_preserves_metadata(init_cuda):
    """Reconstructed EventWaitNode recovers full Event metadata via reverse lookup."""
    dev = Device()
    g = GraphDef()

    event = dev.create_event(EventOptions(enable_timing=False))
    node = g.wait_event(event)

    reconstructed = node.event
    assert reconstructed.is_timing_disabled is True
    assert reconstructed.is_sync_busy_waited is False
    assert reconstructed.device is not None


def test_event_metadata_survives_gc(init_cuda):
    """Event metadata is preserved through reverse lookup even after original is GC'd."""
    dev = Device()
    g = GraphDef()

    event = dev.create_event(EventOptions(enable_timing=True, busy_waited_sync=True))
    node = g.record_event(event)

    del event
    gc.collect()

    retrieved = node.event
    assert retrieved.is_timing_disabled is False
    assert retrieved.is_sync_busy_waited is True
    assert retrieved.is_done is True


def test_event_survives_graph_instantiation_and_execution(init_cuda):
    """Graph with event nodes executes correctly after original Event is deleted."""
    dev = Device()
    g = GraphDef()

    event = dev.create_event(EventOptions(enable_timing=False))
    rec = g.record_event(event)
    rec.wait_event(event)

    del event
    gc.collect()

    graph = g.instantiate()
    stream = dev.create_stream()
    graph.launch(stream)
    stream.sync()


def test_event_survives_graph_clone_and_execution(init_cuda):
    """Cloned graph with event nodes executes after original Event is deleted.

    This is the critical test for CUDA User Objects: a graph clone does
    not inherit Python-level references, so only user objects (which
    propagate through cuGraphClone) can keep the event alive.
    """
    from cuda.core._utils.cuda_utils import driver, handle_return

    dev = Device()
    g = GraphDef()

    event = dev.create_event(EventOptions(enable_timing=False))
    rec = g.record_event(event)
    rec.wait_event(event)

    cloned_cu_graph = handle_return(driver.cuGraphClone(driver.CUgraph(g.handle)))

    del event, g, rec
    gc.collect()

    graph_exec = handle_return(driver.cuGraphInstantiate(cloned_cu_graph, 0))
    stream = dev.create_stream()
    handle_return(driver.cuGraphLaunch(graph_exec, driver.CUstream(int(stream.handle))))
    stream.sync()


# =============================================================================
# Kernel lifetime — kernel nodes should keep the Kernel/Module alive
# =============================================================================


def test_kernel_node_keeps_kernel_alive(init_cuda):
    """KernelNode should keep the Kernel alive after original is deleted."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)

    g = GraphDef()
    node = g.launch(config, kernel)

    del kernel, mod
    gc.collect()

    retrieved = node.kernel
    assert retrieved.attributes.max_threads_per_block() > 0


def test_kernel_survives_graph_instantiation_and_execution(init_cuda):
    """Graph with kernel node executes correctly after Kernel/Module is deleted."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)

    g = GraphDef()
    g.launch(config, kernel)

    del kernel, mod
    gc.collect()

    graph = g.instantiate()
    stream = Device().create_stream()
    graph.launch(stream)
    stream.sync()


def test_kernel_survives_graph_clone_and_execution(init_cuda):
    """Cloned graph with kernel node executes after Kernel/Module is deleted.

    Validates that CUDA User Objects keep the kernel's library alive
    through graph cloning (where Python-level references are lost).
    """
    from cuda.core._utils.cuda_utils import driver, handle_return

    dev = Device()
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)

    g = GraphDef()
    g.launch(config, kernel)

    cloned_cu_graph = handle_return(driver.cuGraphClone(driver.CUgraph(g.handle)))

    del kernel, mod, g
    gc.collect()

    graph_exec = handle_return(driver.cuGraphInstantiate(cloned_cu_graph, 0))
    stream = dev.create_stream()
    handle_return(driver.cuGraphLaunch(graph_exec, driver.CUstream(int(stream.handle))))
    stream.sync()


# =============================================================================
# Kernel handle recovery — from_handle and graph node reconstruction
# =============================================================================


def test_kernel_from_handle_recovers_library(init_cuda):
    """Kernel.from_handle on a cuda.core-created kernel recovers the library
    dependency, keeping it alive after the original objects are deleted."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    handle = int(kernel.handle)

    reconstructed = Kernel.from_handle(handle)

    del kernel, mod
    gc.collect()

    assert reconstructed.attributes.max_threads_per_block() > 0


def test_kernel_node_reconstruction_preserves_validity(init_cuda):
    """A KernelNode reconstructed via DAG traversal has a valid kernel,
    kept alive by user objects and existing node references."""
    mod = compile_common_kernels()
    kernel = mod.get_kernel("empty_kernel")
    config = LaunchConfig(grid=1, block=1)

    g = GraphDef()
    kernel_node = g.launch(config, kernel)
    # Chain a second node so we can reconstruct the kernel node via pred
    event = Device().create_event()
    successor = kernel_node.record_event(event)

    del kernel, mod
    gc.collect()

    # Reconstruct the kernel node through DAG traversal
    # successor.pred -> Node._create -> KernelNode._create_from_driver
    # -> create_kernel_handle_ref -> handle recovery
    reconstructed = successor.pred[0]
    assert isinstance(reconstructed, KernelNode)
    assert reconstructed.kernel.attributes.max_threads_per_block() > 0

    graph = g.instantiate()
    stream = Device().create_stream()
    graph.launch(stream)
    stream.sync()
