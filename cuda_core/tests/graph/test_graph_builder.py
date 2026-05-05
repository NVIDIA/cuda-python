# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GraphBuilder stream capture tests."""

import numpy as np
import pytest
from helpers.graph_kernels import compile_common_kernels, compile_conditional_kernels
from helpers.marks import requires_module
from helpers.misc import try_create_condition

from cuda.core import Device, LaunchConfig, LegacyPinnedMemoryResource, launch
from cuda.core.graph import GraphBuilder, GraphDefinition


def test_graph_is_building(init_cuda):
    gb = Device().create_graph_builder()
    assert gb.is_building is False
    gb.begin_building()
    assert gb.is_building is True
    gb.end_building()
    assert gb.is_building is False


def test_graph_straight(init_cuda):
    mod = compile_common_kernels()
    empty_kernel = mod.get_kernel("empty_kernel")
    launch_stream = Device().create_stream()

    # Simple linear topology
    gb = Device().create_graph_builder().begin_building()
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    graph = gb.end_building().complete()

    # Sanity upload and launch
    graph.upload(launch_stream)
    graph.launch(launch_stream)
    launch_stream.sync()


def test_graph_fork_join(init_cuda):
    mod = compile_common_kernels()
    empty_kernel = mod.get_kernel("empty_kernel")
    launch_stream = Device().create_stream()

    # Simple diamond topology
    gb = Device().create_graph_builder().begin_building()
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)

    with pytest.raises(ValueError, match="^Invalid split count: expecting >= 2, got 1"):
        gb.split(1)

    left, right = gb.split(2)
    launch(left, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(left, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(right, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(right, LaunchConfig(grid=1, block=1), empty_kernel)

    with pytest.raises(ValueError, match="^Must join with at least two graph builders"):
        GraphBuilder.join(left)

    gb = GraphBuilder.join(left, right)

    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    graph = gb.end_building().complete()

    # Sanity upload and launch
    graph.upload(launch_stream)
    graph.launch(launch_stream)
    launch_stream.sync()


def test_graph_is_join_required(init_cuda):
    mod = compile_common_kernels()
    empty_kernel = mod.get_kernel("empty_kernel")

    # Starting builder is always primary
    gb = Device().create_graph_builder()
    assert gb.is_join_required is False
    gb.begin_building()

    # Create root node
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)

    # First returned builder is always the original
    first_split_builders = gb.split(3)
    assert first_split_builders[0] is gb

    # Only the original builder need not join
    assert first_split_builders[0].is_join_required is False
    for builder in first_split_builders[1:]:
        assert builder.is_join_required is True

    # Launch kernel on each split
    for builder in first_split_builders:
        launch(builder, LaunchConfig(grid=1, block=1), empty_kernel)

    # Splitting on new builder will all require joining
    second_split_builders = first_split_builders[-1]
    first_split_builders = first_split_builders[0:-1]
    second_split_builders = second_split_builders.split(3)
    for builder in second_split_builders:
        assert builder.is_join_required is True

    # Launch kernel on each second split
    for builder in second_split_builders:
        launch(builder, LaunchConfig(grid=1, block=1), empty_kernel)

    # Joined builder requires joining if all builder need to join
    gb = GraphBuilder.join(*second_split_builders)
    assert gb.is_join_required is True
    gb = GraphBuilder.join(gb, *first_split_builders)
    assert gb.is_join_required is False

    # Create final node
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    gb.end_building().complete()


@requires_module(np, "2.1")
def test_graph_repeat_capture(init_cuda):
    mod = compile_common_kernels()
    add_one = mod.get_kernel("add_one")

    # Allocate memory
    launch_stream = Device().create_stream()
    mr = LegacyPinnedMemoryResource()
    b = mr.allocate(4)
    arr = np.from_dlpack(b).view(np.int32)
    arr[0] = 0

    # Launch the graph once
    gb = launch_stream.create_graph_builder().begin_building()
    launch(gb, LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)
    graph = gb.end_building().complete()

    # Run the graph once
    graph.launch(launch_stream)
    launch_stream.sync()
    assert arr[0] == 1

    # Continue capturing to extend the graph
    with pytest.raises(RuntimeError, match="^Cannot resume building after building has ended."):
        gb.begin_building()

    # Graph can be re-launched
    graph.launch(launch_stream)
    graph.launch(launch_stream)
    graph.launch(launch_stream)
    launch_stream.sync()
    assert arr[0] == 4

    # Close the memory resource now because the garbage collected might
    # de-allocate it during the next graph builder process
    b.close()


def test_graph_capture_errors(init_cuda):
    gb = Device().create_graph_builder()
    with pytest.raises(RuntimeError, match="^Graph has not finished building."):
        gb.complete()

    gb.begin_building()
    with pytest.raises(RuntimeError, match="^Graph has not finished building."):
        gb.complete()
    gb.end_building().complete()


def test_graph_begin_building_twice(init_cuda):
    """Calling begin_building() while already capturing is a clear error."""
    gb = Device().create_graph_builder()
    gb.begin_building()
    with pytest.raises(RuntimeError, match="^Graph builder is already building."):
        gb.begin_building()
    gb.end_building()


def test_graph_split_requires_building(init_cuda):
    """A builder must be capturing before it can be split."""
    gb = Device().create_graph_builder()
    with pytest.raises(RuntimeError, match="^Graph builder must be building before it can be split."):
        gb.split(2)


def test_graph_complete_after_close_forked(init_cuda):
    """complete() on a forked builder closed via join() must not deref a null handle."""
    mod = compile_common_kernels()
    empty_kernel = mod.get_kernel("empty_kernel")

    gb = Device().create_graph_builder().begin_building()
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    left, right = gb.split(2)
    launch(left, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(right, LaunchConfig(grid=1, block=1), empty_kernel)

    # join() closes the non-root builder (right); it must now be rejected, not crash.
    GraphBuilder.join(left, right)
    with pytest.raises(RuntimeError, match="^Graph builder has been closed."):
        right.complete()


def test_graph_update_after_source_close(init_cuda):
    """Graph.update() with a closed source builder must raise, not deref a null handle."""
    mod = compile_common_kernels()
    empty_kernel = mod.get_kernel("empty_kernel")

    gb = Device().create_graph_builder().begin_building()
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    graph = gb.end_building().complete()

    source = Device().create_graph_builder().begin_building()
    launch(source, LaunchConfig(grid=1, block=1), empty_kernel)
    source.end_building()
    source.close()

    with pytest.raises(ValueError, match="^Source graph builder has been closed."):
        graph.update(source)


def test_graph_gc_mid_capture(init_cuda):
    """Dropping a builder mid-capture ends the orphaned capture so the stream stays usable."""
    import gc

    mod = compile_common_kernels()
    empty_kernel = mod.get_kernel("empty_kernel")

    stream = Device().create_stream()
    gb = stream.create_graph_builder().begin_building()
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)

    # Drop the builder without end_building()/close(); __dealloc__ must end the capture.
    del gb
    gc.collect()

    # If the capture were left active, the stream would be poisoned for new work.
    launch(stream, LaunchConfig(grid=1, block=1), empty_kernel)
    stream.sync()
    stream.close()


def test_graph_embed_non_builder(init_cuda):
    """embed() rejects a non-GraphBuilder argument with a TypeError."""
    gb = Device().create_graph_builder().begin_building()
    with pytest.raises(TypeError):
        gb.embed(object())
    gb.end_building()


def test_graph_capture_callback_python(init_cuda):
    results = []

    def my_callback():
        results.append(42)

    launch_stream = Device().create_stream()
    gb = launch_stream.create_graph_builder().begin_building()

    with pytest.raises(ValueError, match="user_data is only supported"):
        gb.callback(my_callback, user_data=b"hello")

    gb.callback(my_callback)
    graph = gb.end_building().complete()

    graph.launch(launch_stream)
    launch_stream.sync()

    assert results == [42]


def test_graph_capture_callback_ctypes(init_cuda):
    import ctypes

    CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    result = [0]

    @CALLBACK
    def read_byte(data):
        result[0] = ctypes.cast(data, ctypes.POINTER(ctypes.c_uint8))[0]

    launch_stream = Device().create_stream()
    gb = launch_stream.create_graph_builder().begin_building()
    gb.callback(read_byte, user_data=bytes([0xAB]))
    graph = gb.end_building().complete()

    graph.launch(launch_stream)
    launch_stream.sync()

    assert result[0] == 0xAB


@pytest.mark.skipif(tuple(int(i) for i in np.__version__.split(".")[:2]) < (2, 1), reason="need numpy 2.1.0+")
def test_graph_child_graph(init_cuda):
    mod = compile_common_kernels()
    add_one = mod.get_kernel("add_one")

    # Allocate memory
    launch_stream = Device().create_stream()
    mr = LegacyPinnedMemoryResource()
    b = mr.allocate(8)
    arr = np.from_dlpack(b).view(np.int32)
    arr[0] = 0
    arr[1] = 0

    # Capture the child graph
    gb_child = Device().create_graph_builder().begin_building()
    launch(gb_child, LaunchConfig(grid=1, block=1), add_one, arr[1:].ctypes.data)
    launch(gb_child, LaunchConfig(grid=1, block=1), add_one, arr[1:].ctypes.data)
    launch(gb_child, LaunchConfig(grid=1, block=1), add_one, arr[1:].ctypes.data)
    gb_child.end_building()

    # Capture the parent graph
    gb_parent = Device().create_graph_builder().begin_building()
    launch(gb_parent, LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)

    ## Add child
    try:
        gb_parent.embed(gb_child)
    except NotImplementedError as e:
        with pytest.raises(
            NotImplementedError,
            match="^Launching child graphs is not implemented for versions older than CUDA 12",
        ):
            raise e
        gb_parent.end_building()
        b.close()
        pytest.skip("Launching child graphs is not implemented for versions older than CUDA 12")

    launch(gb_parent, LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)
    graph = gb_parent.end_building().complete()

    # Parent updates first value, child updates second value
    assert arr[0] == 0
    assert arr[1] == 0
    graph.launch(launch_stream)
    launch_stream.sync()
    assert arr[0] == 2
    assert arr[1] == 3

    b.close()


def test_graph_close_is_idempotent(init_cuda):
    """Re-entrant close must not double-destroy the graph exec (Glasswing V18.1)."""
    mod = compile_common_kernels()
    empty_kernel = mod.get_kernel("empty_kernel")

    gb = Device().create_graph_builder().begin_building()
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    graph = gb.end_building().complete()
    gb.close()

    graph.close()
    graph.close()
    assert int(graph.handle) == 0


def test_graph_stream_lifetime(init_cuda):
    mod = compile_common_kernels()
    empty_kernel = mod.get_kernel("empty_kernel")

    # Create simple graph from device
    gb = Device().create_graph_builder().begin_building()
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    graph = gb.end_building().complete()

    # Destroy simple graph and builder
    gb.close()
    graph.close()

    # Create simple graph from stream
    stream = Device().create_stream()
    gb = stream.create_graph_builder().begin_building()
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    graph = gb.end_building().complete()

    # Destroy simple graph and builder
    gb.close()
    graph.close()

    # Verify the stream can still launch work
    launch(stream, LaunchConfig(grid=1, block=1), empty_kernel)
    stream.sync()

    # Destroy the stream
    stream.close()


# ---------------------------------------------------------------------------
# GraphBuilder.graph_definition
# ---------------------------------------------------------------------------


def test_graph_definition_returns_graph_definition_after_end_building(init_cuda):
    """Primary builder exposes its captured graph as a GraphDefinition after end_building()."""
    mod = compile_common_kernels()
    empty_kernel = mod.get_kernel("empty_kernel")

    gb = Device().create_graph_builder().begin_building()
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    gb.end_building()

    gd = gb.graph_definition
    assert isinstance(gd, GraphDefinition)
    # The captured graph must contain the launched kernels.
    assert len(gd.nodes()) == 2


def test_graph_definition_raises_before_begin_building(init_cuda):
    """Primary builder has no graph allocated before begin_building()."""
    gb = Device().create_graph_builder()
    with pytest.raises(RuntimeError, match="before begin_building"):
        _ = gb.graph_definition


def test_graph_definition_raises_during_capture(init_cuda):
    """graph_definition is unsafe while the driver is actively capturing."""
    gb = Device().create_graph_builder().begin_building()
    try:
        with pytest.raises(RuntimeError, match="capture is in"):
            _ = gb.graph_definition
    finally:
        gb.end_building()


def test_graph_definition_raises_for_forked(init_cuda):
    """Forked builders share the primary's graph; their property must raise."""
    mod = compile_common_kernels()
    empty_kernel = mod.get_kernel("empty_kernel")

    gb = Device().create_graph_builder().begin_building()
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    primary, sibling = gb.split(2)
    try:
        with pytest.raises(RuntimeError, match="forked"):
            _ = sibling.graph_definition
    finally:
        sibling = GraphBuilder.join(primary, sibling)
        sibling.end_building()


def test_graph_definition_shares_ownership(init_cuda):
    """Closing the builder must not invalidate a held GraphDefinition."""
    mod = compile_common_kernels()
    empty_kernel = mod.get_kernel("empty_kernel")

    gb = Device().create_graph_builder().begin_building()
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    gb.end_building()

    gd = gb.graph_definition
    gb.close()
    # The shared CUgraph keeps the graph alive.
    assert len(gd.nodes()) == 1


def test_graph_definition_round_trips_through_explicit_api(init_cuda):
    """Mutating via the explicit API survives complete() and runs correctly."""
    mod = compile_common_kernels()
    add_one = mod.get_kernel("add_one")

    launch_stream = Device().create_stream()
    mr = LegacyPinnedMemoryResource()
    b = mr.allocate(4)
    arr = np.from_dlpack(b).view(np.int32)
    arr[0] = 0

    gb = launch_stream.create_graph_builder().begin_building()
    launch(gb, LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)
    gb.end_building()

    # Add a second add_one through the explicit GraphDefinition view.
    gd = gb.graph_definition
    captured_node = next(iter(gd.nodes()))
    captured_node.launch(LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)
    assert len(gd.nodes()) == 2

    graph = gb.complete()
    graph.launch(launch_stream)
    launch_stream.sync()
    assert arr[0] == 2

    b.close()


@requires_module(np, "2.1")
def test_graph_definition_hybrid_conditional_body(init_cuda):
    """Populate a conditional body entirely through the explicit API.

    This is the headline hybrid flow enabled by the new property:
    ``if_then`` returns a ``GraphBuilder`` for the body, but instead of
    calling ``begin_building`` and capturing into it, we reach for
    ``graph_definition`` and add nodes through the explicit API.
    """
    mod = compile_conditional_kernels(int)
    add_one = mod.get_kernel("add_one")
    set_handle = mod.get_kernel("set_handle")

    launch_stream = Device().create_stream()
    mr = LegacyPinnedMemoryResource()
    b = mr.allocate(4)
    arr = np.from_dlpack(b).view(np.int32)
    arr[0] = 0

    gb = Device().create_graph_builder().begin_building()
    condition = try_create_condition(gb)
    launch(gb, LaunchConfig(grid=1, block=1), set_handle, condition, 1)
    body_gb = gb.if_then(condition)

    # Skip body_gb.begin_building() entirely -- the body graph already
    # exists at conditional-node creation time and is exposed here.
    body_def = body_gb.graph_definition
    assert isinstance(body_def, GraphDefinition)
    assert len(body_def.nodes()) == 0
    body_def.launch(LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)

    graph = gb.end_building().complete()
    graph.launch(launch_stream)
    launch_stream.sync()
    assert arr[0] == 1

    b.close()


@requires_module(np, "2.1")
def test_graph_definition_conditional_body_after_capture(init_cuda):
    """Capture into a conditional body, then augment it via the explicit API."""
    mod = compile_conditional_kernels(int)
    add_one = mod.get_kernel("add_one")
    set_handle = mod.get_kernel("set_handle")

    launch_stream = Device().create_stream()
    mr = LegacyPinnedMemoryResource()
    b = mr.allocate(4)
    arr = np.from_dlpack(b).view(np.int32)
    arr[0] = 0

    gb = Device().create_graph_builder().begin_building()
    condition = try_create_condition(gb)
    launch(gb, LaunchConfig(grid=1, block=1), set_handle, condition, 1)
    body_gb = gb.if_then(condition).begin_building()

    # Capture one increment into the body.
    launch(body_gb, LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)
    body_gb.end_building()

    # Add a second increment via the explicit API on the same body graph.
    body_def = body_gb.graph_definition
    captured_node = next(iter(body_def.nodes()))
    captured_node.launch(LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)
    assert len(body_def.nodes()) == 2

    graph = gb.end_building().complete()
    graph.launch(launch_stream)
    launch_stream.sync()
    assert arr[0] == 2

    b.close()


@requires_module(np, "2.1")
def test_graph_definition_conditional_body_during_capture_raises(init_cuda):
    """The CAPTURING-state guard fires for conditional bodies too."""
    gb = Device().create_graph_builder().begin_building()
    condition = try_create_condition(gb)
    body_gb = gb.if_then(condition).begin_building()
    try:
        with pytest.raises(RuntimeError, match="capture is in"):
            _ = body_gb.graph_definition
    finally:
        body_gb.end_building()
        gb.end_building()
