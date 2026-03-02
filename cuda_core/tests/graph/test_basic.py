# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Basic graph construction and topology tests."""

import numpy as np
import pytest
from cuda.core import Device, GraphBuilder, LaunchConfig, LegacyPinnedMemoryResource, launch
from helpers.graph_kernels import compile_common_kernels


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


@pytest.mark.skipif(tuple(int(i) for i in np.__version__.split(".")[:2]) < (2, 1), reason="need numpy 2.1.0+")
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
