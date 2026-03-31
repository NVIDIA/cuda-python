# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Tests for whole-graph update (Graph.update)."""

import numpy as np
import pytest
from helpers.graph_kernels import compile_common_kernels, compile_conditional_kernels

from cuda.core import Device, LaunchConfig, LegacyPinnedMemoryResource, launch
from cuda.core._graph._graphdef import GraphDef
from cuda.core._utils.cuda_utils import CUDAError


@pytest.mark.parametrize("builder", ["GraphBuilder", "GraphDef"])
@pytest.mark.skipif(tuple(int(i) for i in np.__version__.split(".")[:2]) < (2, 1), reason="need numpy 2.1.0+")
def test_graph_update_kernel_args(init_cuda, builder):
    """Update redirects a kernel to write to a different pointer."""
    mod = compile_common_kernels()
    add_one = mod.get_kernel("add_one")

    launch_stream = Device().create_stream()
    mr = LegacyPinnedMemoryResource()
    b = mr.allocate(8)
    arr = np.from_dlpack(b).view(np.int32)
    arr[0] = 0
    arr[1] = 0

    if builder == "GraphBuilder":

        def build(ptr):
            gb = Device().create_graph_builder().begin_building()
            launch(gb, LaunchConfig(grid=1, block=1), add_one, ptr)
            launch(gb, LaunchConfig(grid=1, block=1), add_one, ptr)
            finished = gb.end_building()
            return finished.complete(), finished
    elif builder == "GraphDef":

        def build(ptr):
            g = GraphDef()
            g.launch(LaunchConfig(grid=1, block=1), add_one, ptr)
            g.launch(LaunchConfig(grid=1, block=1), add_one, ptr)
            return g.instantiate(), g

    graph, _ = build(arr[0:].ctypes.data)
    _, source1 = build(arr[1:].ctypes.data)

    graph.launch(launch_stream)
    launch_stream.sync()
    assert arr[0] == 2
    assert arr[1] == 0

    graph.update(source1)
    graph.launch(launch_stream)
    launch_stream.sync()
    assert arr[0] == 2
    assert arr[1] == 2

    b.close()


@pytest.mark.skipif(tuple(int(i) for i in np.__version__.split(".")[:2]) < (2, 1), reason="need numpy 2.1.0+")
def test_graph_update_conditional(init_cuda):
    """Update swaps conditional switch graphs with matching topology."""
    mod = compile_conditional_kernels(int)
    add_one = mod.get_kernel("add_one")

    launch_stream = Device().create_stream()
    mr = LegacyPinnedMemoryResource()
    b = mr.allocate(12)
    arr = np.from_dlpack(b).view(np.int32)
    arr[0] = 0
    arr[1] = 0
    arr[2] = 0

    def build_graph(condition_value):
        # Begin capture
        gb = Device().create_graph_builder().begin_building()

        # Add Node A (sets condition)
        handle = gb.create_conditional_handle(default_value=condition_value)

        # Add Node B (while condition)
        try:
            gb_case = list(gb.switch(handle, 3))
        except Exception as e:
            with pytest.raises(RuntimeError, match="^(Driver|Binding) version"):
                raise e
            gb.end_building()
            raise e

        ## Case 0
        gb_case[0].begin_building()
        launch(gb_case[0], LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)
        launch(gb_case[0], LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)
        launch(gb_case[0], LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)
        gb_case[0].end_building()

        ## Case 1
        gb_case[1].begin_building()
        launch(gb_case[1], LaunchConfig(grid=1, block=1), add_one, arr[1:].ctypes.data)
        launch(gb_case[1], LaunchConfig(grid=1, block=1), add_one, arr[1:].ctypes.data)
        launch(gb_case[1], LaunchConfig(grid=1, block=1), add_one, arr[1:].ctypes.data)
        gb_case[1].end_building()

        ## Case 2
        gb_case[2].begin_building()
        launch(gb_case[2], LaunchConfig(grid=1, block=1), add_one, arr[2:].ctypes.data)
        launch(gb_case[2], LaunchConfig(grid=1, block=1), add_one, arr[2:].ctypes.data)
        launch(gb_case[2], LaunchConfig(grid=1, block=1), add_one, arr[2:].ctypes.data)
        gb_case[2].end_building()

        return gb.end_building()

    try:
        graph_variants = [build_graph(0), build_graph(1), build_graph(2)]
    except Exception as e:
        with pytest.raises(RuntimeError, match="^(Driver|Binding) version"):
            raise e
        b.close()
        pytest.skip("Driver does not support conditional switch")

    # Launch the first graph
    graph = graph_variants[0].complete()
    graph.launch(launch_stream)
    launch_stream.sync()
    assert arr[0] == 3
    assert arr[1] == 0
    assert arr[2] == 0

    # Update with second variant and launch again
    graph.update(graph_variants[1])
    graph.launch(launch_stream)
    launch_stream.sync()
    assert arr[0] == 3
    assert arr[1] == 3
    assert arr[2] == 0

    # Update with third variant and launch again
    graph.update(graph_variants[2])
    graph.launch(launch_stream)
    launch_stream.sync()
    assert arr[0] == 3
    assert arr[1] == 3
    assert arr[2] == 3

    # Close the memory resource now because the garbage collected might
    # de-allocate it during the next graph builder process
    b.close()


# =============================================================================
# Error cases
# =============================================================================


def test_graph_update_unfinished_builder(init_cuda):
    """Update with an unfinished GraphBuilder raises ValueError."""
    mod = compile_common_kernels()
    empty_kernel = mod.get_kernel("empty_kernel")

    gb_finished = Device().create_graph_builder().begin_building()
    launch(gb_finished, LaunchConfig(grid=1, block=1), empty_kernel)
    graph = gb_finished.end_building().complete()

    gb_unfinished = Device().create_graph_builder().begin_building()
    launch(gb_unfinished, LaunchConfig(grid=1, block=1), empty_kernel)

    with pytest.raises(ValueError, match="Graph has not finished building"):
        graph.update(gb_unfinished)

    gb_unfinished.end_building()


def test_graph_update_topology_mismatch(init_cuda):
    """Update with a different topology raises CUDAError."""
    mod = compile_common_kernels()
    empty_kernel = mod.get_kernel("empty_kernel")

    # Two-node graph
    gb1 = Device().create_graph_builder().begin_building()
    launch(gb1, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(gb1, LaunchConfig(grid=1, block=1), empty_kernel)
    graph = gb1.end_building().complete()

    # Three-node graph (different topology)
    gb2 = Device().create_graph_builder().begin_building()
    launch(gb2, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(gb2, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(gb2, LaunchConfig(grid=1, block=1), empty_kernel)
    gb2.end_building()

    expected = r"Graph update failed: The update failed because the topology changed \(CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED\)"
    with pytest.raises(CUDAError, match=expected):
        graph.update(gb2)


def test_graph_update_wrong_type(init_cuda):
    """Update with an invalid type raises TypeError."""
    mod = compile_common_kernels()
    empty_kernel = mod.get_kernel("empty_kernel")

    gb = Device().create_graph_builder().begin_building()
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    graph = gb.end_building().complete()

    with pytest.raises(TypeError, match="expected GraphBuilder or GraphDef"):
        graph.update("not a graph")
