# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Tests for whole-graph update (Graph.update)."""

import numpy as np
import pytest
from helpers.graph_kernels import compile_conditional_kernels

from cuda.core import Device, LaunchConfig, LegacyPinnedMemoryResource, launch


@pytest.mark.skipif(tuple(int(i) for i in np.__version__.split(".")[:2]) < (2, 1), reason="need numpy 2.1.0+")
def test_graph_update(init_cuda):
    mod = compile_conditional_kernels(int)
    add_one = mod.get_kernel("add_one")

    # Allocate memory
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
    assert arr[0] == 0
    assert arr[1] == 0
    assert arr[2] == 0
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

    b.close()
