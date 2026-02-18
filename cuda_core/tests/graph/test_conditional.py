# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Conditional graph node tests (if, if-else, switch, while)."""

import ctypes

import numpy as np
import pytest
from cuda.core import Device, GraphBuilder, LaunchConfig, LegacyPinnedMemoryResource, launch
from helpers.graph_kernels import compile_conditional_kernels


@pytest.mark.parametrize(
    "condition_value", [True, False, ctypes.c_bool(True), ctypes.c_bool(False), np.bool_(True), np.bool_(False), 1, 0]
)
@pytest.mark.skipif(tuple(int(i) for i in np.__version__.split(".")[:2]) < (2, 1), reason="need numpy 2.1.0+")
def test_graph_conditional_if(init_cuda, condition_value):
    mod = compile_conditional_kernels(type(condition_value))
    add_one = mod.get_kernel("add_one")
    set_handle = mod.get_kernel("set_handle")

    # Allocate memory
    launch_stream = Device().create_stream()
    mr = LegacyPinnedMemoryResource()
    b = mr.allocate(8)
    arr = np.from_dlpack(b).view(np.int32)
    arr[0] = 0
    arr[1] = 0

    # Begin capture
    gb = Device().create_graph_builder().begin_building()

    # Add Node A (sets condition)
    try:
        handle = gb.create_conditional_handle()
    except RuntimeError as e:
        with pytest.raises(RuntimeError, match="^Driver version"):
            raise e
        gb.end_building()
        b.close()
        pytest.skip("Driver does not support conditional handle")
    launch(gb, LaunchConfig(grid=1, block=1), set_handle, handle, condition_value)

    # Add Node B (if condition)
    gb_if = gb.if_cond(handle).begin_building()
    launch(gb_if, LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)
    gb_if_0, gb_if_1 = gb_if.split(2)
    launch(gb_if_0, LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)
    launch(gb_if_1, LaunchConfig(grid=1, block=1), add_one, arr[1:].ctypes.data)
    gb_if = GraphBuilder.join(gb_if_0, gb_if_1)
    launch(gb_if, LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)
    gb_if.end_building()

    # Add Node C (...)
    # Note: We use the original graph to continue building past the cond node
    launch(gb, LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)

    graph = gb.end_building().complete()

    # Left path increments first value, right path increments second value
    assert arr[0] == 0
    assert arr[1] == 0
    graph.launch(launch_stream)
    launch_stream.sync()
    if condition_value:
        assert arr[0] == 4
        assert arr[1] == 1
    else:
        assert arr[0] == 1
        assert arr[1] == 0

    # Close the memory resource now because the garbage collected might
    # de-allocate it during the next graph builder process
    b.close()


@pytest.mark.parametrize(
    "condition_value", [True, False, ctypes.c_bool(True), ctypes.c_bool(False), np.bool_(True), np.bool_(False), 1, 0]
)
@pytest.mark.skipif(tuple(int(i) for i in np.__version__.split(".")[:2]) < (2, 1), reason="need numpy 2.1.0+")
def test_graph_conditional_if_else(init_cuda, condition_value):
    mod = compile_conditional_kernels(type(condition_value))
    add_one = mod.get_kernel("add_one")
    set_handle = mod.get_kernel("set_handle")

    # Allocate memory
    launch_stream = Device().create_stream()
    mr = LegacyPinnedMemoryResource()
    b = mr.allocate(8)
    arr = np.from_dlpack(b).view(np.int32)
    arr[0] = 0
    arr[1] = 0

    # Begin capture
    gb = Device().create_graph_builder().begin_building()

    # Add Node A (sets condition)
    handle = gb.create_conditional_handle()
    launch(gb, LaunchConfig(grid=1, block=1), set_handle, handle, condition_value)

    # Add Node B (if condition)
    try:
        gb_if, gb_else = gb.if_else(handle)
    except RuntimeError as e:
        with pytest.raises(RuntimeError, match="^(Driver|Binding) version"):
            raise e
        gb.end_building()
        b.close()
        pytest.skip("Driver does not support conditional if-else")

    ## IF nodes
    gb_if = gb_if.begin_building()
    launch(gb_if, LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)
    gb_if_0, gb_if_1 = gb_if.split(2)
    launch(gb_if_0, LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)
    launch(gb_if_1, LaunchConfig(grid=1, block=1), add_one, arr[1:].ctypes.data)
    gb_if = GraphBuilder.join(gb_if_0, gb_if_1)
    launch(gb_if, LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)
    gb_if.end_building()

    ## ELSE nodes
    gb_else = gb_else.begin_building()
    launch(gb_else, LaunchConfig(grid=1, block=1), add_one, arr[1:].ctypes.data)
    launch(gb_else, LaunchConfig(grid=1, block=1), add_one, arr[1:].ctypes.data)
    launch(gb_else, LaunchConfig(grid=1, block=1), add_one, arr[1:].ctypes.data)
    gb_else.end_building()

    # Add Node C (...)
    # Note: We use the original graph to continue building past the cond node
    launch(gb, LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)

    graph = gb.end_building().complete()

    # True condition increments both values, while False increments only second value
    assert arr[0] == 0
    assert arr[1] == 0
    graph.launch(launch_stream)
    launch_stream.sync()
    if condition_value:
        assert arr[0] == 4
        assert arr[1] == 1
    else:
        assert arr[0] == 1
        assert arr[1] == 3

    # Close the memory resource now because the garbage collected might
    # de-allocate it during the next graph builder process
    b.close()


@pytest.mark.parametrize("condition_value", [0, 1, 2, 3])
@pytest.mark.skipif(tuple(int(i) for i in np.__version__.split(".")[:2]) < (2, 1), reason="need numpy 2.1.0+")
def test_graph_conditional_switch(init_cuda, condition_value):
    mod = compile_conditional_kernels(type(condition_value))
    add_one = mod.get_kernel("add_one")
    set_handle = mod.get_kernel("set_handle")

    # Allocate memory
    launch_stream = Device().create_stream()
    mr = LegacyPinnedMemoryResource()
    b = mr.allocate(12)
    arr = np.from_dlpack(b).view(np.int32)
    arr[0] = 0
    arr[1] = 0
    arr[2] = 0

    # Begin capture
    gb = Device().create_graph_builder().begin_building()

    # Add Node A (sets condition)
    handle = gb.create_conditional_handle()
    launch(gb, LaunchConfig(grid=1, block=1), set_handle, handle, condition_value)

    # Add Node B (while condition)
    try:
        gb_case = list(gb.switch(handle, 3))
    except RuntimeError as e:
        with pytest.raises(RuntimeError, match="^(Driver|Binding) version"):
            raise e
        gb.end_building()
        b.close()
        pytest.skip("Driver does not support conditional switch")

    ## Case 0
    gb_case[0].begin_building()
    launch(gb_case[0], LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)
    launch(gb_case[0], LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)
    launch(gb_case[0], LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)
    gb_case[0].end_building()

    ## Case 1
    gb_case[1].begin_building()
    launch(gb_case[1], LaunchConfig(grid=1, block=1), add_one, arr[1:].ctypes.data)
    gb_case_1_left, gb_case_1_right = gb_case[1].split(2)
    launch(gb_case_1_left, LaunchConfig(grid=1, block=1), add_one, arr[1:].ctypes.data)
    launch(gb_case_1_right, LaunchConfig(grid=1, block=1), add_one, arr[2:].ctypes.data)
    gb_case[1] = GraphBuilder.join(gb_case_1_left, gb_case_1_right)
    gb_case[1].end_building()

    ## Case 2
    gb_case[2].begin_building()
    launch(gb_case[2], LaunchConfig(grid=1, block=1), add_one, arr[2:].ctypes.data)
    launch(gb_case[2], LaunchConfig(grid=1, block=1), add_one, arr[2:].ctypes.data)
    launch(gb_case[2], LaunchConfig(grid=1, block=1), add_one, arr[2:].ctypes.data)
    gb_case[2].end_building()

    # Add Node C (...)
    # Note: We use the original graph to continue building past the cond node
    launch(gb, LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)

    graph = gb.end_building().complete()

    # Each case focuses on their own index
    assert arr[0] == 0
    assert arr[1] == 0
    assert arr[2] == 0
    graph.launch(launch_stream)
    launch_stream.sync()
    if condition_value == 0:
        assert arr[0] == 4
        assert arr[1] == 0
        assert arr[2] == 0
    elif condition_value == 1:
        assert arr[0] == 1
        assert arr[1] == 2
        assert arr[2] == 1
    elif condition_value == 2:
        assert arr[0] == 1
        assert arr[1] == 0
        assert arr[2] == 3
    elif condition_value == 3:
        # No branch is taken if case index is out of range
        assert arr[0] == 1
        assert arr[1] == 0
        assert arr[2] == 0

    # Close the memory resource now because the garbage collected might
    # de-allocate it during the next graph builder process
    b.close()


@pytest.mark.parametrize("condition_value", [True, False, 1, 0])
@pytest.mark.skipif(tuple(int(i) for i in np.__version__.split(".")[:2]) < (2, 1), reason="need numpy 2.1.0+")
def test_graph_conditional_while(init_cuda, condition_value):
    mod = compile_conditional_kernels(type(condition_value))
    add_one = mod.get_kernel("add_one")
    loop_kernel = mod.get_kernel("loop_kernel")
    empty_kernel = mod.get_kernel("empty_kernel")

    # Allocate memory
    launch_stream = Device().create_stream()
    mr = LegacyPinnedMemoryResource()
    b = mr.allocate(4)
    arr = np.from_dlpack(b).view(np.int32)
    arr[0] = 0

    # Begin capture
    gb = Device().create_graph_builder().begin_building()

    # Node A is skipped because we can instead use a non-zero default value
    handle = gb.create_conditional_handle(default_value=condition_value)

    # Add Node B (while condition)
    gb_while = gb.while_loop(handle)
    gb_while.begin_building()
    launch(gb_while, LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)
    launch(gb_while, LaunchConfig(grid=1, block=1), add_one, arr.ctypes.data)
    launch(gb_while, LaunchConfig(grid=1, block=1), loop_kernel, handle)
    gb_while.end_building()

    # Add Node C (...)
    # Note: We use the original gb to continue building past the cond node
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)

    graph = gb.end_building().complete()

    # Default value is used to start the loop
    assert arr[0] == 0
    graph.launch(launch_stream)
    launch_stream.sync()
    if condition_value:
        assert arr[0] == 20
    else:
        assert arr[0] == 0

    # Close the memory resource now because the garbage collected might
    # de-allocate it during the next graph builder process
    b.close()
