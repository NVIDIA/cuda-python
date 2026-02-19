# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Graph options and build mode tests."""

import pytest
from cuda.core import Device, GraphBuilder, GraphCompleteOptions, GraphDebugPrintOptions, LaunchConfig, launch
from helpers.graph_kernels import compile_common_kernels, compile_conditional_kernels


def test_graph_dot_print_options(init_cuda, tmp_path):
    mod = compile_conditional_kernels(bool)
    set_handle = mod.get_kernel("set_handle")
    empty_kernel = mod.get_kernel("empty_kernel")

    # Begin capture
    gb = Device().create_graph_builder().begin_building()

    # Add Node A (sets condition)
    handle = gb.create_conditional_handle()
    launch(gb, LaunchConfig(grid=1, block=1), set_handle, handle, False)

    # Add Node B (if condition)
    gb_if = gb.if_cond(handle).begin_building()
    launch(gb_if, LaunchConfig(grid=1, block=1), empty_kernel)
    gb_if_0, gb_if_1 = gb_if.split(2)
    launch(gb_if_0, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(gb_if_1, LaunchConfig(grid=1, block=1), empty_kernel)
    gb_if = GraphBuilder.join(gb_if_0, gb_if_1)
    launch(gb_if, LaunchConfig(grid=1, block=1), empty_kernel)
    gb_if.end_building()

    # Add Node C (...)
    # Note: We use the original graph to continue building past the cond node
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    gb.end_building()

    # Print using all options
    path = bytes(str(tmp_path / "vlad.dot"), "utf-8")
    options = GraphDebugPrintOptions(**{field: True for field in GraphDebugPrintOptions.__dataclass_fields__})
    gb.debug_dot_print(path, options)


def test_graph_complete_options(init_cuda):
    mod = compile_common_kernels()
    empty_kernel = mod.get_kernel("empty_kernel")
    launch_stream = Device().create_stream()

    # Simple linear topology
    gb = Device().create_graph_builder().begin_building()
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    gb.end_building()

    options = GraphCompleteOptions(auto_free_on_launch=True)
    gb.complete(options).close()
    options = GraphCompleteOptions(upload_stream=launch_stream)
    gb.complete(options).close()
    options = GraphCompleteOptions(device_launch=True)
    gb.complete(options).close()
    options = GraphCompleteOptions(use_node_priority=True)
    gb.complete(options).close()


def test_graph_build_mode(init_cuda):
    mod = compile_common_kernels()
    empty_kernel = mod.get_kernel("empty_kernel")

    # Simple linear topology
    gb = Device().create_graph_builder().begin_building(mode="global")
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    gb.end_building()

    gb = Device().create_graph_builder().begin_building(mode="thread_local")
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    gb.end_building()

    gb = Device().create_graph_builder().begin_building(mode="relaxed")
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    gb.end_building()

    with pytest.raises(ValueError, match="^Unsupported build mode:"):
        gb = Device().create_graph_builder().begin_building(mode=None)
