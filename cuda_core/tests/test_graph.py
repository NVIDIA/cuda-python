# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import numpy as np
import pytest

try:
    from cuda.bindings import nvrtc
except ImportError:
    from cuda import nvrtc
from cuda.core.experimental import (
    Device,
    DeviceMemoryResource,
    DeviceMemoryResourceOptions,
    GraphBuilder,
    GraphCompleteOptions,
    GraphDebugPrintOptions,
    LaunchConfig,
    LegacyPinnedMemoryResource,
    Program,
    ProgramOptions,
    launch,
)
from cuda.core.experimental._utils.cuda_utils import NVRTCError, handle_return
from helpers.buffers import compare_equal_buffers, make_scratch_buffer


def _common_kernels():
    code = """
    __global__ void empty_kernel() {}
    __global__ void add_one(int *a) { *a += 1; }
    """
    arch = "".join(f"{i}" for i in Device().compute_capability)
    program_options = ProgramOptions(std="c++17", arch=f"sm_{arch}")
    prog = Program(code, code_type="c++", options=program_options)
    mod = prog.compile("cubin", name_expressions=("empty_kernel", "add_one"))
    return mod


def _common_kernels_conditional():
    code = """
    extern "C" __device__ __cudart_builtin__ void CUDARTAPI cudaGraphSetConditional(cudaGraphConditionalHandle handle,
                                                                                    unsigned int value);
    __global__ void empty_kernel() {}
    __global__ void add_one(int *a) { *a += 1; }
    __global__ void set_handle(cudaGraphConditionalHandle handle, int value) { cudaGraphSetConditional(handle, value); }
    __global__ void loop_kernel(cudaGraphConditionalHandle handle)
    {
        static int count = 10;
        cudaGraphSetConditional(handle, --count ? 1 : 0);
    }
    """
    arch = "".join(f"{i}" for i in Device().compute_capability)
    program_options = ProgramOptions(std="c++17", arch=f"sm_{arch}")
    prog = Program(code, code_type="c++", options=program_options)
    try:
        mod = prog.compile("cubin", name_expressions=("empty_kernel", "add_one", "set_handle", "loop_kernel"))
    except NVRTCError as e:
        with pytest.raises(NVRTCError, match='error: identifier "cudaGraphConditionalHandle" is undefined'):
            raise e
        nvrtcVersion = handle_return(nvrtc.nvrtcVersion())
        pytest.skip(f"NVRTC version {nvrtcVersion} does not support conditionals")
    return mod


def _common_kernels_alloc():
    code = """
    __global__ void set_zero(char *a, size_t nbytes) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        for (size_t i = idx; i < nbytes; i += stride) {
            a[i] = 0;
        }
    }
    __global__ void add_one(char *a, size_t nbytes) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        for (size_t i = idx; i < nbytes; i += stride) {
            a[i] += 1;
        }
    }
    """
    arch = "".join(f"{i}" for i in Device().compute_capability)
    program_options = ProgramOptions(std="c++17", arch=f"sm_{arch}")
    prog = Program(code, code_type="c++", options=program_options)
    mod = prog.compile("cubin", name_expressions=("set_zero", "add_one"))
    return mod


def test_graph_is_building(init_cuda):
    gb = Device().create_graph_builder()
    assert gb.is_building is False
    gb.begin_building()
    assert gb.is_building is True
    gb.end_building()
    assert gb.is_building is False


def test_graph_straight(init_cuda):
    mod = _common_kernels()
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
    mod = _common_kernels()
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
    mod = _common_kernels()
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
    mod = _common_kernels()
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


@pytest.mark.parametrize("condition_value", [True, False])
@pytest.mark.skipif(tuple(int(i) for i in np.__version__.split(".")[:2]) < (2, 1), reason="need numpy 2.1.0+")
def test_graph_conditional_if(init_cuda, condition_value):
    mod = _common_kernels_conditional()
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


@pytest.mark.parametrize("condition_value", [True, False])
@pytest.mark.skipif(tuple(int(i) for i in np.__version__.split(".")[:2]) < (2, 1), reason="need numpy 2.1.0+")
def test_graph_conditional_if_else(init_cuda, condition_value):
    mod = _common_kernels_conditional()
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
        with pytest.raises(RuntimeError, match="^Driver version"):
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
    mod = _common_kernels_conditional()
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
        with pytest.raises(RuntimeError, match="^Driver version"):
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


@pytest.mark.parametrize("condition_value", [True, False])
@pytest.mark.skipif(tuple(int(i) for i in np.__version__.split(".")[:2]) < (2, 1), reason="need numpy 2.1.0+")
def test_graph_conditional_while(init_cuda, condition_value):
    mod = _common_kernels_conditional()
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


@pytest.mark.skipif(tuple(int(i) for i in np.__version__.split(".")[:2]) < (2, 1), reason="need numpy 2.1.0+")
def test_graph_child_graph(init_cuda):
    mod = _common_kernels()
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
        gb_parent.add_child(gb_child)
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

    # Close the memory resource now because the garbage collected might
    # de-allocate it during the next graph builder process
    b.close()


@pytest.mark.skipif(tuple(int(i) for i in np.__version__.split(".")[:2]) < (2, 1), reason="need numpy 2.1.0+")
def test_graph_update(init_cuda):
    mod = _common_kernels_conditional()
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
            with pytest.raises(RuntimeError, match="^Driver version"):
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
        with pytest.raises(RuntimeError, match="^Driver version"):
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

    # Close the memory resource now because the garbage collected might
    # de-allocate it during the next graph builder process
    b.close()


def test_graph_stream_lifetime(init_cuda):
    mod = _common_kernels()
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


def test_graph_dot_print_options(init_cuda, tmp_path):
    mod = _common_kernels_conditional()
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
    mod = _common_kernels()
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
    mod = _common_kernels()
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


def test_graph_alloc(init_cuda):
    device = Device()
    stream = device.create_stream()
    options = DeviceMemoryResourceOptions(mempool_enabled=False)
    mr = DeviceMemoryResource(device, options=options)

    # Get kernels.
    mod = _common_kernels_alloc()
    set_zero = mod.get_kernel("set_zero")
    add_one = mod.get_kernel("add_one")

    NBYTES = 64
    target = mr.allocate(NBYTES, stream=stream)

    # Begin graph capture.
    gb = Device().create_graph_builder().begin_building(mode="thread_local")

    work_buffer = mr.allocate(NBYTES, stream=gb.stream)
    launch(gb, LaunchConfig(grid=1, block=1), set_zero, int(work_buffer.handle), NBYTES)
    launch(gb, LaunchConfig(grid=1, block=1), add_one, int(work_buffer.handle), NBYTES)
    launch(gb, LaunchConfig(grid=1, block=1), add_one, int(work_buffer.handle), NBYTES)
    target.copy_from(work_buffer, stream=gb.stream)

    # Finalize the graph.
    graph = gb.end_building().complete()

    # Upload and launch
    graph.upload(stream)
    graph.launch(stream)
    stream.sync()

    # Check the result.
    expected_buffer = make_scratch_buffer(device, 2, NBYTES)
    compare_buffer = make_scratch_buffer(device, 0, NBYTES)
    compare_buffer.copy_from(target, stream=stream)
    stream.sync()
    assert compare_equal_buffers(expected_buffer, compare_buffer)

# TODO: check that mr.attributes is None with mempool_enabled=False
