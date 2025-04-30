# Copyright 2025 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.


from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch, GraphBuilder

# from cuda.core.experimental import Device, Stream, StreamOptions
# from cuda.core.experimental._event import Event
# from cuda.core.experimental._stream import LEGACY_DEFAULT_STREAM, PER_THREAD_DEFAULT_STREAM, default_stream


def test_graph_is_capture_alive(init_cuda):
    gb = Device().create_gb()
    assert gb.is_capture_active() is False
    gb.begin_building()
    assert gb.is_capture_active() is True
    gb.end_building()
    assert gb.is_capture_active() is False


def test_graph_straight(init_cuda):
    # TODO: Maybe share these between tests?
    code = """
    __global__ void empty_kernel() {}
    """
    arch = "".join(f"{i}" for i in Device().compute_capability)
    program_options = ProgramOptions(std="c++17", arch=f"sm_{arch}")
    prog = Program(code, code_type="c++", options=program_options)
    mod = prog.compile("cubin", name_expressions=("empty_kernel",))
    empty_kernel = mod.get_kernel("empty_kernel")

    # Test start
    gb = Device().create_gb().begin_building()
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    graph = gb.end_building().complete()


def test_graph_fork_join(init_cuda):
    # TODO: Maybe share these between tests?
    code = """
    __global__ void empty_kernel() {}
    """
    arch = "".join(f"{i}" for i in Device().compute_capability)
    program_options = ProgramOptions(std="c++17", arch=f"sm_{arch}")
    prog = Program(code, code_type="c++", options=program_options)
    mod = prog.compile("cubin", name_expressions=("empty_kernel",))
    empty_kernel = mod.get_kernel("empty_kernel")

    # Test start
    gb = Device().create_gb().begin_building()
    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)

    left, right = gb.split(2)
    launch(left, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(left, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(right, LaunchConfig(grid=1, block=1), empty_kernel)
    launch(right, LaunchConfig(grid=1, block=1), empty_kernel)
    gb = GraphBuilder.join(left, right)

    launch(gb, LaunchConfig(grid=1, block=1), empty_kernel)
    graph = gb.end_building().complete()
    # gb.debug_dot_print(b"vlad.dot")


# TODO: Test with subgraph
# TODO: Test with conditional
# TODO: Test using graph builder created from device
# TODO: Test using graph builder created from stream
# TODO: Check that the split invalidates the original builder
