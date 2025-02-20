# Copyright 2025 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.


from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch

# from cuda.core.experimental import Device, Stream, StreamOptions
# from cuda.core.experimental._event import Event
# from cuda.core.experimental._stream import LEGACY_DEFAULT_STREAM, PER_THREAD_DEFAULT_STREAM, default_stream


def test_graph_is_capture_alive(init_cuda):
    graph_builder = Device().build_graph()
    assert graph_builder.is_capture_active() is False
    graph_builder.begin_capture()
    assert graph_builder.is_capture_active() is True
    graph_builder.end_capture()
    assert graph_builder.is_capture_active() is False


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
    graph_builder = Device().build_graph()
    config = LaunchConfig(grid=1, block=1, stream=graph_builder.stream)

    assert graph_builder.is_capture_active() is False
    graph_builder.begin_capture()
    launch(empty_kernel, {"grid": 1, "block": 1, "stream": graph_builder.stream})
    launch(empty_kernel, {"grid": 1, "block": 1, "stream": graph_builder.stream})
    launch(empty_kernel, {"grid": 1, "block": 1, "stream": graph_builder.stream})
    graph_builder.end_capture()


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
    graph_builder = Device().build_graph()
    assert graph_builder.is_capture_active() is False
    graph_builder.begin_capture()
    launch(empty_kernel, {"grid": 1, "block": 1, "stream": graph_builder.stream})

    left, right = graph_builder.fork(2)
    launch(empty_kernel, {"grid": 1, "block": 1, "stream": left.stream})
    launch(empty_kernel, {"grid": 1, "block": 1, "stream": left.stream})
    launch(empty_kernel, {"grid": 1, "block": 1, "stream": right.stream})
    launch(empty_kernel, {"grid": 1, "block": 1, "stream": right.stream})
    graph_builder.join(left, right)

    launch(empty_kernel, {"grid": 1, "block": 1, "stream": graph_builder.stream})
    graph_builder.end_capture()

    graph_builder.debug_dot_print(b"vlad.dot")
