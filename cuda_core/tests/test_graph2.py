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


# def test_no_graph(init_cuda):
#     device = Device()
#     stream = device.create_stream()
# 
#     # Get kernels.
#     mod = _common_kernels()
#     set_zero = mod.get_kernel("set_zero")
#     add_one = mod.get_kernel("add_one")
# 
#     # Run operations.
#     NBYTES = 1
#     mr = DeviceMemoryResource(device)
#     work_buffer = mr.allocate(NBYTES, stream=stream)
#     launch(stream, LaunchConfig(grid=1, block=1), set_zero, int(work_buffer.handle))
#     launch(stream, LaunchConfig(grid=1, block=1), add_one, int(work_buffer.handle))
# 
#     # Check the result.
#     one = make_scratch_buffer(device, 1, NBYTES)
#     compare_buffer = make_scratch_buffer(device, 0, NBYTES)
#     compare_buffer.copy_from(work_buffer, stream=stream)
#     stream.sync()
#     assert compare_equal_buffers(one, compare_buffer)

    # # Let's have a look.
    # # options = GraphDebugPrintOptions(**{field: True for field in GraphDebugPrintOptions.__dataclass_fields__})
    # # gb.debug_dot_print(b"./debug.dot", options)


def test_graph(init_cuda):
    device = Device()
    stream = device.create_stream()
    options = DeviceMemoryResourceOptions(use_pool=False)
    mr = DeviceMemoryResource(device, options=options)

    # Get kernels.
    mod = _common_kernels()
    set_zero = mod.get_kernel("set_zero")
    add_one = mod.get_kernel("add_one")

    NBYTES = 64
    target = mr.allocate(NBYTES, stream=stream)

    # Begin graph capture.
    gb = Device().create_graph_builder().begin_building(mode="thread_local")

    # import code
    # code.interact(local=dict(globals(), **locals()))
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

