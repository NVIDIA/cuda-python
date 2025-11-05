# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest

from cuda.core.experimental import (
    Device,
    DeviceMemoryResource,
    GraphMemoryResource,
    LaunchConfig,
    Program,
    ProgramOptions,
    launch,
)
from cuda.core.experimental._utils.cuda_utils import NVRTCError, handle_return
from helpers.buffers import compare_equal_buffers, make_scratch_buffer

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



@pytest.mark.parametrize("repeat", [0,1,2])
@pytest.mark.parametrize("use_graph", [False, True])
def test_graph_alloc(init_cuda, use_graph, repeat):
    """Test graph capture with memory allocated by GraphMemoryResource."""
    NBYTES = 64
    device = Device()
    stream = device.create_stream()
    dmr = DeviceMemoryResource(device)
    gmr = GraphMemoryResource(device)

    out = dmr.allocate(NBYTES, stream=stream)

    # Get kernels and define the calling sequence.
    mod = _common_kernels_alloc()
    set_zero = mod.get_kernel("set_zero")
    add_one = mod.get_kernel("add_one")

    def apply_kernels(mr, stream, out):
        buffer = mr.allocate(NBYTES, stream=stream)
        config = LaunchConfig(grid=1, block=1)
        for kernel in [set_zero, add_one, add_one]:
            launch(stream, config, kernel, buffer, NBYTES)
        out.copy_from(buffer, stream=stream)
        buffer.close()

    # ====== Begin work sequence ======
    if use_graph:
        # Trim memory to zero and reset high watermarks.
        gmr.trim()
        # gmr.attributes.reserved_mem_high = 0  ## not working
        # gmr.attributes.used_mem_high = 0

        assert gmr.attributes.reserved_mem_current == 0
        # assert gmr.attributes.reserved_mem_high == 0
        assert gmr.attributes.used_mem_current == 0
        # assert gmr.attributes.used_mem_high == 0

        # Begin graph capture.
        gb = Device().create_graph_builder().begin_building(mode="thread_local")

        # Capture work.
        apply_kernels(mr=gmr, stream=gb.stream, out=out)

        # Finalize the graph.
        graph = gb.end_building().complete()

        # Upload and launch
        graph.upload(stream)
        graph.launch(stream)
    else:
        # Do work without graph capture.
        apply_kernels(mr=dmr, stream=stream, out=out)

    stream.sync()
    # ====== End work sequence ======

    # Check the result on the host.
    host_ans = make_scratch_buffer(device, 2, NBYTES)
    host_tmp = make_scratch_buffer(device, 0, NBYTES)
    host_tmp.copy_from(out, stream=stream)
    stream.sync()
    assert compare_equal_buffers(host_ans, host_tmp)
    host_ans.close()
    host_tmp.close()

    # # Check memory usage.
    # if use_graph:
    #     assert dmr.attributes.used_mem_current == NBYTES
    #     assert gmr.attributes.used_mem_current > 0
    #     out.close()
    #     assert gmr.attributes.used_mem_current == 0
    # else:
    #     assert dmr.attributes.used_mem_current == NBYTES
    #     assert gmr.attributes.used_mem_current == 0
    #     out.close()
    #     assert dmr.attributes.used_mem_current == 0

