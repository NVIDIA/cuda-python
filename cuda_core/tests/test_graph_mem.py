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
from helpers.buffers import compare_buffer_to_constant

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


@pytest.mark.parametrize("repeat", range(3))
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
        # import code
        # code.interact(local=dict(globals(), **locals()))

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
    assert compare_buffer_to_constant(out, 2)

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


@pytest.mark.parametrize("mode", ["global", "thread_local", "relaxed"])
def test_gmr_check_capture(init_cuda, mode):
   """
   Test expected errors (and non-errors) using GraphMemoryResource with graph
   capture.
   """
   device = Device()
   stream = device.create_stream()
   gmr = GraphMemoryResource(device)

   # Not capturing
   with pytest.raises(RuntimeError,
       match=r"GraphMemoryResource cannot perform memory operations on a "
             r"non-capturing stream\."
   ):
       gmr.allocate(1, stream=stream)

   # Capturing
   gb = device.create_graph_builder().begin_building(mode=mode)
   gmr.allocate(1, stream=gb.stream).close()  # no error
   gb.end_building().complete()


@pytest.mark.parametrize("mode", ["global", "thread_local", "relaxed"])
def test_mr_check_capture(init_cuda, mode):
   """
   Test expected errors (and non-errors) using DeviceMemoryResource with graph
   capture.
   """
   device = Device()
   stream = device.create_stream()
   dmr = DeviceMemoryResource(device)

   # Not capturing
   dmr.allocate(1, stream=stream).close()  # no error

   # Capturing
   gb = device.create_graph_builder().begin_building(mode=mode)
   with pytest.raises(RuntimeError,
       match=r"DeviceMemoryResource cannot perform memory operations on a capturing "
             r"stream \(consider using GraphMemoryResource\)\."
   ):
       dmr.allocate(1, stream=gb.stream)
   gb.end_building().complete()


# This tests causes unraisable errors at shutdown.
# @pytest.mark.parametrize("mode", ["global", "thread_local"])
# def test_cross_stream_capture_error(init_cuda, mode):
#    """
#    Test errors related to unsafe API calls in global or thread_local capture
#    mode.
#    """
#    # When graph capturing is turned on for an unrelated stream, the driver
#    # raises an error. Not sure how to detect this.
#    from cuda.core.experimental._utils.cuda_utils import CUDAError  # FIXME
#    device = Device()
#    stream = device.create_stream()
#    dmr = DeviceMemoryResource(device)
#
#    with pytest.raises(RuntimeError, match="Build process encountered an error"):
#        gb = device.create_graph_builder().begin_building(mode)
#        with pytest.raises(CUDAError, match="CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED"):
#            dmr.allocate(1, stream=stream)  # not targeting gb.stream
#        gb.end_building().complete()

