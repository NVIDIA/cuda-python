# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest

from cuda.core.experimental import (
    Device,
    DeviceMemoryResource,
    GraphCompleteOptions,
    GraphMemoryResource,
    launch,
    LaunchConfig,
    Program,
    ProgramOptions,
)
from cuda.core.experimental._utils.cuda_utils import NVRTCError, handle_return
from helpers.buffers import compare_buffer_to_constant, make_scratch_buffer, set_buffer

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


class GraphMemoryTestManager:
    """
    Manages changes to the state of the graph memory system, for testing.
    """
    def __init__(self, gmr, stream, mode=None):
        self.device = Device(gmr.device_id)
        self.gmr = gmr
        self.stream = stream
        self.mode = "relaxed" if mode is None else mode

    def reset(self):
        """Trim unused graph memory and reset usage statistics."""
        self.gmr.trim()
        self.gmr.attributes.reserved_mem_high = 0
        self.gmr.attributes.used_mem_high = 0

    def alloc(self, num, nbytes):
        """Allocate num buffers of size nbytes from graph memory."""
        gb = self.device.create_graph_builder().begin_building(self.mode)
        buffers = [self.gmr.allocate(nbytes, stream=gb.stream) for _ in range(num)]
        graph = gb.end_building().complete()
        graph.upload(self.stream)
        graph.launch(self.stream)
        self.stream.sync()
        return buffers

    def free(self, buffers):
        """Free graph memory buffers."""
        for buffer in buffers:
            buffer.close(stream=self.stream)
        self.stream.sync()


@pytest.mark.parametrize("mode", ["no_graph", "global", "thread_local", "relaxed"])
def test_graph_alloc(init_cuda, mode):
    """Test basic graph capture with memory allocated and deallocated by GraphMemoryResource."""
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

    # Apply kernels, with or without graph capture.
    if mode == "no_graph":
        # Do work without graph capture.
        apply_kernels(mr=dmr, stream=stream, out=out)
    else:
        # Capture work, then upload and launch.
        gb = device.create_graph_builder().begin_building(mode)
        apply_kernels(mr=gmr, stream=gb.stream, out=out)
        graph = gb.end_building().complete()
        graph.upload(stream)
        graph.launch(stream)

    stream.sync()

    # Check the result on the host.
    assert compare_buffer_to_constant(out, 2)


@pytest.mark.parametrize("mode", ["global", "thread_local", "relaxed"])
def test_graph_alloc_with_output(init_cuda, mode):
    """Test for memory allocated in a graph being used outside the graph."""
    NBYTES = 64
    device = Device()
    stream = device.create_stream()
    gmr = GraphMemoryResource(device)

    # Get kernels and define the calling sequence.
    mod = _common_kernels_alloc()
    set_zero = mod.get_kernel("set_zero")
    add_one = mod.get_kernel("add_one")

    # Make an input of 0s.
    in_ = make_scratch_buffer(device, 0, NBYTES)

    # Construct a graph to copy and increment the input. It returns a new
    # buffer allocated within the graph.  The auto_free_on_launch option
    # is required to properly use the output buffer.
    gb = device.create_graph_builder().begin_building(mode)
    out = gmr.allocate(NBYTES, gb.stream)
    out.copy_from(in_, stream=gb.stream)
    launch(gb, LaunchConfig(grid=1, block=1), add_one, out, NBYTES)
    options = GraphCompleteOptions(auto_free_on_launch=True)
    graph = gb.end_building().complete(options)

    # Launch the graph. The output buffer is allocated and set to one.
    graph.upload(stream)
    graph.launch(stream)
    stream.sync()
    assert compare_buffer_to_constant(out, 1)

    # Update the input buffer and rerun the graph.
    set_buffer(in_, 5)
    graph.upload(stream)
    graph.launch(stream)
    stream.sync()
    assert compare_buffer_to_constant(out, 6)


@pytest.mark.parametrize("mode", ["global", "thread_local", "relaxed"])
def test_graph_mem_set_attributes(init_cuda, mode):
    device = Device()
    stream = device.create_stream()
    gmr = GraphMemoryResource(device)
    mman = GraphMemoryTestManager(gmr, stream, mode)

    # Make an allocation and obvserve usage.
    buffer = mman.alloc(1, 1024)
    assert gmr.attributes.reserved_mem_current > 0
    assert gmr.attributes.reserved_mem_high > 0
    assert gmr.attributes.used_mem_current > 0
    assert gmr.attributes.used_mem_high > 0

    # Incorrect attribute usage.
    with pytest.raises(AttributeError,
        match=r"property 'reserved_mem_current' .* has no setter"
    ):
        gmr.attributes.reserved_mem_current = 0

    with pytest.raises(AttributeError,
        match=r"Attribute 'reserved_mem_high' may only be set to zero \(got 1\)\."
    ):
        gmr.attributes.reserved_mem_high = 1

    with pytest.raises(AttributeError,
        match=r"property 'used_mem_current' .* has no setter"
    ):
        gmr.attributes.used_mem_current = 0

    with pytest.raises(AttributeError,
        match=r"Attribute 'used_mem_high' may only be set to zero \(got 1\)\."
    ):
        gmr.attributes.used_mem_high = 1

    # Free memory, but usage is not reduced yet.
    mman.free(buffer)
    assert gmr.attributes.reserved_mem_current > 0
    assert gmr.attributes.reserved_mem_high > 0
    assert gmr.attributes.used_mem_current > 0
    assert gmr.attributes.used_mem_high > 0

    gmr.trim()

    # The high-water marks remain after free and trim.
    assert gmr.attributes.reserved_mem_current == 0
    assert gmr.attributes.reserved_mem_high > 0
    assert gmr.attributes.used_mem_current == 0
    assert gmr.attributes.used_mem_high > 0

    # Reset the high-water marks.
    gmr.attributes.reserved_mem_high = 0
    gmr.attributes.used_mem_high = 0

    assert gmr.attributes.reserved_mem_high == 0
    assert gmr.attributes.used_mem_high == 0

    mman.reset()


@pytest.mark.parametrize("mode", ["global", "thread_local", "relaxed"])
def test_gmr_check_capture_state(init_cuda, mode):
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
   buffer = gmr.allocate(1, stream=gb.stream).close()  # no error
   gb.end_building().complete()


@pytest.mark.parametrize("mode", ["global", "thread_local", "relaxed"])
def test_dmr_check_capture_state(init_cuda, mode):
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

