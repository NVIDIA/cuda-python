# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Device-side graph launch tests.

Device-side graph launch allows a kernel running on the GPU to launch a CUDA graph.
This feature requires:
- CUDA 12.0+
- Hopper architecture (sm_90+)
- The kernel calling cudaGraphLaunch() must itself be launched from within a graph
"""

import numpy as np
import pytest
from cuda.core import (
    Device,
    GraphCompleteOptions,
    LaunchConfig,
    LegacyPinnedMemoryResource,
    Linker,
    LinkerOptions,
    ObjectCode,
    Program,
    ProgramOptions,
    launch,
)


def _get_device_arch():
    """Get the current device's architecture string."""
    return "".join(f"{i}" for i in Device().compute_capability)


def _compile_work_kernel():
    """Compile a simple kernel that increments a value."""
    code = """
    extern "C" __global__ void increment(int* value) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            atomicAdd(value, 1);
        }
    }
    """
    arch = _get_device_arch()
    opts = ProgramOptions(std="c++17", arch=f"sm_{arch}")
    return Program(code, "c++", options=opts).compile("cubin").get_kernel("increment")


def _compile_device_launcher_kernel():
    """Compile a kernel that launches a graph from device code.

    This kernel uses cudaGraphLaunch() to launch a graph from device code.
    It requires linking with libcudadevrt.a (the CUDA device runtime library).

    Raises pytest.skip if libcudadevrt.a cannot be found.
    """
    pathfinder = pytest.importorskip("cuda.pathfinder")
    try:
        cudadevrt_path = pathfinder.find_static_lib("cudadevrt")
    except pathfinder.StaticLibNotFoundError as e:
        pytest.skip(f"cudadevrt library not found: {e}")

    code = """
    extern "C" __global__ void launch_graph_from_device(cudaGraphExec_t graph) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            cudaGraphLaunch(graph, cudaStreamGraphTailLaunch);
        }
    }
    """
    arch = _get_device_arch()
    opts = ProgramOptions(std="c++17", arch=f"sm_{arch}", relocatable_device_code=True)
    ptx = Program(code, "c++", options=opts).compile("ptx")

    # Link with device runtime library
    cudadevrt = ObjectCode.from_library(cudadevrt_path)

    linker = Linker(ptx, cudadevrt, options=LinkerOptions(arch=f"sm_{arch}"))
    return linker.link("cubin").get_kernel("launch_graph_from_device")


@pytest.mark.skipif(
    Device().compute_capability.major < 9,
    reason="Device-side graph launch requires Hopper (sm_90+) architecture",
)
@pytest.mark.skipif(tuple(int(i) for i in np.__version__.split(".")[:2]) < (2, 1), reason="need numpy 2.1.0+")
def test_device_launch_basic(init_cuda):
    """Test basic device-side graph launch functionality.

    This test verifies that a graph can be launched from device code by:
    1. Creating an inner graph (with device_launch=True) that increments a value
    2. Creating an outer graph that contains a kernel calling cudaGraphLaunch()
    3. Launching the outer graph and verifying the inner graph executed
    """
    dev = Device()
    dev.set_current()
    stream = dev.create_stream()

    # Compile kernels
    work_kernel = _compile_work_kernel()
    launcher_kernel = _compile_device_launcher_kernel()

    # Allocate and initialize memory
    mr = LegacyPinnedMemoryResource()
    buf = mr.allocate(4, stream=stream)
    arr = np.from_dlpack(buf).view(np.int32)
    arr[0] = 0
    stream.sync()

    # Create the inner graph (the graph to be launched from device)
    gb_inner = dev.create_graph_builder().begin_building()
    launch(gb_inner, LaunchConfig(grid=1, block=1), work_kernel, arr.ctypes.data)
    inner_graph = gb_inner.end_building().complete(
        options=GraphCompleteOptions(device_launch=True, upload_stream=stream)
    )
    stream.sync()

    # Create the outer graph (launches inner graph from device)
    inner_graph_handle = int(inner_graph.handle)
    gb_outer = dev.create_graph_builder().begin_building()
    launch(gb_outer, LaunchConfig(grid=1, block=1), launcher_kernel, inner_graph_handle)
    outer_graph = gb_outer.end_building().complete()

    # Launch outer graph (which triggers device-side launch of inner graph)
    outer_graph.launch(stream)
    stream.sync()

    # Verify result
    assert arr[0] == 1, f"Expected 1, got {arr[0]}"

    buf.close()


@pytest.mark.skipif(
    Device().compute_capability.major < 9,
    reason="Device-side graph launch requires Hopper (sm_90+) architecture",
)
@pytest.mark.skipif(tuple(int(i) for i in np.__version__.split(".")[:2]) < (2, 1), reason="need numpy 2.1.0+")
def test_device_launch_multiple(init_cuda):
    """Test that device-side graph launch can be executed multiple times.

    This test verifies that both the outer and inner graphs can be reused
    for multiple launches.
    """
    dev = Device()
    dev.set_current()
    stream = dev.create_stream()

    # Compile kernels
    work_kernel = _compile_work_kernel()
    launcher_kernel = _compile_device_launcher_kernel()

    # Allocate and initialize memory
    mr = LegacyPinnedMemoryResource()
    buf = mr.allocate(4, stream=stream)
    arr = np.from_dlpack(buf).view(np.int32)
    arr[0] = 0
    stream.sync()

    # Create the inner graph
    gb_inner = dev.create_graph_builder().begin_building()
    launch(gb_inner, LaunchConfig(grid=1, block=1), work_kernel, arr.ctypes.data)
    inner_graph = gb_inner.end_building().complete(
        options=GraphCompleteOptions(device_launch=True, upload_stream=stream)
    )
    stream.sync()

    # Create the outer graph
    inner_graph_handle = int(inner_graph.handle)
    gb_outer = dev.create_graph_builder().begin_building()
    launch(gb_outer, LaunchConfig(grid=1, block=1), launcher_kernel, inner_graph_handle)
    outer_graph = gb_outer.end_building().complete()

    # Launch multiple times
    num_launches = 5
    for _ in range(num_launches):
        outer_graph.launch(stream)
    stream.sync()

    # Verify result
    assert arr[0] == num_launches, f"Expected {num_launches}, got {arr[0]}"

    buf.close()
