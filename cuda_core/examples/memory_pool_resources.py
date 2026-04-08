# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates the newer memory-pool APIs by combining
# PinnedMemoryResource, ManagedMemoryResource, and GraphMemoryResource in one
# workflow.
#
# ################################################################################

# /// script
# dependencies = ["cuda_bindings", "cuda_core", "nvidia-cuda-nvrtc", "numpy>=2.1"]
# ///

import sys

import numpy as np

from cuda.core import (
    Device,
    GraphMemoryResource,
    LaunchConfig,
    ManagedMemoryResource,
    ManagedMemoryResourceOptions,
    PinnedMemoryResource,
    PinnedMemoryResourceOptions,
    Program,
    ProgramOptions,
    launch,
)

code = """
extern "C" __global__ void scale_and_bias(float* data, size_t size, float scale, float bias) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < size; i += stride) {
        data[i] = data[i] * scale + bias;
    }
}
"""


def main():
    if np.lib.NumpyVersion(np.__version__) < "2.1.0":
        print("This example requires NumPy 2.1.0 or later", file=sys.stderr)
        sys.exit(1)

    device = Device()
    device.set_current()
    stream = device.create_stream()

    managed_mr = None
    pinned_mr = None
    graph_mr = None
    managed_buffer = None
    pinned_buffer = None
    graph_capture = None
    graph = None

    try:
        options = ProgramOptions(std="c++17", arch=f"sm_{device.arch}")
        program = Program(code, code_type="c++", options=options)
        module = program.compile("cubin")
        kernel = module.get_kernel("scale_and_bias")

        size = 256
        dtype = np.float32
        nbytes = size * dtype().itemsize
        config = LaunchConfig(grid=(size + 127) // 128, block=128)

        managed_options = ManagedMemoryResourceOptions(
            preferred_location=device.device_id,
            preferred_location_type="device",
        )
        managed_mr = ManagedMemoryResource(options=managed_options)

        pinned_options = {"ipc_enabled": False}
        host_numa_id = getattr(device.properties, "host_numa_id", -1)
        if host_numa_id >= 0:
            pinned_options["numa_id"] = host_numa_id
        pinned_mr = PinnedMemoryResource(options=PinnedMemoryResourceOptions(**pinned_options))

        graph_mr = GraphMemoryResource(device)

        managed_buffer = managed_mr.allocate(nbytes, stream=stream)
        pinned_buffer = pinned_mr.allocate(nbytes, stream=stream)

        managed_array = np.from_dlpack(managed_buffer).view(np.float32)
        pinned_array = np.from_dlpack(pinned_buffer).view(np.float32)

        managed_array[:] = np.arange(size, dtype=dtype)
        managed_original = managed_array.copy()
        stream.sync()

        managed_buffer.copy_to(pinned_buffer, stream=stream)
        stream.sync()
        assert np.array_equal(pinned_array, managed_original)

        graph_builder = device.create_graph_builder().begin_building("relaxed")
        scratch_buffer = graph_mr.allocate(nbytes, stream=graph_builder)
        scratch_buffer.copy_from(managed_buffer, stream=graph_builder)
        launch(graph_builder, config, kernel, scratch_buffer, np.uint64(size), np.float32(2.0), np.float32(1.0))
        managed_buffer.copy_from(scratch_buffer, stream=graph_builder)
        scratch_buffer.close()

        graph_capture = graph_builder.end_building()
        graph = graph_capture.complete()
        graph.upload(stream)
        graph.launch(stream)
        stream.sync()

        np.testing.assert_allclose(managed_array, managed_original * 2 + 1)
        managed_buffer.copy_to(pinned_buffer, stream=stream)
        stream.sync()
        np.testing.assert_allclose(pinned_array, managed_original * 2 + 1)

        print(f"PinnedMemoryResource numa_id: {pinned_mr.numa_id}")
        print(f"ManagedMemoryResource preferred_location: {managed_mr.preferred_location}")
        print(f"GraphMemoryResource reserved high watermark: {graph_mr.attributes.reserved_mem_high}")
    finally:
        if graph is not None:
            graph.close()
        if graph_capture is not None:
            graph_capture.close()
        if pinned_buffer is not None:
            pinned_buffer.close(stream)
        if managed_buffer is not None:
            managed_buffer.close(stream)
        if graph_mr is not None:
            graph_mr.close()
        if pinned_mr is not None:
            pinned_mr.close()
        if managed_mr is not None:
            managed_mr.close()
        stream.close()


if __name__ == "__main__":
    main()
