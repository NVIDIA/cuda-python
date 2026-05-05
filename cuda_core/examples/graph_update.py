# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates Graph.update() by reusing the same executable graph
# with a new capture that has the same topology but different kernel arguments.
#
# ################################################################################

# /// script
# dependencies = ["cuda_bindings", "cuda_core", "nvidia-cuda-nvrtc", "numpy>=2.1"]
# ///

import sys

import numpy as np

from cuda.core import (
    Device,
    LaunchConfig,
    LegacyPinnedMemoryResource,
    Program,
    ProgramOptions,
    launch,
)

code = """
extern "C" __global__ void add_one(int* value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *value += 1;
    }
}
"""


def build_increment_graph(device, kernel, target_ptr):
    builder = device.create_graph_builder().begin_building()
    config = LaunchConfig(grid=1, block=1)
    launch(builder, config, kernel, target_ptr)
    launch(builder, config, kernel, target_ptr)
    return builder.end_building()


def main():
    if np.lib.NumpyVersion(np.__version__) < "2.1.0":
        print("This example requires NumPy 2.1.0 or later", file=sys.stderr)
        sys.exit(1)

    device = Device()
    device.set_current()
    stream = device.create_stream()
    pinned_mr = LegacyPinnedMemoryResource()
    buffer = None
    initial_capture = None
    update_capture = None
    graph = None

    try:
        options = ProgramOptions(std="c++17", arch=f"sm_{device.arch}")
        program = Program(code, code_type="c++", options=options)
        module = program.compile("cubin")
        kernel = module.get_kernel("add_one")

        buffer = pinned_mr.allocate(2 * np.dtype(np.int32).itemsize)
        values = np.from_dlpack(buffer).view(np.int32)
        values[:] = 0

        initial_capture = build_increment_graph(device, kernel, values[0:].ctypes.data)
        update_capture = build_increment_graph(device, kernel, values[1:].ctypes.data)
        graph = initial_capture.complete()

        graph.upload(stream)
        graph.launch(stream)
        stream.sync()
        assert tuple(values) == (2, 0)

        graph.update(update_capture)
        graph.upload(stream)
        graph.launch(stream)
        stream.sync()
        assert tuple(values) == (2, 2)

        print("Graph.update() reused the executable graph with a new target pointer.")
        print(f"Final host values: {tuple(values)}")
    finally:
        if graph is not None:
            graph.close()
        if update_capture is not None:
            update_capture.close()
        if initial_capture is not None:
            initial_capture.close()
        if buffer is not None:
            buffer.close()
        stream.close()


if __name__ == "__main__":
    main()
