# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates scheduling host work on a CUDA stream with
# cuda.core.host_launch(). A kernel updates pinned host memory, a host callback
# observes that intermediate state and updates it on the CPU, and a second
# kernel runs afterward on the same stream.
#
# ################################################################################

# /// script
# dependencies = ["cuda_bindings", "cuda_core", "nvidia-cuda-nvrtc", "numpy>=2.1"]
# ///

import numpy as np

from cuda.core import Device, LaunchConfig, LegacyPinnedMemoryResource, Program, ProgramOptions, host_launch, launch

code = r"""
extern "C" __global__ void increment_index(int* values, unsigned long long index) {
    values[index] += 1;
}
"""


def main():
    dev = Device()
    dev.set_current()
    stream = dev.create_stream()
    pinned_mr = LegacyPinnedMemoryResource()
    pinned_buffer = None

    try:
        program_options = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
        prog = Program(code, code_type="c++", options=program_options)
        mod = prog.compile("cubin")
        kernel = mod.get_kernel("increment_index")

        pinned_buffer = pinned_mr.allocate(2 * np.dtype(np.int32).itemsize)
        values = np.from_dlpack(pinned_buffer).view(np.int32)
        values[:] = 0

        config = LaunchConfig(grid=1, block=1)

        launch(stream, config, kernel, pinned_buffer, np.uint64(0))

        def capture_progress():
            assert values[0] == 1
            assert values[1] == 0
            values[1] += 1

        host_launch(stream, capture_progress)
        launch(stream, config, kernel, pinned_buffer, np.uint64(1))
        stream.sync()

        np.testing.assert_array_equal(values, np.array([1, 2], dtype=np.int32))
    finally:
        if pinned_buffer is not None:
            pinned_buffer.close(stream)
        stream.close()


if __name__ == "__main__":
    main()
