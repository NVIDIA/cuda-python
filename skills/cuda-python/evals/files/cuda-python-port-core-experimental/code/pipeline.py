# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Link a device-function library into a kernel, run it, and time it (written for cuda-core 0.3)."""

import sys

import numpy as np

from cuda.core.experimental import (
    Device,
    EventOptions,
    LaunchConfig,
    LegacyPinnedMemoryResource,
    Linker,
    LinkerOptions,
    Program,
    ProgramOptions,
    launch,
    system,
)
from cuda.core.experimental.utils import StridedMemoryView

LIBRARY = r"""
extern "C" __device__ float smoothstep(float x)
{
    x = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
    return x * x * (3.0f - 2.0f * x);
}
"""

KERNEL = r"""
extern "C" __device__ float smoothstep(float x);

extern "C" __global__
void ramp(const float* x, float* y, float lo, float hi, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = smoothstep((x[i] - lo) / (hi - lo));
    }
}
"""

N = 1 << 20


def main():
    sys.stdout.write(f"devices: {system.num_devices}\n")
    dev = Device(0)
    dev.set_current()
    stream = dev.create_stream()

    # Compile both translation units to LTO IR and link them into one cubin.
    options = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}", link_time_optimization=True)
    lib_ir = Program(LIBRARY, code_type="c++", options=options).compile("ltoir")
    ker_ir = Program(KERNEL, code_type="c++", options=options).compile("ltoir")
    linker = Linker(lib_ir, ker_ir, options=LinkerOptions(arch=f"sm_{dev.arch}", link_time_optimization=True))
    sys.stdout.write(f"linker backend: {linker.backend}\n")
    kernel = linker.link("cubin").get_kernel("ramp")
    sys.stdout.write(f"kernel arguments: {kernel.num_arguments}\n")
    sys.stdout.write(f"max threads per block: {kernel.attributes.max_threads_per_block()}\n")

    # Pinned host staging buffers and device buffers.
    nbytes = N * 4
    pinned = LegacyPinnedMemoryResource()
    h_x = pinned.allocate(nbytes)
    h_y = pinned.allocate(nbytes)
    d_x = dev.allocate(nbytes)
    d_y = dev.allocate(nbytes)

    x = np.from_dlpack(h_x).view(np.float32)
    y = np.from_dlpack(h_y).view(np.float32)
    x[:] = np.linspace(-1.0, 2.0, N, dtype=np.float32)

    table = np.zeros((4, N // 4), dtype=np.float32)
    view = StridedMemoryView(table, stream_ptr=-1)
    sys.stdout.write(f"host view: shape={view.shape} dtype={view.dtype} on_gpu={view.is_device_accessible}\n")

    h_x.copy_to(d_x, stream=stream)
    config = LaunchConfig(grid=N // 256, block=256, cooperative_launch=False)

    timing = EventOptions(enable_timing=True)
    start = stream.record(options=timing)
    launch(stream, config, kernel, d_x, d_y, np.float32(0.0), np.float32(1.0), np.int32(N))
    end = stream.record(options=timing)
    d_y.copy_to(h_y, stream=stream)
    stream.sync()

    sys.stdout.write(f"timing disabled: {end.is_timing_disabled}\n")
    sys.stdout.write(f"kernel time measured: {end - start > 0.0}\n")
    sys.stdout.write(f"sum={y.sum(dtype=np.float64):.3f} y[N//2]={y[N // 2]:.6f}\n")

    for buf in (d_x, d_y, h_x, h_y):
        buf.close()
    stream.close()


if __name__ == "__main__":
    main()
