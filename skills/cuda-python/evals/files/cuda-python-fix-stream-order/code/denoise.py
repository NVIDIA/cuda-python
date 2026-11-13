# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Denoise frames: CuPy pre-processing, a custom CUDA kernel, CuPy post-processing, all asynchronous."""

import sys

import cupy as cp
import numpy as np

from cuda.core import Device, LaunchConfig, Program, launch

CODE = r"""
extern "C" __global__
void clamp_outliers(float* x, float limit, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = x[i] > limit ? limit : (x[i] < -limit ? -limit : x[i]);
    }
}
"""

N = 1 << 25
FRAMES = 8
LIMIT = 1.5


def main():
    dev = Device()
    dev.set_current()
    kernel = Program(CODE, code_type="c++").compile("cubin").get_kernel("clamp_outliers")
    config = LaunchConfig(grid=(N + 255) // 256, block=256)

    cupy_stream = cp.cuda.Stream(non_blocking=True)
    kernel_stream = dev.create_stream()

    totals = cp.zeros(FRAMES, dtype=cp.float64)
    with cupy_stream:
        ramp = cp.arange(N, dtype=cp.float32)
        for frame in range(FRAMES):
            # Pre-process with CuPy: a synthetic noisy frame, sorted.
            x = cp.sin(ramp * (0.001 * (frame + 1))) * 2.0
            x = cp.sort(x)
            # Custom kernel, in place.
            launch(kernel_stream, config, kernel, x.data.ptr, np.float32(LIMIT), np.int32(N))
            # Post-process with CuPy: accumulate on the device, no host round trip per frame.
            totals[frame] = cp.abs(x).sum(dtype=cp.float64)

    cupy_stream.synchronize()
    kernel_stream.sync()
    for frame, total in enumerate(totals.get()):
        sys.stdout.write(f"frame {frame}: sum|x| = {total:.1f}\n")


if __name__ == "__main__":
    main()
