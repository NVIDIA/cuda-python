# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Normalize raw sensor readings on the GPU: y = (x - offset) * gain."""

import sys

import numpy as np

from cuda.core import Device, LaunchConfig, LegacyPinnedMemoryResource, Program, launch

CODE = r"""
extern "C" __global__
void normalize(const float* x, float* y, float offset, double gain, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = (float)((x[i] - offset) * gain);
    }
}
"""

N = 1 << 20
OFFSET = 2.5
GAIN = 0.125


def main():
    dev = Device()
    dev.set_current()
    stream = dev.create_stream()
    kernel = Program(CODE, code_type="c++").compile("cubin").get_kernel("normalize")

    nbytes = N * np.dtype(np.float32).itemsize
    pinned = LegacyPinnedMemoryResource()
    h_x = pinned.allocate(nbytes, stream=stream)
    h_y = pinned.allocate(nbytes, stream=stream)
    d_x = dev.allocate(nbytes, stream=stream)
    d_y = dev.allocate(nbytes, stream=stream)

    x = np.from_dlpack(h_x).view(np.float32)
    y = np.from_dlpack(h_y).view(np.float32)
    x[:] = np.linspace(0.0, 10.0, N, dtype=np.float32)

    h_x.copy_to(d_x, stream=stream)
    block = 256
    config = LaunchConfig(grid=N // block, block=block)
    launch(stream, config, kernel, d_x, d_y, OFFSET, np.float32(GAIN), np.int32(N))
    d_y.copy_to(h_y, stream=stream)
    stream.sync()

    sys.stdout.write(f"min={y.min():.4f} mean={y.mean(dtype=np.float64):.4f} max={y.max():.4f}\n")

    for buf in (d_x, d_y, h_x, h_y):
        buf.close(stream)
    stream.close()


if __name__ == "__main__":
    main()
