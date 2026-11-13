# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Explicit 1-D heat diffusion on a small rod, many time steps."""

import sys
import time

import cupy as cp
import numpy as np

from cuda.core import Device, LaunchConfig, Program, launch

CODE = r"""
extern "C" __global__
void diffuse(const float* u, float* u_new, float alpha, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i == 0 || i == n - 1) {
        u_new[i] = u[i];  // fixed-temperature ends
    } else {
        u_new[i] = u[i] + alpha * (u[i - 1] - 2.0f * u[i] + u[i + 1]);
    }
}
"""

N = 4096
STEPS = 200_000
ALPHA = 0.2


def step(stream, kernel, u, u_new):
    config = LaunchConfig(grid=(N + 255) // 256, block=256)
    launch(stream, config, kernel, u.data.ptr, u_new.data.ptr, np.float32(ALPHA), np.int32(N))
    stream.sync()


def main():
    dev = Device()
    dev.set_current()
    stream = dev.create_stream()
    kernel = Program(CODE, code_type="c++").compile("cubin").get_kernel("diffuse")

    u = cp.zeros(N, dtype=cp.float32)
    u[N // 4 : N // 2] = 100.0
    u_new = cp.empty_like(u)
    cp.cuda.get_current_stream().synchronize()

    t0 = time.perf_counter()
    for _ in range(STEPS):
        step(stream, kernel, u, u_new)
        u, u_new = u_new, u
    stream.sync()
    elapsed = time.perf_counter() - t0

    sys.stdout.write(f"steps: {STEPS}  time: {elapsed:.3f} s  ({elapsed / STEPS * 1e6:.2f} us/step)\n")
    sys.stdout.write(f"checksum: {float(u.sum(dtype=cp.float64)):.3f}  peak: {float(u.max()):.4f}\n")


if __name__ == "__main__":
    main()
