# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure the error of storing float32 weights as half precision, per layer."""

import sys

import cupy as cp
import numpy as np

from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch

CUDA_ROOT = "/usr/local/cuda-12.4"
INCLUDE_DIRS = [f"{CUDA_ROOT}/include"]

CODE = r"""
#include <cuda_fp16.h>
#include <cuda/std/array>

constexpr cuda::std::array<float, 4> layer_scale = {1.0f, 8.0f, 64.0f, 512.0f};

extern "C" __global__
void half_error(const float* w, float* err, int layer, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = w[i] * layer_scale[layer];
        err[i] = fabsf(__half2float(__float2half(v)) - v);
    }
}
"""

N = 1 << 20


def main():
    dev = Device()
    dev.set_current()
    stream = dev.create_stream()

    options = ProgramOptions(std="c++17", include_path=INCLUDE_DIRS)
    kernel = Program(CODE, code_type="c++", options=options).compile("cubin").get_kernel("half_error")

    w = cp.linspace(-1.0, 1.0, N, dtype=cp.float32)
    err = cp.empty_like(w)
    cp.cuda.get_current_stream().synchronize()

    block = 256
    config = LaunchConfig(grid=(N + block - 1) // block, block=block)
    for layer in range(4):
        launch(stream, config, kernel, w.data.ptr, err.data.ptr, np.int32(layer), np.int32(N))
        stream.sync()
        sys.stdout.write(f"layer {layer}: max half-precision error = {float(err.max()):.6f}\n")


if __name__ == "__main__":
    main()
