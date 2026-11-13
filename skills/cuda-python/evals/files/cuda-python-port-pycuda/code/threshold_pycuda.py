# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Soft-threshold a noisy signal on the GPU and time the kernel (PyCUDA version)."""

import sys

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as drv
from pycuda.compiler import SourceModule

mod = SourceModule(
    """
__global__ void soft_threshold(float *out, const float *in, float lam, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in[i];
        float mag = fabsf(v) - lam;
        out[i] = mag > 0.0f ? copysignf(mag, v) : 0.0f;
    }
}
"""
)
soft_threshold = mod.get_function("soft_threshold")

n = 1 << 22
lam = 0.25
t = np.arange(n, dtype=np.float32)
signal = (np.sin(t * 1e-3) + 0.3 * np.cos(t * 0.37)).astype(np.float32)
out = np.zeros_like(signal)

# Page-locked host memory for fast transfers, explicit device allocations.
h_in = drv.pagelocked_empty(n, dtype=np.float32)
h_out = drv.pagelocked_empty(n, dtype=np.float32)
h_in[:] = signal
d_in = drv.mem_alloc(h_in.nbytes)
d_out = drv.mem_alloc(h_out.nbytes)

stream = drv.Stream()
start, end = drv.Event(), drv.Event()

drv.memcpy_htod_async(d_in, h_in, stream)
start.record(stream)
soft_threshold(d_out, d_in, np.float32(lam), np.int32(n), block=(256, 1, 1), grid=((n + 255) // 256, 1), stream=stream)
end.record(stream)
drv.memcpy_dtoh_async(h_out, d_out, stream)
stream.synchronize()
out[:] = h_out

sys.stdout.write(f"kernel time measured: {end.time_since(start) > 0.0}\n")
sys.stdout.write(f"nonzero: {int(np.count_nonzero(out))} of {n}\n")
sys.stdout.write(f"sum of |out|: {np.abs(out).sum(dtype=np.float64):.3f}\n")
sys.stdout.write(f"out[1000:1004]: {[round(float(v), 5) for v in out[1000:1004]]}\n")

d_in.free()
d_out.free()
