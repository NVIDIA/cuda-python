# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Apply a per-sensor gain and a bias to the temperature channel of a readings table, in place."""

import sys

import cupy as cp
import numpy as np

from cuda.core import Device, LaunchConfig, Program, launch

CODE = r"""
extern "C" __global__
void calibrate(float* temps, const float* gain, float bias, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        temps[i] = temps[i] * gain[i] + bias;
    }
}
"""

N_SENSORS = 100_000
BIAS = 0.75


def main():
    dev = Device()
    dev.set_current()
    stream = dev.create_stream()
    kernel = Program(CODE, code_type="c++").compile("cubin").get_kernel("calibrate")

    # One row per sensor: (pressure, temperature, humidity).
    readings = (cp.arange(3 * N_SENSORS, dtype=cp.float32) % 50.0).reshape(N_SENSORS, 3)
    gain = cp.linspace(0.5, 1.5, N_SENSORS)
    cp.cuda.get_current_stream().synchronize()

    temps = readings[:, 1]
    block = 256
    config = LaunchConfig(grid=N_SENSORS // block, block=block)
    launch(stream, config, kernel, temps.data.ptr, gain.data.ptr, BIAS, np.int32(N_SENSORS))

    sys.stdout.write(f"temperature mean: {float(temps.mean()):.4f}\n")
    sys.stdout.write(f"temperature last: {float(temps[-1]):.4f}\n")
    sys.stdout.write(f"pressure mean:    {float(readings[:, 0].mean()):.4f}\n")
    sys.stdout.write(f"humidity mean:    {float(readings[:, 2].mean()):.4f}\n")


if __name__ == "__main__":
    main()
