# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Count the cells of a temperature grid that are above a threshold, on the GPU."""

import ctypes
import sys

import numpy as np

from cuda.bindings import driver, nvrtc

CODE = r"""
__global__ void count_hot(const float* grid, unsigned int* count, float threshold, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && grid[i] > threshold) {
        atomicAdd(count, 1u);
    }
}
"""

N = 1 << 20
THRESHOLD = 75.0


def main():
    driver.cuInit(0)
    _, device = driver.cuDeviceGet(0)
    _, context = driver.cuCtxCreate(None, 0, device)
    _, major = driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device
    )
    _, minor = driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device
    )

    # Compile the kernel to a cubin for this GPU.
    _, program = nvrtc.nvrtcCreateProgram(CODE.encode(), b"hot_cells.cu", 0, [], [])
    options = [f"--gpu-architecture=sm_{major}{minor}".encode()]
    nvrtc.nvrtcCompileProgram(program, len(options), options)
    _, cubin_size = nvrtc.nvrtcGetCUBINSize(program)
    cubin = b" " * cubin_size
    nvrtc.nvrtcGetCUBIN(program, cubin)

    _, module = driver.cuModuleLoadData(np.char.array(cubin).ctypes.data)
    _, kernel = driver.cuModuleGetFunction(module, b"count_hot")

    # Temperatures between 20 and 100 degrees, in a fixed pattern.
    grid = (20.0 + 80.0 * ((np.arange(N, dtype=np.int64) * 7919) % 1000) / 1000.0).astype(np.float32)
    count = np.zeros(1, dtype=np.uint32)

    _, d_grid = driver.cuMemAlloc(grid.nbytes)
    _, d_count = driver.cuMemAlloc(count.nbytes)
    driver.cuMemcpyHtoD(d_grid, grid.ctypes.data, grid.nbytes)
    driver.cuMemcpyHtoD(d_count, count.ctypes.data, count.nbytes)

    block = 256
    blocks = (N + block - 1) // block
    args = (
        (int(d_grid), int(d_count), THRESHOLD, N),
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int),
    )
    driver.cuLaunchKernel(kernel, blocks, 1, 1, block, 1, 1, 0, 0, args, 0)
    driver.cuCtxSynchronize()
    driver.cuMemcpyDtoH(count.ctypes.data, d_count, count.nbytes)

    sys.stdout.write(f"hot cells: {int(count[0])} of {N}\n")

    driver.cuMemFree(d_grid)
    driver.cuMemFree(d_count)
    driver.cuModuleUnload(module)
    driver.cuCtxDestroy(context)


if __name__ == "__main__":
    main()
