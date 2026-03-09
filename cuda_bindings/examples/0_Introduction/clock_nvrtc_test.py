# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import platform

import numpy as np
from common import common
from common.helper_cuda import check_cuda_errors, find_cuda_device

from cuda.bindings import driver as cuda

clock_nvrtc = """\
extern "C" __global__  void timedReduction(const float *hinput, float *output, clock_t *timer)
{
    // __shared__ float shared[2 * blockDim.x];
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (tid == 0) timer[bid] = clock();

    // Copy hinput.
    shared[tid] = hinput[tid];
    shared[tid + blockDim.x] = hinput[tid + blockDim.x];

    // Perform reduction to find minimum.
    for (int d = blockDim.x; d > 0; d /= 2)
    {
        __syncthreads();

        if (tid < d)
        {
            float f0 = shared[tid];
            float f1 = shared[tid + d];

            if (f1 < f0)
            {
                shared[tid] = f1;
            }
        }
    }

    // Write result.
    if (tid == 0) output[bid] = shared[0];

    __syncthreads();

    if (tid == 0) timer[bid+gridDim.x] = clock();
}
"""

num_blocks = 64
num_threads = 256


def elems_to_bytes(nelems, dt):
    return nelems * np.dtype(dt).itemsize


def main():
    import pytest

    if platform.machine() == "armv7l":
        pytest.skip("clock_nvrtc is not supported on ARMv7")

    timer = np.empty(num_blocks * 2, dtype="int64")
    hinput = np.empty(num_threads * 2, dtype="float32")

    for i in range(num_threads * 2):
        hinput[i] = i

    dev_id = find_cuda_device()
    kernel_helper = common.KernelHelper(clock_nvrtc, dev_id)
    kernel_addr = kernel_helper.get_function(b"timedReduction")

    dinput = check_cuda_errors(cuda.cuMemAlloc(hinput.nbytes))
    doutput = check_cuda_errors(cuda.cuMemAlloc(elems_to_bytes(num_blocks, np.float32)))
    dtimer = check_cuda_errors(cuda.cuMemAlloc(timer.nbytes))
    check_cuda_errors(cuda.cuMemcpyHtoD(dinput, hinput, hinput.nbytes))

    args = ((dinput, doutput, dtimer), (None, None, None))
    shared_memory_nbytes = elems_to_bytes(2 * num_threads, np.float32)

    grid_dims = (num_blocks, 1, 1)
    block_dims = (num_threads, 1, 1)

    check_cuda_errors(
        cuda.cuLaunchKernel(
            kernel_addr,
            *grid_dims,  # grid dim
            *block_dims,  # block dim
            shared_memory_nbytes,
            0,  # shared mem, stream
            args,
            0,
        )
    )  # arguments

    check_cuda_errors(cuda.cuCtxSynchronize())
    check_cuda_errors(cuda.cuMemcpyDtoH(timer, dtimer, timer.nbytes))
    check_cuda_errors(cuda.cuMemFree(dinput))
    check_cuda_errors(cuda.cuMemFree(doutput))
    check_cuda_errors(cuda.cuMemFree(dtimer))

    avg_elapsed_clocks = 0.0

    for i in range(num_blocks):
        avg_elapsed_clocks += timer[i + num_blocks] - timer[i]

    avg_elapsed_clocks = avg_elapsed_clocks / num_blocks
    print(f"Average clocks/block = {avg_elapsed_clocks}")


if __name__ == "__main__":
    main()
