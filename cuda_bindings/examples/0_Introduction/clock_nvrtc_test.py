# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import platform

import numpy as np
from common import common
from common.helper_cuda import checkCudaErrors, findCudaDevice
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

NUM_BLOCKS = 64
NUM_THREADS = 256


def elems_to_bytes(nelems, dt):
    return nelems * np.dtype(dt).itemsize


def main():
    print("CUDA Clock sample")

    if platform.machine() == "armv7l":
        print("clock_nvrtc is not supported on ARMv7 - waiving sample")
        return

    timer = np.empty(NUM_BLOCKS * 2, dtype="int64")
    hinput = np.empty(NUM_THREADS * 2, dtype="float32")

    for i in range(0, NUM_THREADS * 2):
        hinput[i] = i

    devID = findCudaDevice()
    kernelHelper = common.KernelHelper(clock_nvrtc, devID)
    kernel_addr = kernelHelper.getFunction(b"timedReduction")

    dinput = checkCudaErrors(cuda.cuMemAlloc(hinput.nbytes))
    doutput = checkCudaErrors(cuda.cuMemAlloc(elems_to_bytes(NUM_BLOCKS, np.float32)))
    dtimer = checkCudaErrors(cuda.cuMemAlloc(timer.nbytes))
    checkCudaErrors(cuda.cuMemcpyHtoD(dinput, hinput, hinput.nbytes))

    args = ((dinput, doutput, dtimer), (None, None, None))
    shared_memory_nbytes = elems_to_bytes(2 * NUM_THREADS, np.float32)

    grid_dims = (NUM_BLOCKS, 1, 1)
    block_dims = (NUM_THREADS, 1, 1)

    checkCudaErrors(
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

    checkCudaErrors(cuda.cuCtxSynchronize())
    checkCudaErrors(cuda.cuMemcpyDtoH(timer, dtimer, timer.nbytes))
    checkCudaErrors(cuda.cuMemFree(dinput))
    checkCudaErrors(cuda.cuMemFree(doutput))
    checkCudaErrors(cuda.cuMemFree(dtimer))

    avgElapsedClocks = 0.0

    for i in range(0, NUM_BLOCKS):
        avgElapsedClocks += timer[i + NUM_BLOCKS] - timer[i]

    avgElapsedClocks = avgElapsedClocks / NUM_BLOCKS
    print(f"Average clocks/block = {avgElapsedClocks}")


if __name__ == "__main__":
    main()
