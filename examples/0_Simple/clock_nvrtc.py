# Copyright 2021 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import numpy as np
from cuda import cuda
from examples.common import common
from examples.common.helper_cuda import checkCudaErrors, findCudaDevice

clock_nvrtc = '''\
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
'''

NUM_BLOCKS = 64 
NUM_THREADS  = 256

def main():
    print("CUDA Clock sample")

    timer = np.empty(NUM_BLOCKS * 2, dtype='int64')
    hinput = np.empty(NUM_THREADS * 2, dtype='float32')

    for i in range(0, NUM_THREADS * 2):
        hinput[i] = i

    devID = findCudaDevice()
    kernelHelper = common.KernelHelper(clock_nvrtc, devID)
    kernel_addr = kernelHelper.getFunction(b'timedReduction')

    dinput = checkCudaErrors(cuda.cuMemAlloc(np.dtype(np.float32).itemsize * NUM_THREADS * 2))
    doutput = checkCudaErrors(cuda.cuMemAlloc(np.dtype(np.float32).itemsize * NUM_BLOCKS))
    dtimer = checkCudaErrors(cuda.cuMemAlloc(np.dtype(np.int64).itemsize * NUM_BLOCKS * 2))
    checkCudaErrors(cuda.cuMemcpyHtoD(dinput, hinput, np.dtype(np.float32).itemsize * NUM_THREADS * 2))



    arr = ((dinput, doutput, dtimer),
           (None, None, None))

    checkCudaErrors(cuda.cuLaunchKernel(kernel_addr,
                                        NUM_BLOCKS, 1, 1,  # grid dim
                                        NUM_THREADS, 1, 1, # block dim
                                        np.dtype(np.float32).itemsize * 2 *NUM_THREADS, cuda.CUstream(0), # shared mem, stream
                                        arr, 0)) # arguments

    checkCudaErrors(cuda.cuCtxSynchronize())
    checkCudaErrors(cuda.cuMemcpyDtoH(timer, dtimer, np.dtype(np.int64).itemsize * NUM_BLOCKS * 2))
    checkCudaErrors(cuda.cuMemFree(dinput))
    checkCudaErrors(cuda.cuMemFree(doutput))
    checkCudaErrors(cuda.cuMemFree(dtimer))

    avgElapsedClocks = 0.0

    for i in range(0,NUM_BLOCKS):
        avgElapsedClocks += timer[i + NUM_BLOCKS] - timer[i]

    avgElapsedClocks = avgElapsedClocks/NUM_BLOCKS;
    print("Average clocks/block = {}".format(avgElapsedClocks))

if __name__=="__main__":
    main()
