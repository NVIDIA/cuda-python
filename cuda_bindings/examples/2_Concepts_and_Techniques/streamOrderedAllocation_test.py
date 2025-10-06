# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import math
import platform
import random as rnd
import sys

import numpy as np
from common import common
from common.helper_cuda import checkCudaErrors, findCudaDevice
from common.helper_string import checkCmdLineFlag
from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart

streamOrderedAllocation = """\
/* Add two vectors on the GPU */
extern "C"
__global__ void vectorAddGPU(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < N) {
        c[idx] =  a[idx] + b[idx];
    }
}
"""

MAX_ITER = 20


def basicStreamOrderedAllocation(dev, nelem, a, b, c):
    num_bytes = nelem * np.dtype(np.float32).itemsize

    print("Starting basicStreamOrderedAllocation()")
    checkCudaErrors(cudart.cudaSetDevice(dev))
    stream = checkCudaErrors(cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking))

    d_a = checkCudaErrors(cudart.cudaMallocAsync(num_bytes, stream))
    d_b = checkCudaErrors(cudart.cudaMallocAsync(num_bytes, stream))
    d_c = checkCudaErrors(cudart.cudaMallocAsync(num_bytes, stream))
    checkCudaErrors(cudart.cudaMemcpyAsync(d_a, a, num_bytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
    checkCudaErrors(cudart.cudaMemcpyAsync(d_b, b, num_bytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))

    block = cudart.dim3()
    block.x = 256
    block.y = 1
    block.z = 1
    grid = cudart.dim3()
    grid.x = math.ceil(nelem / float(block.x))
    grid.y = 1
    grid.z = 1

    kernelArgs = (
        (d_a, d_b, d_c, nelem),
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int),
    )
    checkCudaErrors(
        cuda.cuLaunchKernel(
            _vectorAddGPU,
            grid.x,
            grid.y,
            grid.z,  # grid dim
            block.x,
            block.y,
            block.z,  # block dim
            0,
            stream,  # shared mem and stream
            kernelArgs,
            0,
        )
    )  # arguments

    checkCudaErrors(cudart.cudaFreeAsync(d_a, stream))
    checkCudaErrors(cudart.cudaFreeAsync(d_b, stream))
    checkCudaErrors(cudart.cudaMemcpyAsync(c, d_c, num_bytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
    checkCudaErrors(cudart.cudaFreeAsync(d_c, stream))
    checkCudaErrors(cudart.cudaStreamSynchronize(stream))

    # Compare the results
    print("> Checking the results from vectorAddGPU() ...")
    errorNorm = 0.0
    refNorm = 0.0

    for n in range(nelem):
        ref = a[n] + b[n]
        diff = c[n] - ref
        errorNorm += diff * diff
        refNorm += ref * ref

    errorNorm = math.sqrt(errorNorm)
    refNorm = math.sqrt(refNorm)

    if errorNorm / refNorm < 1.0e-6:
        print("basicStreamOrderedAllocation PASSED")

    checkCudaErrors(cudart.cudaStreamDestroy(stream))

    return errorNorm / refNorm < 1.0e-6


# streamOrderedAllocationPostSync(): demonstrates If the application wants the memory to persist in the pool beyond
# synchronization, then it sets the release threshold on the pool. This way, when the application reaches the "steady state",
# it is no longer allocating/freeing memory from the OS.
def streamOrderedAllocationPostSync(dev, nelem, a, b, c):
    num_bytes = nelem * np.dtype(np.float32).itemsize

    print("Starting streamOrderedAllocationPostSync()")
    checkCudaErrors(cudart.cudaSetDevice(dev))
    stream = checkCudaErrors(cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking))
    start = checkCudaErrors(cudart.cudaEventCreate())
    end = checkCudaErrors(cudart.cudaEventCreate())

    memPool = checkCudaErrors(cudart.cudaDeviceGetDefaultMemPool(dev))
    thresholdVal = cuda.cuuint64_t(ctypes.c_uint64(-1).value)
    # Set high release threshold on the default pool so that cudaFreeAsync will not actually release memory to the system.
    # By default, the release threshold for a memory pool is set to zero. This implies that the CUDA driver is
    # allowed to release a memory chunk back to the system as long as it does not contain any active suballocations.
    checkCudaErrors(
        cudart.cudaMemPoolSetAttribute(
            memPool,
            cudart.cudaMemPoolAttr.cudaMemPoolAttrReleaseThreshold,
            thresholdVal,
        )
    )
    # Record teh start event
    checkCudaErrors(cudart.cudaEventRecord(start, stream))
    for _i in range(MAX_ITER):
        d_a = checkCudaErrors(cudart.cudaMallocAsync(num_bytes, stream))
        d_b = checkCudaErrors(cudart.cudaMallocAsync(num_bytes, stream))
        d_c = checkCudaErrors(cudart.cudaMallocAsync(num_bytes, stream))
        checkCudaErrors(cudart.cudaMemcpyAsync(d_a, a, num_bytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
        checkCudaErrors(cudart.cudaMemcpyAsync(d_b, b, num_bytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))

        block = cudart.dim3()
        block.x = 256
        block.y = 1
        block.z = 1
        grid = cudart.dim3()
        grid.x = math.ceil(nelem / float(block.x))
        grid.y = 1
        grid.z = 1

        kernelArgs = (
            (d_a, d_b, d_c, nelem),
            (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int),
        )
        checkCudaErrors(
            cuda.cuLaunchKernel(
                _vectorAddGPU,
                grid.x,
                grid.y,
                grid.z,  # grid dim
                block.x,
                block.y,
                block.z,  # block dim
                0,
                stream,  # shared mem and stream
                kernelArgs,
                0,
            )
        )  # arguments

        checkCudaErrors(cudart.cudaFreeAsync(d_a, stream))
        checkCudaErrors(cudart.cudaFreeAsync(d_b, stream))
        checkCudaErrors(cudart.cudaMemcpyAsync(c, d_c, num_bytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
        checkCudaErrors(cudart.cudaFreeAsync(d_c, stream))
        checkCudaErrors(cudart.cudaStreamSynchronize(stream))
    checkCudaErrors(cudart.cudaEventRecord(end, stream))
    # Wait for the end event to complete
    checkCudaErrors(cudart.cudaEventSynchronize(end))

    msecTotal = checkCudaErrors(cudart.cudaEventElapsedTime(start, end))
    print(f"Total elapsed time = {msecTotal} ms over {MAX_ITER} iterations")

    # Compare the results
    print("> Checking the results from vectorAddGPU() ...")
    errorNorm = 0.0
    refNorm = 0.0

    for n in range(nelem):
        ref = a[n] + b[n]
        diff = c[n] - ref
        errorNorm += diff * diff
        refNorm += ref * ref

    errorNorm = math.sqrt(errorNorm)
    refNorm = math.sqrt(refNorm)

    if errorNorm / refNorm < 1.0e-6:
        print("streamOrderedAllocationPostSync PASSED")

    checkCudaErrors(cudart.cudaStreamDestroy(stream))

    return errorNorm / refNorm < 1.0e-6


def main():
    if platform.system() == "Darwin":
        print("streamOrderedAllocation is not supported on Mac OSX - waiving sample")
        return

    cuda.cuInit(0)
    if checkCmdLineFlag("help"):
        print("Usage:  streamOrderedAllocation [OPTION]\n")
        print("Options:")
        print("  device=[device #]  Specify the device to be used")
        return

    dev = findCudaDevice()

    version = checkCudaErrors(cudart.cudaDriverGetVersion())
    if version < 11030:
        isMemPoolSupported = False
    else:
        isMemPoolSupported = checkCudaErrors(
            cudart.cudaDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, dev)
        )
    if not isMemPoolSupported:
        print("Waiving execution as device does not support Memory Pools")
        return

    global _vectorAddGPU
    kernelHelper = common.KernelHelper(streamOrderedAllocation, dev)
    _vectorAddGPU = kernelHelper.getFunction(b"vectorAddGPU")

    # Allocate CPU memory
    nelem = 1048576
    nelem * np.dtype(np.float32).itemsize

    a = np.zeros(nelem, dtype="float32")
    b = np.zeros(nelem, dtype="float32")
    c = np.zeros(nelem, dtype="float32")
    # Initialize the vectors
    for i in range(nelem):
        a[i] = rnd.random()
        b[i] = rnd.random()

    ret1 = basicStreamOrderedAllocation(dev, nelem, a, b, c)
    ret2 = streamOrderedAllocationPostSync(dev, nelem, a, b, c)

    if not ret1 or not ret2:
        sys.exit(-1)


if __name__ == "__main__":
    main()
