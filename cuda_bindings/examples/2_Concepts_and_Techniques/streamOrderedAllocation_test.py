# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import math
import platform
import random as rnd
import sys

import numpy as np
from common import common
from common.helper_cuda import check_cuda_errors, find_cuda_device
from common.helper_string import check_cmd_line_flag

from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart

stream_ordered_allocation = """\
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


def basic_stream_ordered_allocation(dev, nelem, a, b, c):
    num_bytes = nelem * np.dtype(np.float32).itemsize

    print("Starting basicStreamOrderedAllocation()")
    check_cuda_errors(cudart.cudaSetDevice(dev))
    stream = check_cuda_errors(cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking))

    d_a = check_cuda_errors(cudart.cudaMallocAsync(num_bytes, stream))
    d_b = check_cuda_errors(cudart.cudaMallocAsync(num_bytes, stream))
    d_c = check_cuda_errors(cudart.cudaMallocAsync(num_bytes, stream))
    check_cuda_errors(cudart.cudaMemcpyAsync(d_a, a, num_bytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
    check_cuda_errors(cudart.cudaMemcpyAsync(d_b, b, num_bytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))

    block = cudart.dim3()
    block.x = 256
    block.y = 1
    block.z = 1
    grid = cudart.dim3()
    grid.x = math.ceil(nelem / float(block.x))
    grid.y = 1
    grid.z = 1

    kernel_args = (
        (d_a, d_b, d_c, nelem),
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int),
    )
    check_cuda_errors(
        cuda.cuLaunchKernel(
            _vector_add_gpu,
            grid.x,
            grid.y,
            grid.z,  # grid dim
            block.x,
            block.y,
            block.z,  # block dim
            0,
            stream,  # shared mem and stream
            kernel_args,
            0,
        )
    )  # arguments

    check_cuda_errors(cudart.cudaFreeAsync(d_a, stream))
    check_cuda_errors(cudart.cudaFreeAsync(d_b, stream))
    check_cuda_errors(cudart.cudaMemcpyAsync(c, d_c, num_bytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
    check_cuda_errors(cudart.cudaFreeAsync(d_c, stream))
    check_cuda_errors(cudart.cudaStreamSynchronize(stream))

    # Compare the results
    print("> Checking the results from vectorAddGPU() ...")
    error_norm = 0.0
    ref_norm = 0.0

    for n in range(nelem):
        ref = a[n] + b[n]
        diff = c[n] - ref
        error_norm += diff * diff
        ref_norm += ref * ref

    error_norm = math.sqrt(error_norm)
    ref_norm = math.sqrt(ref_norm)

    check_cuda_errors(cudart.cudaStreamDestroy(stream))

    return error_norm / ref_norm < 1.0e-6


# streamOrderedAllocationPostSync(): demonstrates If the application wants the memory to persist in the pool beyond
# synchronization, then it sets the release threshold on the pool. This way, when the application reaches the "steady state",
# it is no longer allocating/freeing memory from the OS.
def stream_ordered_allocation_post_sync(dev, nelem, a, b, c):
    num_bytes = nelem * np.dtype(np.float32).itemsize

    print("Starting streamOrderedAllocationPostSync()")
    check_cuda_errors(cudart.cudaSetDevice(dev))
    stream = check_cuda_errors(cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking))
    start = check_cuda_errors(cudart.cudaEventCreate())
    end = check_cuda_errors(cudart.cudaEventCreate())

    mem_pool = check_cuda_errors(cudart.cudaDeviceGetDefaultMemPool(dev))
    threshold_val = cuda.cuuint64_t(ctypes.c_uint64(-1).value)
    # Set high release threshold on the default pool so that cudaFreeAsync will not actually release memory to the system.
    # By default, the release threshold for a memory pool is set to zero. This implies that the CUDA driver is
    # allowed to release a memory chunk back to the system as long as it does not contain any active suballocations.
    check_cuda_errors(
        cudart.cudaMemPoolSetAttribute(
            mem_pool,
            cudart.cudaMemPoolAttr.cudaMemPoolAttrReleaseThreshold,
            threshold_val,
        )
    )
    # Record teh start event
    check_cuda_errors(cudart.cudaEventRecord(start, stream))
    for _i in range(MAX_ITER):
        d_a = check_cuda_errors(cudart.cudaMallocAsync(num_bytes, stream))
        d_b = check_cuda_errors(cudart.cudaMallocAsync(num_bytes, stream))
        d_c = check_cuda_errors(cudart.cudaMallocAsync(num_bytes, stream))
        check_cuda_errors(
            cudart.cudaMemcpyAsync(d_a, a, num_bytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        )
        check_cuda_errors(
            cudart.cudaMemcpyAsync(d_b, b, num_bytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        )

        block = cudart.dim3()
        block.x = 256
        block.y = 1
        block.z = 1
        grid = cudart.dim3()
        grid.x = math.ceil(nelem / float(block.x))
        grid.y = 1
        grid.z = 1

        kernel_args = (
            (d_a, d_b, d_c, nelem),
            (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int),
        )
        check_cuda_errors(
            cuda.cuLaunchKernel(
                _vector_add_gpu,
                grid.x,
                grid.y,
                grid.z,  # grid dim
                block.x,
                block.y,
                block.z,  # block dim
                0,
                stream,  # shared mem and stream
                kernel_args,
                0,
            )
        )  # arguments

        check_cuda_errors(cudart.cudaFreeAsync(d_a, stream))
        check_cuda_errors(cudart.cudaFreeAsync(d_b, stream))
        check_cuda_errors(
            cudart.cudaMemcpyAsync(c, d_c, num_bytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        )
        check_cuda_errors(cudart.cudaFreeAsync(d_c, stream))
        check_cuda_errors(cudart.cudaStreamSynchronize(stream))
    check_cuda_errors(cudart.cudaEventRecord(end, stream))
    # Wait for the end event to complete
    check_cuda_errors(cudart.cudaEventSynchronize(end))

    msec_total = check_cuda_errors(cudart.cudaEventElapsedTime(start, end))
    print(f"Total elapsed time = {msec_total} ms over {MAX_ITER} iterations")

    # Compare the results
    print("> Checking the results from vectorAddGPU() ...")
    error_norm = 0.0
    ref_norm = 0.0

    for n in range(nelem):
        ref = a[n] + b[n]
        diff = c[n] - ref
        error_norm += diff * diff
        ref_norm += ref * ref

    error_norm = math.sqrt(error_norm)
    ref_norm = math.sqrt(ref_norm)

    check_cuda_errors(cudart.cudaStreamDestroy(stream))

    return error_norm / ref_norm < 1.0e-6


def main():
    import pytest

    if platform.system() == "Darwin":
        pytest.skip("streamOrderedAllocation is not supported on Mac OSX")

    cuda.cuInit(0)
    if check_cmd_line_flag("help"):
        print("Usage:  streamOrderedAllocation [OPTION]\n", file=sys.stderr)
        print("Options:", file=sys.stderr)
        print("  device=[device #]  Specify the device to be used", file=sys.stderr)
        sys.exit(1)

    dev = find_cuda_device()

    version = check_cuda_errors(cudart.cudaDriverGetVersion())
    if version < 11030:
        is_mem_pool_supported = False
    else:
        is_mem_pool_supported = check_cuda_errors(
            cudart.cudaDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, dev)
        )
    if not is_mem_pool_supported:
        pytest.skip("Waiving execution as device does not support Memory Pools")

    global _vector_add_gpu
    kernel_helper = common.KernelHelper(stream_ordered_allocation, dev)
    _vector_add_gpu = kernel_helper.get_function(b"vectorAddGPU")

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

    ret1 = basic_stream_ordered_allocation(dev, nelem, a, b, c)
    ret2 = stream_ordered_allocation_post_sync(dev, nelem, a, b, c)

    if not ret1 or not ret2:
        sys.exit(1)


if __name__ == "__main__":
    main()
