# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import math
import platform
import random as rnd
import sys

import numpy as np
from common import common
from common.helper_cuda import checkCudaErrors
from common.helper_string import checkCmdLineFlag, getCmdLineArgumentInt
from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart

simpleZeroCopy = """\
extern "C"
__global__ void vectorAddGPU(float *a, float *b, float *c, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}
"""


def main():
    idev = 0
    bPinGenericMemory = False

    if platform.system() == "Darwin":
        print("simpleZeroCopy is not supported on Mac OSX - waiving sample")
        return

    if platform.machine() == "armv7l":
        print("simpleZeroCopy is not supported on ARMv7 - waiving sample")
        return

    if platform.machine() == "aarch64":
        print("simpleZeroCopy is not supported on aarch64 - waiving sample")
        return

    if platform.machine() == "sbsa":
        print("simpleZeroCopy is not supported on sbsa - waiving sample")
        return

    if checkCmdLineFlag("help"):
        print("Usage:  simpleZeroCopy [OPTION]\n")
        print("Options:")
        print("  device=[device #]  Specify the device to be used")
        print("  use_generic_memory (optional) use generic page-aligned for system memory")
        return

    # Get the device selected by the user or default to 0, and then set it.
    if checkCmdLineFlag("device="):
        deviceCount = cudart.cudaGetDeviceCount()
        idev = int(getCmdLineArgumentInt("device="))

        if idev >= deviceCount or idev < 0:
            print(f"Device number {idev} is invalid, will use default CUDA device 0.")
            idev = 0

    if checkCmdLineFlag("use_generic_memory"):
        bPinGenericMemory = True

    if bPinGenericMemory:
        print("> Using Generic System Paged Memory (malloc)")
    else:
        print("> Using CUDA Host Allocated (cudaHostAlloc)")

    checkCudaErrors(cudart.cudaSetDevice(idev))

    # Verify the selected device supports mapped memory and set the device flags for mapping host memory.
    deviceProp = checkCudaErrors(cudart.cudaGetDeviceProperties(idev))

    if not deviceProp.canMapHostMemory:
        print(f"Device {idev} does not support mapping CPU host memory!")
        return

    checkCudaErrors(cudart.cudaSetDeviceFlags(cudart.cudaDeviceMapHost))

    # Allocate mapped CPU memory

    nelem = 1048576
    num_bytes = nelem * np.dtype(np.float32).itemsize

    if bPinGenericMemory:
        a = np.empty(nelem, dtype=np.float32)
        b = np.empty(nelem, dtype=np.float32)
        c = np.empty(nelem, dtype=np.float32)

        checkCudaErrors(cudart.cudaHostRegister(a, num_bytes, cudart.cudaHostRegisterMapped))
        checkCudaErrors(cudart.cudaHostRegister(b, num_bytes, cudart.cudaHostRegisterMapped))
        checkCudaErrors(cudart.cudaHostRegister(c, num_bytes, cudart.cudaHostRegisterMapped))
    else:
        flags = cudart.cudaHostAllocMapped
        a_ptr = checkCudaErrors(cudart.cudaHostAlloc(num_bytes, flags))
        b_ptr = checkCudaErrors(cudart.cudaHostAlloc(num_bytes, flags))
        c_ptr = checkCudaErrors(cudart.cudaHostAlloc(num_bytes, flags))

        a = (ctypes.c_float * nelem).from_address(a_ptr)
        b = (ctypes.c_float * nelem).from_address(b_ptr)
        c = (ctypes.c_float * nelem).from_address(c_ptr)

    # Initialize the vectors
    for n in range(nelem):
        a[n] = rnd.random()
        b[n] = rnd.random()

    # Get the device pointers for the pinned CPU memory mapped into the GPU memory space
    d_a = checkCudaErrors(cudart.cudaHostGetDevicePointer(a, 0))
    d_b = checkCudaErrors(cudart.cudaHostGetDevicePointer(b, 0))
    d_c = checkCudaErrors(cudart.cudaHostGetDevicePointer(c, 0))

    # Call the GPU kernel using the CPU pointers residing in CPU mapped memory
    print("> vectorAddGPU kernel will add vectors using mapped CPU memory...")
    block = cudart.dim3()
    block.x = 256
    block.y = 1
    block.z = 1
    grid = cudart.dim3()
    grid.x = math.ceil(nelem / float(block.x))
    grid.y = 1
    grid.z = 1
    kernelHelper = common.KernelHelper(simpleZeroCopy, idev)
    _vectorAddGPU = kernelHelper.getFunction(b"vectorAddGPU")
    kernelArgs = (
        (d_a, d_b, d_c, nelem),
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int),
    )
    checkCudaErrors(
        cuda.cuLaunchKernel(
            _vectorAddGPU,
            grid.x,
            grid.y,
            grid.z,
            block.x,
            block.y,
            block.z,
            0,
            cuda.CU_STREAM_LEGACY,
            kernelArgs,
            0,
        )
    )
    checkCudaErrors(cudart.cudaDeviceSynchronize())

    print("> Checking the results from vectorAddGPU() ...")
    # Compare the results
    errorNorm = 0.0
    refNorm = 0.0

    for n in range(nelem):
        ref = a[n] + b[n]
        diff = c[n] - ref
        errorNorm += diff * diff
        refNorm += ref * ref

    errorNorm = math.sqrt(errorNorm)
    refNorm = math.sqrt(refNorm)

    # Memory clean up

    print("Releasing CPU memory...")

    if bPinGenericMemory:
        checkCudaErrors(cudart.cudaHostUnregister(a))
        checkCudaErrors(cudart.cudaHostUnregister(b))
        checkCudaErrors(cudart.cudaHostUnregister(c))
    else:
        checkCudaErrors(cudart.cudaFreeHost(a))
        checkCudaErrors(cudart.cudaFreeHost(b))
        checkCudaErrors(cudart.cudaFreeHost(c))

    if errorNorm / refNorm >= 1.0e-7:
        print("FAILED")
        sys.exit(-1)
    print("PASSED")


if __name__ == "__main__":
    main()
