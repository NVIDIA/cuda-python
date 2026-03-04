# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import math
import platform
import random as rnd
import sys

import numpy as np
from common import common
from common.helper_cuda import check_cuda_errors
from common.helper_string import check_cmd_line_flag, get_cmd_line_argument_int

from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart

simple_zero_copy = """\
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
    b_pin_generic_memory = False

    import pytest

    if platform.system() == "Darwin":
        pytest.skip("simpleZeroCopy is not supported on Mac OSX")

    if platform.machine() == "armv7l":
        pytest.skip("simpleZeroCopy is not supported on ARMv7")

    if platform.machine() == "aarch64":
        pytest.skip("simpleZeroCopy is not supported on aarch64")

    if platform.machine() == "sbsa":
        pytest.skip("simpleZeroCopy is not supported on sbsa")

    if check_cmd_line_flag("help"):
        print("Usage:  simpleZeroCopy [OPTION]\n", file=sys.stderr)
        print("Options:", file=sys.stderr)
        print("  device=[device #]  Specify the device to be used", file=sys.stderr)
        print("  use_generic_memory (optional) use generic page-aligned for system memory", file=sys.stderr)
        sys.exit(1)

    # Get the device selected by the user or default to 0, and then set it.
    if check_cmd_line_flag("device="):
        device_count = cudart.cudaGetDeviceCount()
        idev = int(get_cmd_line_argument_int("device="))

        if idev >= device_count or idev < 0:
            print(f"Device number {idev} is invalid, will use default CUDA device 0.")
            idev = 0

    if check_cmd_line_flag("use_generic_memory"):
        b_pin_generic_memory = True

    if b_pin_generic_memory:
        print("> Using Generic System Paged Memory (malloc)")
    else:
        print("> Using CUDA Host Allocated (cudaHostAlloc)")

    check_cuda_errors(cudart.cudaSetDevice(idev))

    # Verify the selected device supports mapped memory and set the device flags for mapping host memory.
    device_prop = check_cuda_errors(cudart.cudaGetDeviceProperties(idev))

    if not device_prop.canMapHostMemory:
        pytest.skip(f"Device {idev} does not support mapping CPU host memory!")

    check_cuda_errors(cudart.cudaSetDeviceFlags(cudart.cudaDeviceMapHost))

    # Allocate mapped CPU memory

    nelem = 1048576
    num_bytes = nelem * np.dtype(np.float32).itemsize

    if b_pin_generic_memory:
        a = np.empty(nelem, dtype=np.float32)
        b = np.empty(nelem, dtype=np.float32)
        c = np.empty(nelem, dtype=np.float32)

        check_cuda_errors(cudart.cudaHostRegister(a, num_bytes, cudart.cudaHostRegisterMapped))
        check_cuda_errors(cudart.cudaHostRegister(b, num_bytes, cudart.cudaHostRegisterMapped))
        check_cuda_errors(cudart.cudaHostRegister(c, num_bytes, cudart.cudaHostRegisterMapped))
    else:
        flags = cudart.cudaHostAllocMapped
        a_ptr = check_cuda_errors(cudart.cudaHostAlloc(num_bytes, flags))
        b_ptr = check_cuda_errors(cudart.cudaHostAlloc(num_bytes, flags))
        c_ptr = check_cuda_errors(cudart.cudaHostAlloc(num_bytes, flags))

        a = (ctypes.c_float * nelem).from_address(a_ptr)
        b = (ctypes.c_float * nelem).from_address(b_ptr)
        c = (ctypes.c_float * nelem).from_address(c_ptr)

    # Initialize the vectors
    for n in range(nelem):
        a[n] = rnd.random()
        b[n] = rnd.random()

    # Get the device pointers for the pinned CPU memory mapped into the GPU memory space
    d_a = check_cuda_errors(cudart.cudaHostGetDevicePointer(a, 0))
    d_b = check_cuda_errors(cudart.cudaHostGetDevicePointer(b, 0))
    d_c = check_cuda_errors(cudart.cudaHostGetDevicePointer(c, 0))

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
    kernel_helper = common.KernelHelper(simple_zero_copy, idev)
    _vector_add_gpu = kernel_helper.get_function(b"vectorAddGPU")
    kernel_args = (
        (d_a, d_b, d_c, nelem),
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int),
    )
    check_cuda_errors(
        cuda.cuLaunchKernel(
            _vector_add_gpu,
            grid.x,
            grid.y,
            grid.z,
            block.x,
            block.y,
            block.z,
            0,
            cuda.CU_STREAM_LEGACY,
            kernel_args,
            0,
        )
    )
    check_cuda_errors(cudart.cudaDeviceSynchronize())

    print("> Checking the results from vectorAddGPU() ...")
    # Compare the results
    error_norm = 0.0
    ref_norm = 0.0

    for n in range(nelem):
        ref = a[n] + b[n]
        diff = c[n] - ref
        error_norm += diff * diff
        ref_norm += ref * ref

    error_norm = math.sqrt(error_norm)
    ref_norm = math.sqrt(ref_norm)

    # Memory clean up

    print("Releasing CPU memory...")

    if b_pin_generic_memory:
        check_cuda_errors(cudart.cudaHostUnregister(a))
        check_cuda_errors(cudart.cudaHostUnregister(b))
        check_cuda_errors(cudart.cudaHostUnregister(c))
    else:
        check_cuda_errors(cudart.cudaFreeHost(a))
        check_cuda_errors(cudart.cudaFreeHost(b))
        check_cuda_errors(cudart.cudaFreeHost(c))

    if error_norm / ref_norm >= 1.0e-7:
        print("FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
