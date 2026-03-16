# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import platform
import sys

import numpy as np
from common import common
from common.helper_cuda import check_cuda_errors

from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart

simplep2p = """\
extern "C"
__global__ void SimpleKernel(float *src, float *dst)
{
    // Just a dummy kernel, doing enough for us to verify that everything
    // worked
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx] * 2.0f;
}
"""


def main():
    import pytest

    if platform.system() == "Darwin":
        pytest.skip("simpleP2P is not supported on Mac OSX")

    if platform.machine() == "armv7l":
        pytest.skip("simpleP2P is not supported on ARMv7")

    if platform.machine() == "aarch64":
        pytest.skip("simpleP2P is not supported on aarch64")

    if platform.machine() == "sbsa":
        pytest.skip("simpleP2P is not supported on sbsa")

    # Number of GPUs
    print("Checking for multiple GPUs...")
    gpu_n = check_cuda_errors(cudart.cudaGetDeviceCount())
    print(f"CUDA-capable device count: {gpu_n}")

    if gpu_n < 2:
        pytest.skip("Two or more GPUs with Peer-to-Peer access capability are required")

    prop = [check_cuda_errors(cudart.cudaGetDeviceProperties(i)) for i in range(gpu_n)]
    # Check possibility for peer access
    print("\nChecking GPU(s) for support of peer to peer memory access...")

    p2p_capable_gp_us = [-1, -1]
    for i in range(gpu_n):
        p2p_capable_gp_us[0] = i
        for j in range(gpu_n):
            if i == j:
                continue
            i_access_j = check_cuda_errors(cudart.cudaDeviceCanAccessPeer(i, j))
            j_access_i = check_cuda_errors(cudart.cudaDeviceCanAccessPeer(j, i))
            print(
                "> Peer access from {} (GPU{}) -> {} (GPU{}) : {}\n".format(
                    prop[i].name, i, prop[j].name, j, "Yes" if i_access_j else "No"
                )
            )
            print(
                "> Peer access from {} (GPU{}) -> {} (GPU{}) : {}\n".format(
                    prop[j].name, j, prop[i].name, i, "Yes" if i_access_j else "No"
                )
            )
            if i_access_j and j_access_i:
                p2p_capable_gp_us[1] = j
                break
        if p2p_capable_gp_us[1] != -1:
            break

    if p2p_capable_gp_us[0] == -1 or p2p_capable_gp_us[1] == -1:
        pytest.skip("Peer to Peer access is not available amongst GPUs in the system")

    # Use first pair of p2p capable GPUs detected
    gpuid = [p2p_capable_gp_us[0], p2p_capable_gp_us[1]]

    # Enable peer access
    print(f"Enabling peer access between GPU{gpuid[0]} and GPU{gpuid[1]}...")
    check_cuda_errors(cudart.cudaSetDevice(gpuid[0]))
    check_cuda_errors(cudart.cudaDeviceEnablePeerAccess(gpuid[1], 0))
    check_cuda_errors(cudart.cudaSetDevice(gpuid[1]))
    check_cuda_errors(cudart.cudaDeviceEnablePeerAccess(gpuid[0], 0))

    # Allocate buffers
    buf_size = 1024 * 1024 * 16 * np.dtype(np.float32).itemsize
    print(f"Allocating buffers ({int(buf_size / 1024 / 1024)}MB on GPU{gpuid[0]}, GPU{gpuid[1]} and CPU Host)...")
    check_cuda_errors(cudart.cudaSetDevice(gpuid[0]))
    g0 = check_cuda_errors(cudart.cudaMalloc(buf_size))
    check_cuda_errors(cudart.cudaSetDevice(gpuid[1]))
    g1 = check_cuda_errors(cudart.cudaMalloc(buf_size))
    h0 = check_cuda_errors(cudart.cudaMallocHost(buf_size))  # Automatically portable with UVA

    # Create CUDA event handles
    print("Creating event handles...")
    eventflags = cudart.cudaEventBlockingSync
    start_event = check_cuda_errors(cudart.cudaEventCreateWithFlags(eventflags))
    stop_event = check_cuda_errors(cudart.cudaEventCreateWithFlags(eventflags))

    # P2P memcopy() benchmark
    check_cuda_errors(cudart.cudaEventRecord(start_event, cudart.cudaStream_t(0)))

    for i in range(100):
        # With UVA we don't need to specify source and target devices, the
        # runtime figures this out by itself from the pointers
        # Ping-pong copy between GPUs
        if i % 2 == 0:
            check_cuda_errors(cudart.cudaMemcpy(g1, g0, buf_size, cudart.cudaMemcpyKind.cudaMemcpyDefault))
        else:
            check_cuda_errors(cudart.cudaMemcpy(g0, g1, buf_size, cudart.cudaMemcpyKind.cudaMemcpyDefault))

    check_cuda_errors(cudart.cudaEventRecord(stop_event, cudart.cudaStream_t(0)))
    check_cuda_errors(cudart.cudaEventSynchronize(stop_event))
    time_memcpy = check_cuda_errors(cudart.cudaEventElapsedTime(start_event, stop_event))
    print(
        f"cudaMemcpyPeer / cudaMemcpy between GPU{gpuid[0]} and GPU{gpuid[1]}: {(1.0 / (time_memcpy / 1000.0)) * (100.0 * buf_size) / 1024.0 / 1024.0 / 1024.0:.2f}GB/s"
    )

    # Prepare host buffer and copy to GPU 0
    print(f"Preparing host buffer and memcpy to GPU{gpuid[0]}...")

    h0_local = (ctypes.c_float * int(buf_size / np.dtype(np.float32).itemsize)).from_address(h0)
    for i in range(int(buf_size / np.dtype(np.float32).itemsize)):
        h0_local[i] = i % 4096

    check_cuda_errors(cudart.cudaSetDevice(gpuid[0]))
    check_cuda_errors(cudart.cudaMemcpy(g0, h0, buf_size, cudart.cudaMemcpyKind.cudaMemcpyDefault))

    # Kernel launch configuration
    threads = cudart.dim3()
    threads.x = 512
    threads.y = 1
    threads.z = 1
    blocks = cudart.dim3()
    blocks.x = (buf_size / np.dtype(np.float32).itemsize) / threads.x
    blocks.y = 1
    blocks.z = 1

    # Run kernel on GPU 1, reading input from the GPU 0 buffer, writing
    # output to the GPU 1 buffer
    print(f"Run kernel on GPU{gpuid[1]}, taking source data from GPU{gpuid[0]} and writing to GPU{gpuid[1]}...")
    check_cuda_errors(cudart.cudaSetDevice(gpuid[1]))

    kernel_helper = [None] * 2
    _simple_kernel = [None] * 2
    kernel_args = [None] * 2

    kernel_helper[1] = common.KernelHelper(simplep2p, gpuid[1])
    _simple_kernel[1] = kernel_helper[1].get_function(b"SimpleKernel")
    kernel_args[1] = ((g0, g1), (ctypes.c_void_p, ctypes.c_void_p))
    check_cuda_errors(
        cuda.cuLaunchKernel(
            _simple_kernel[1],
            blocks.x,
            blocks.y,
            blocks.z,
            threads.x,
            threads.y,
            threads.z,
            0,
            0,
            kernel_args[1],
            0,
        )
    )

    check_cuda_errors(cudart.cudaDeviceSynchronize())

    # Run kernel on GPU 0, reading input from the GPU 1 buffer, writing
    # output to the GPU 0 buffer
    print(f"Run kernel on GPU{gpuid[0]}, taking source data from GPU{gpuid[1]} and writing to GPU{gpuid[0]}...")
    check_cuda_errors(cudart.cudaSetDevice(gpuid[0]))
    kernel_helper[0] = common.KernelHelper(simplep2p, gpuid[0])
    _simple_kernel[0] = kernel_helper[0].get_function(b"SimpleKernel")
    kernel_args[0] = ((g1, g0), (ctypes.c_void_p, ctypes.c_void_p))
    check_cuda_errors(
        cuda.cuLaunchKernel(
            _simple_kernel[0],
            blocks.x,
            blocks.y,
            blocks.z,
            threads.x,
            threads.y,
            threads.z,
            0,
            0,
            kernel_args[0],
            0,
        )
    )

    check_cuda_errors(cudart.cudaDeviceSynchronize())

    # Copy data back to host and verify
    print(f"Copy data back to host from GPU{gpuid[0]} and verify results...")
    check_cuda_errors(cudart.cudaMemcpy(h0, g0, buf_size, cudart.cudaMemcpyKind.cudaMemcpyDefault))

    error_count = 0

    for i in range(int(buf_size / np.dtype(np.float32).itemsize)):
        # Re-generate input data and apply 2x '* 2.0f' computation of both
        # kernel runs
        if h0_local[i] != float(i % 4096) * 2.0 * 2.0:
            print(f"Verification error @ element {i}: val = {h0_local[i]}, ref = {float(i % 4096) * 2.0 * 2.0}\n")
            error_count += 1
            if error_count > 10:
                break

    # Disable peer access (also unregisters memory for non-UVA cases)
    print("Disabling peer access...")
    check_cuda_errors(cudart.cudaSetDevice(gpuid[0]))
    check_cuda_errors(cudart.cudaDeviceDisablePeerAccess(gpuid[1]))
    check_cuda_errors(cudart.cudaSetDevice(gpuid[1]))
    check_cuda_errors(cudart.cudaDeviceDisablePeerAccess(gpuid[0]))

    # Cleanup and shutdown
    print("Shutting down...")
    check_cuda_errors(cudart.cudaEventDestroy(start_event))
    check_cuda_errors(cudart.cudaEventDestroy(stop_event))
    check_cuda_errors(cudart.cudaSetDevice(gpuid[0]))
    check_cuda_errors(cudart.cudaFree(g0))
    check_cuda_errors(cudart.cudaSetDevice(gpuid[1]))
    check_cuda_errors(cudart.cudaFree(g1))
    check_cuda_errors(cudart.cudaFreeHost(h0))

    for i in range(gpu_n):
        check_cuda_errors(cudart.cudaSetDevice(i))

    if error_count != 0:
        print("Test failed!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
