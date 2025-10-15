# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import platform
import sys

import numpy as np
from common import common
from common.helper_cuda import checkCudaErrors
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
    print("Starting...")

    if platform.system() == "Darwin":
        print("simpleP2P is not supported on Mac OSX - waiving sample")
        return

    if platform.machine() == "armv7l":
        print("simpleP2P is not supported on ARMv7 - waiving sample")
        return

    if platform.machine() == "aarch64":
        print("simpleP2P is not supported on aarch64 - waiving sample")
        return

    if platform.machine() == "sbsa":
        print("simpleP2P is not supported on sbsa - waiving sample")
        return

    # Number of GPUs
    print("Checking for multiple GPUs...")
    gpu_n = checkCudaErrors(cudart.cudaGetDeviceCount())
    print(f"CUDA-capable device count: {gpu_n}")

    if gpu_n < 2:
        print("Two or more GPUs with Peer-to-Peer access capability are required")
        return

    prop = [checkCudaErrors(cudart.cudaGetDeviceProperties(i)) for i in range(gpu_n)]
    # Check possibility for peer access
    print("\nChecking GPU(s) for support of peer to peer memory access...")

    p2pCapableGPUs = [-1, -1]
    for i in range(gpu_n):
        p2pCapableGPUs[0] = i
        for j in range(gpu_n):
            if i == j:
                continue
            i_access_j = checkCudaErrors(cudart.cudaDeviceCanAccessPeer(i, j))
            j_access_i = checkCudaErrors(cudart.cudaDeviceCanAccessPeer(j, i))
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
                p2pCapableGPUs[1] = j
                break
        if p2pCapableGPUs[1] != -1:
            break

    if p2pCapableGPUs[0] == -1 or p2pCapableGPUs[1] == -1:
        print("Two or more GPUs with Peer-to-Peer access capability are required.")
        print("Peer to Peer access is not available amongst GPUs in the system, waiving test.")
        return

    # Use first pair of p2p capable GPUs detected
    gpuid = [p2pCapableGPUs[0], p2pCapableGPUs[1]]

    # Enable peer access
    print(f"Enabling peer access between GPU{gpuid[0]} and GPU{gpuid[1]}...")
    checkCudaErrors(cudart.cudaSetDevice(gpuid[0]))
    checkCudaErrors(cudart.cudaDeviceEnablePeerAccess(gpuid[1], 0))
    checkCudaErrors(cudart.cudaSetDevice(gpuid[1]))
    checkCudaErrors(cudart.cudaDeviceEnablePeerAccess(gpuid[0], 0))

    # Allocate buffers
    buf_size = 1024 * 1024 * 16 * np.dtype(np.float32).itemsize
    print(f"Allocating buffers ({int(buf_size / 1024 / 1024)}MB on GPU{gpuid[0]}, GPU{gpuid[1]} and CPU Host)...")
    checkCudaErrors(cudart.cudaSetDevice(gpuid[0]))
    g0 = checkCudaErrors(cudart.cudaMalloc(buf_size))
    checkCudaErrors(cudart.cudaSetDevice(gpuid[1]))
    g1 = checkCudaErrors(cudart.cudaMalloc(buf_size))
    h0 = checkCudaErrors(cudart.cudaMallocHost(buf_size))  # Automatically portable with UVA

    # Create CUDA event handles
    print("Creating event handles...")
    eventflags = cudart.cudaEventBlockingSync
    start_event = checkCudaErrors(cudart.cudaEventCreateWithFlags(eventflags))
    stop_event = checkCudaErrors(cudart.cudaEventCreateWithFlags(eventflags))

    # P2P memcopy() benchmark
    checkCudaErrors(cudart.cudaEventRecord(start_event, cudart.cudaStream_t(0)))

    for i in range(100):
        # With UVA we don't need to specify source and target devices, the
        # runtime figures this out by itself from the pointers
        # Ping-pong copy between GPUs
        if i % 2 == 0:
            checkCudaErrors(cudart.cudaMemcpy(g1, g0, buf_size, cudart.cudaMemcpyKind.cudaMemcpyDefault))
        else:
            checkCudaErrors(cudart.cudaMemcpy(g0, g1, buf_size, cudart.cudaMemcpyKind.cudaMemcpyDefault))

    checkCudaErrors(cudart.cudaEventRecord(stop_event, cudart.cudaStream_t(0)))
    checkCudaErrors(cudart.cudaEventSynchronize(stop_event))
    time_memcpy = checkCudaErrors(cudart.cudaEventElapsedTime(start_event, stop_event))
    print(
        f"cudaMemcpyPeer / cudaMemcpy between GPU{gpuid[0]} and GPU{gpuid[1]}: {(1.0 / (time_memcpy / 1000.0)) * (100.0 * buf_size) / 1024.0 / 1024.0 / 1024.0:.2f}GB/s"
    )

    # Prepare host buffer and copy to GPU 0
    print(f"Preparing host buffer and memcpy to GPU{gpuid[0]}...")

    h0_local = (ctypes.c_float * int(buf_size / np.dtype(np.float32).itemsize)).from_address(h0)
    for i in range(int(buf_size / np.dtype(np.float32).itemsize)):
        h0_local[i] = i % 4096

    checkCudaErrors(cudart.cudaSetDevice(gpuid[0]))
    checkCudaErrors(cudart.cudaMemcpy(g0, h0, buf_size, cudart.cudaMemcpyKind.cudaMemcpyDefault))

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
    checkCudaErrors(cudart.cudaSetDevice(gpuid[1]))

    kernelHelper = [None] * 2
    _simpleKernel = [None] * 2
    kernelArgs = [None] * 2

    kernelHelper[1] = common.KernelHelper(simplep2p, gpuid[1])
    _simpleKernel[1] = kernelHelper[1].getFunction(b"SimpleKernel")
    kernelArgs[1] = ((g0, g1), (ctypes.c_void_p, ctypes.c_void_p))
    checkCudaErrors(
        cuda.cuLaunchKernel(
            _simpleKernel[1],
            blocks.x,
            blocks.y,
            blocks.z,
            threads.x,
            threads.y,
            threads.z,
            0,
            0,
            kernelArgs[1],
            0,
        )
    )

    checkCudaErrors(cudart.cudaDeviceSynchronize())

    # Run kernel on GPU 0, reading input from the GPU 1 buffer, writing
    # output to the GPU 0 buffer
    print(f"Run kernel on GPU{gpuid[0]}, taking source data from GPU{gpuid[1]} and writing to GPU{gpuid[0]}...")
    checkCudaErrors(cudart.cudaSetDevice(gpuid[0]))
    kernelHelper[0] = common.KernelHelper(simplep2p, gpuid[0])
    _simpleKernel[0] = kernelHelper[0].getFunction(b"SimpleKernel")
    kernelArgs[0] = ((g1, g0), (ctypes.c_void_p, ctypes.c_void_p))
    checkCudaErrors(
        cuda.cuLaunchKernel(
            _simpleKernel[0],
            blocks.x,
            blocks.y,
            blocks.z,
            threads.x,
            threads.y,
            threads.z,
            0,
            0,
            kernelArgs[0],
            0,
        )
    )

    checkCudaErrors(cudart.cudaDeviceSynchronize())

    # Copy data back to host and verify
    print(f"Copy data back to host from GPU{gpuid[0]} and verify results...")
    checkCudaErrors(cudart.cudaMemcpy(h0, g0, buf_size, cudart.cudaMemcpyKind.cudaMemcpyDefault))

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
    checkCudaErrors(cudart.cudaSetDevice(gpuid[0]))
    checkCudaErrors(cudart.cudaDeviceDisablePeerAccess(gpuid[1]))
    checkCudaErrors(cudart.cudaSetDevice(gpuid[1]))
    checkCudaErrors(cudart.cudaDeviceDisablePeerAccess(gpuid[0]))

    # Cleanup and shutdown
    print("Shutting down...")
    checkCudaErrors(cudart.cudaEventDestroy(start_event))
    checkCudaErrors(cudart.cudaEventDestroy(stop_event))
    checkCudaErrors(cudart.cudaSetDevice(gpuid[0]))
    checkCudaErrors(cudart.cudaFree(g0))
    checkCudaErrors(cudart.cudaSetDevice(gpuid[1]))
    checkCudaErrors(cudart.cudaFree(g1))
    checkCudaErrors(cudart.cudaFreeHost(h0))

    for i in range(gpu_n):
        checkCudaErrors(cudart.cudaSetDevice(i))

    if error_count != 0:
        print("Test failed!")
        sys.exit(-1)
    print("Test passed!")


if __name__ == "__main__":
    main()
