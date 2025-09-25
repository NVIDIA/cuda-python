# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import math
import platform
import sys
from random import random

import numpy as np
from common import common
from common.helper_cuda import checkCudaErrors, findCudaDevice
from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart

conjugateGradientMultiBlockCG = """\
#line __LINE__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


__device__ void gpuSpMV(int *I, int *J, float *val, int nnz, int num_rows,
                        float alpha, float *inputVecX, float *outputVecY,
                        cg::thread_block &cta, const cg::grid_group &grid) {
  for (int i = grid.thread_rank(); i < num_rows; i += grid.size()) {
    int row_elem = I[i];
    int next_row_elem = I[i + 1];
    int num_elems_this_row = next_row_elem - row_elem;

    float output = 0.0;
    for (int j = 0; j < num_elems_this_row; j++) {
      // I or J or val arrays - can be put in shared memory
      // as the access is random and reused in next calls of gpuSpMV function.
      output += alpha * val[row_elem + j] * inputVecX[J[row_elem + j]];
    }

    outputVecY[i] = output;
  }
}

__device__ void gpuSaxpy(float *x, float *y, float a, int size,
                         const cg::grid_group &grid) {
  for (int i = grid.thread_rank(); i < size; i += grid.size()) {
    y[i] = a * x[i] + y[i];
  }
}

__device__ void gpuDotProduct(float *vecA, float *vecB, double *result,
                              int size, const cg::thread_block &cta,
                              const cg::grid_group &grid) {
  extern __shared__ double tmp[];

  double temp_sum = 0.0;
  for (int i = grid.thread_rank(); i < size; i += grid.size()) {
    temp_sum += static_cast<double>(vecA[i] * vecB[i]);
  }

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  temp_sum = cg::reduce(tile32, temp_sum, cg::plus<double>());

  if (tile32.thread_rank() == 0) {
    tmp[tile32.meta_group_rank()] = temp_sum;
  }

  cg::sync(cta);

  if (tile32.meta_group_rank() == 0) {
     temp_sum = tile32.thread_rank() < tile32.meta_group_size() ? tmp[tile32.thread_rank()] : 0.0;
     temp_sum = cg::reduce(tile32, temp_sum, cg::plus<double>());

    if (tile32.thread_rank() == 0) {
      atomicAdd(result, temp_sum);
    }
  }
}

__device__ void gpuCopyVector(float *srcA, float *destB, int size,
                              const cg::grid_group &grid) {
  for (int i = grid.thread_rank(); i < size; i += grid.size()) {
    destB[i] = srcA[i];
  }
}

__device__ void gpuScaleVectorAndSaxpy(const float *x, float *y, float a, float scale, int size,
                         const cg::grid_group &grid) {
  for (int i = grid.thread_rank(); i < size; i += grid.size()) {
    y[i] = a * x[i] + scale * y[i];
  }
}

extern "C" __global__ void gpuConjugateGradient(int *I, int *J, float *val,
                                                float *x, float *Ax, float *p,
                                                float *r, double *dot_result,
                                                int nnz, int N, float tol) {
  cg::thread_block cta = cg::this_thread_block();
  cg::grid_group grid = cg::this_grid();

  int max_iter = 10000;

  float alpha = 1.0;
  float alpham1 = -1.0;
  float r0 = 0.0, r1, b, a, na;

  gpuSpMV(I, J, val, nnz, N, alpha, x, Ax, cta, grid);

  cg::sync(grid);

  gpuSaxpy(Ax, r, alpham1, N, grid);

  cg::sync(grid);

  gpuDotProduct(r, r, dot_result, N, cta, grid);

  cg::sync(grid);

  r1 = *dot_result;

  int k = 1;
  while (r1 > tol * tol && k <= max_iter) {
    if (k > 1) {
      b = r1 / r0;
      gpuScaleVectorAndSaxpy(r, p, alpha, b, N, grid);
    } else {
      gpuCopyVector(r, p, N, grid);
    }

    cg::sync(grid);

    gpuSpMV(I, J, val, nnz, N, alpha, p, Ax, cta, grid);

    if (threadIdx.x == 0 && blockIdx.x == 0) *dot_result = 0.0;

    cg::sync(grid);

    gpuDotProduct(p, Ax, dot_result, N, cta, grid);

    cg::sync(grid);

    a = r1 / *dot_result;

    gpuSaxpy(p, x, a, N, grid);
    na = -a;
    gpuSaxpy(Ax, r, na, N, grid);

    r0 = r1;

    cg::sync(grid);
    if (threadIdx.x == 0 && blockIdx.x == 0) *dot_result = 0.0;

    cg::sync(grid);

    gpuDotProduct(r, r, dot_result, N, cta, grid);

    cg::sync(grid);

    r1 = *dot_result;
    k++;
  }
}
"""


def genTridiag(I, J, val, N, nz):
    I[0] = 0
    J[0] = 0
    J[1] = 0

    val[0] = float(random()) + 10.0
    val[1] = float(random())

    for i in range(1, N):
        if i > 1:
            I[i] = I[i - 1] + 3
        else:
            I[1] = 2

        start = (i - 1) * 3 + 2
        J[start] = i - 1
        J[start + 1] = i

        if i < N - 1:
            J[start + 2] = i + 1

        val[start] = val[start - 1]
        val[start + 1] = float(random()) + 10.0

        if i < N - 1:
            val[start + 2] = float(random())
    I[N] = nz


THREADS_PER_BLOCK = 512
sSDKname = "conjugateGradientMultiBlockCG"


def main():
    tol = 1e-5

    print(f"Starting [{sSDKname}]...\n")
    # WAIVE: Due to bug in NVRTC
    return

    if platform.system() == "Darwin":
        print("conjugateGradientMultiBlockCG is not supported on Mac OSX - waiving sample")
        return

    if platform.machine() == "armv7l":
        print("conjugateGradientMultiBlockCG is not supported on ARMv7 - waiving sample")
        return

    if platform.machine() == "qnx":
        print("conjugateGradientMultiBlockCG is not supported on QNX - waiving sample")
        return

    # This will pick the best possible CUDA capable device
    devID = findCudaDevice()
    deviceProp = checkCudaErrors(cudart.cudaGetDeviceProperties(devID))

    if not deviceProp.managedMemory:
        # This sample requires being run on a device that supports Unified Memory
        print("Unified Memory not supported on this device")
        return

    # This sample requires being run on a device that supports Cooperative Kernel
    # Launch
    if not deviceProp.cooperativeLaunch:
        print(f"\nSelected GPU {devID:%d} does not support Cooperative Kernel Launch, Waiving the run")
        return

    # Statistics about the GPU device
    print(
        f"> GPU device has {deviceProp.multiProcessorCount:%d} Multi-Processors, SM {deviceProp.major:%d}.{deviceProp.minor:%d} compute capabilities\n"
    )

    # Get kernel
    kernelHelper = common.KernelHelper(conjugateGradientMultiBlockCG, devID)
    _gpuConjugateGradient = kernelHelper.getFunction(b"gpuConjugateGradient")

    # Generate a random tridiagonal symmetric matrix in CSR format
    N = 1048576
    nz = (N - 2) * 3 + 4

    I = checkCudaErrors(cudart.cudaMallocManaged(np.dtype(np.int32).itemsize * (N + 1), cudart.cudaMemAttachGlobal))
    J = checkCudaErrors(cudart.cudaMallocManaged(np.dtype(np.int32).itemsize * nz, cudart.cudaMemAttachGlobal))
    val = checkCudaErrors(cudart.cudaMallocManaged(np.dtype(np.float32).itemsize * nz, cudart.cudaMemAttachGlobal))
    I_local = (ctypes.c_int * (N + 1)).from_address(I)
    J_local = (ctypes.c_int * nz).from_address(J)
    val_local = (ctypes.c_float * nz).from_address(val)

    genTridiag(I_local, J_local, val_local, N, nz)

    x = checkCudaErrors(cudart.cudaMallocManaged(np.dtype(np.float32).itemsize * N, cudart.cudaMemAttachGlobal))
    rhs = checkCudaErrors(cudart.cudaMallocManaged(np.dtype(np.float32).itemsize * N, cudart.cudaMemAttachGlobal))
    dot_result = checkCudaErrors(cudart.cudaMallocManaged(np.dtype(np.float64).itemsize, cudart.cudaMemAttachGlobal))
    x_local = (ctypes.c_float * N).from_address(x)
    rhs_local = (ctypes.c_float * N).from_address(rhs)
    dot_result_local = (ctypes.c_double).from_address(dot_result)
    dot_result_local = 0

    # temp memory for CG
    r = checkCudaErrors(cudart.cudaMallocManaged(np.dtype(np.float32).itemsize * N, cudart.cudaMemAttachGlobal))
    p = checkCudaErrors(cudart.cudaMallocManaged(np.dtype(np.float32).itemsize * N, cudart.cudaMemAttachGlobal))
    Ax = checkCudaErrors(cudart.cudaMallocManaged(np.dtype(np.float32).itemsize * N, cudart.cudaMemAttachGlobal))
    r_local = (ctypes.c_float * N).from_address(r)

    checkCudaErrors(cudart.cudaDeviceSynchronize())

    start = checkCudaErrors(cudart.cudaEventCreate())
    stop = checkCudaErrors(cudart.cudaEventCreate())

    for i in range(N):
        r_local[i] = rhs_local[i] = 1.0
        x_local[i] = 0.0

    kernelArgs_value = (I, J, val, x, Ax, p, r, dot_result, nz, N, tol)
    kernelArgs_types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    )
    kernelArgs = (kernelArgs_value, kernelArgs_types)

    sMemSize = np.dtype(np.float64).itemsize * ((THREADS_PER_BLOCK / 32) + 1)
    numThreads = THREADS_PER_BLOCK
    numBlocksPerSm = checkCudaErrors(
        cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(_gpuConjugateGradient, numThreads, sMemSize)
    )
    numSms = deviceProp.multiProcessorCount
    dimGrid = cudart.dim3()
    dimGrid.x = numSms * numBlocksPerSm
    dimGrid.y = 1
    dimGrid.z = 1
    dimBlock = cudart.dim3()
    dimBlock.x = THREADS_PER_BLOCK
    dimBlock.y = 1
    dimBlock.z = 1

    checkCudaErrors(cudart.cudaEventRecord(start, 0))
    checkCudaErrors(
        cuda.cuLaunchCooperativeKernel(
            _gpuConjugateGradient,
            dimGrid.x,
            dimGrid.y,
            dimGrid.z,
            dimBlock.x,
            dimBlock.y,
            dimBlock.z,
            0,
            0,
            kernelArgs,
        )
    )
    checkCudaErrors(cudart.cudaEventRecord(stop, 0))
    checkCudaErrors(cudart.cudaDeviceSynchronize())

    time = checkCudaErrors(cudart.cudaEventElapsedTime(start, stop))
    print(f"GPU Final, residual = {math.sqrt(dot_result_local):e}, kernel execution time = {time:f} ms")

    err = 0.0
    for i in range(N):
        rsum = 0.0

        for j in range(I_local[i], I_local[i + 1]):
            rsum += val_local[j] * x_local[J_local[j]]

        diff = math.fabs(rsum - rhs_local[i])

        if diff > err:
            err = diff

    checkCudaErrors(cudart.cudaFree(I))
    checkCudaErrors(cudart.cudaFree(J))
    checkCudaErrors(cudart.cudaFree(val))
    checkCudaErrors(cudart.cudaFree(x))
    checkCudaErrors(cudart.cudaFree(rhs))
    checkCudaErrors(cudart.cudaFree(r))
    checkCudaErrors(cudart.cudaFree(p))
    checkCudaErrors(cudart.cudaFree(Ax))
    checkCudaErrors(cudart.cudaFree(dot_result))
    checkCudaErrors(cudart.cudaEventDestroy(start))
    checkCudaErrors(cudart.cudaEventDestroy(stop))

    print(f"Test Summary:  Error amount = {err:f}")
    print("&&&& conjugateGradientMultiBlockCG %s\n" % ("PASSED" if math.sqrt(dot_result_local) < tol else "FAILED"))

    if math.sqrt(dot_result_local) >= tol:
        sys.exit(-1)
