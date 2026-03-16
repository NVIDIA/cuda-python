# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import math
import platform
import sys
from random import random

import numpy as np
from common import common
from common.helper_cuda import check_cuda_errors, find_cuda_device

from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart

conjugate_gradient_multi_block_cg = """\
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


def gen_tridiag(i, j, val, n, nz):
    i[0] = 0
    j[0] = 0
    j[1] = 0

    val[0] = float(random()) + 10.0
    val[1] = float(random())

    for i in range(1, n):
        if i > 1:
            i[i] = i[i - 1] + 3
        else:
            i[1] = 2

        start = (i - 1) * 3 + 2
        j[start] = i - 1
        j[start + 1] = i

        if i < n - 1:
            j[start + 2] = i + 1

        val[start] = val[start - 1]
        val[start + 1] = float(random()) + 10.0

        if i < n - 1:
            val[start + 2] = float(random())
    i[n] = nz


THREADS_PER_BLOCK = 512
s_sd_kname = "conjugateGradientMultiBlockCG"


def main():
    tol = 1e-5

    import pytest

    # WAIVE: Due to bug in NVRTC
    return

    if platform.system() == "Darwin":
        pytest.skip("conjugateGradientMultiBlockCG is not supported on Mac OSX")

    if platform.machine() == "armv7l":
        pytest.skip("conjugateGradientMultiBlockCG is not supported on ARMv7")

    if platform.machine() == "qnx":
        pytest.skip("conjugateGradientMultiBlockCG is not supported on QNX")

    # This will pick the best possible CUDA capable device
    dev_id = find_cuda_device()
    device_prop = check_cuda_errors(cudart.cudaGetDeviceProperties(dev_id))

    if not device_prop.managedMemory:
        pytest.skip("Unified Memory not supported on this device")

    # This sample requires being run on a device that supports Cooperative Kernel
    # Launch
    if not device_prop.cooperativeLaunch:
        pytest.skip(f"Selected GPU {dev_id} does not support Cooperative Kernel Launch")

    # Statistics about the GPU device
    print(
        f"> GPU device has {device_prop.multiProcessorCount:%d} Multi-Processors, SM {device_prop.major:%d}.{device_prop.minor:%d} compute capabilities\n"
    )

    # Get kernel
    kernel_helper = common.KernelHelper(conjugate_gradient_multi_block_cg, dev_id)
    _gpu_conjugate_gradient = kernel_helper.get_function(b"gpuConjugateGradient")

    # Generate a random tridiagonal symmetric matrix in CSR format
    n = 1048576
    nz = (n - 2) * 3 + 4

    i = check_cuda_errors(cudart.cudaMallocManaged(np.dtype(np.int32).itemsize * (n + 1), cudart.cudaMemAttachGlobal))
    j = check_cuda_errors(cudart.cudaMallocManaged(np.dtype(np.int32).itemsize * nz, cudart.cudaMemAttachGlobal))
    val = check_cuda_errors(cudart.cudaMallocManaged(np.dtype(np.float32).itemsize * nz, cudart.cudaMemAttachGlobal))
    i_local = (ctypes.c_int * (n + 1)).from_address(i)
    j_local = (ctypes.c_int * nz).from_address(j)
    val_local = (ctypes.c_float * nz).from_address(val)

    gen_tridiag(i_local, j_local, val_local, n, nz)

    x = check_cuda_errors(cudart.cudaMallocManaged(np.dtype(np.float32).itemsize * n, cudart.cudaMemAttachGlobal))
    rhs = check_cuda_errors(cudart.cudaMallocManaged(np.dtype(np.float32).itemsize * n, cudart.cudaMemAttachGlobal))
    dot_result = check_cuda_errors(cudart.cudaMallocManaged(np.dtype(np.float64).itemsize, cudart.cudaMemAttachGlobal))
    x_local = (ctypes.c_float * n).from_address(x)
    rhs_local = (ctypes.c_float * n).from_address(rhs)
    dot_result_local = (ctypes.c_double).from_address(dot_result)
    dot_result_local = 0

    # temp memory for CG
    r = check_cuda_errors(cudart.cudaMallocManaged(np.dtype(np.float32).itemsize * n, cudart.cudaMemAttachGlobal))
    p = check_cuda_errors(cudart.cudaMallocManaged(np.dtype(np.float32).itemsize * n, cudart.cudaMemAttachGlobal))
    ax = check_cuda_errors(cudart.cudaMallocManaged(np.dtype(np.float32).itemsize * n, cudart.cudaMemAttachGlobal))
    r_local = (ctypes.c_float * n).from_address(r)

    check_cuda_errors(cudart.cudaDeviceSynchronize())

    start = check_cuda_errors(cudart.cudaEventCreate())
    stop = check_cuda_errors(cudart.cudaEventCreate())

    for i in range(n):
        r_local[i] = rhs_local[i] = 1.0
        x_local[i] = 0.0

    kernel_args_value = (i, j, val, x, ax, p, r, dot_result, nz, n, tol)
    kernel_args_types = (
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
    kernel_args = (kernel_args_value, kernel_args_types)

    s_mem_size = np.dtype(np.float64).itemsize * ((THREADS_PER_BLOCK / 32) + 1)
    num_threads = THREADS_PER_BLOCK
    num_blocks_per_sm = check_cuda_errors(
        cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(_gpu_conjugate_gradient, num_threads, s_mem_size)
    )
    num_sms = device_prop.multiProcessorCount
    dim_grid = cudart.dim3()
    dim_grid.x = num_sms * num_blocks_per_sm
    dim_grid.y = 1
    dim_grid.z = 1
    dim_block = cudart.dim3()
    dim_block.x = THREADS_PER_BLOCK
    dim_block.y = 1
    dim_block.z = 1

    check_cuda_errors(cudart.cudaEventRecord(start, 0))
    check_cuda_errors(
        cuda.cuLaunchCooperativeKernel(
            _gpu_conjugate_gradient,
            dim_grid.x,
            dim_grid.y,
            dim_grid.z,
            dim_block.x,
            dim_block.y,
            dim_block.z,
            0,
            0,
            kernel_args,
        )
    )
    check_cuda_errors(cudart.cudaEventRecord(stop, 0))
    check_cuda_errors(cudart.cudaDeviceSynchronize())

    time = check_cuda_errors(cudart.cudaEventElapsedTime(start, stop))
    print(f"GPU Final, residual = {math.sqrt(dot_result_local):e}, kernel execution time = {time:f} ms")

    err = 0.0
    for i in range(n):
        rsum = 0.0

        for j in range(i_local[i], i_local[i + 1]):
            rsum += val_local[j] * x_local[j_local[j]]

        diff = math.fabs(rsum - rhs_local[i])

        if diff > err:
            err = diff

    check_cuda_errors(cudart.cudaFree(i))
    check_cuda_errors(cudart.cudaFree(j))
    check_cuda_errors(cudart.cudaFree(val))
    check_cuda_errors(cudart.cudaFree(x))
    check_cuda_errors(cudart.cudaFree(rhs))
    check_cuda_errors(cudart.cudaFree(r))
    check_cuda_errors(cudart.cudaFree(p))
    check_cuda_errors(cudart.cudaFree(ax))
    check_cuda_errors(cudart.cudaFree(dot_result))
    check_cuda_errors(cudart.cudaEventDestroy(start))
    check_cuda_errors(cudart.cudaEventDestroy(stop))

    print(f"Test Summary:  Error amount = {err:f}")
    if math.sqrt(dot_result_local) >= tol:
        print("conjugateGradientMultiBlockCG FAILED", file=sys.stderr)
        sys.exit(1)
