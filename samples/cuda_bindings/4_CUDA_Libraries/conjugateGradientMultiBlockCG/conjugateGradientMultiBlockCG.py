# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# /// script
# dependencies = ["cuda-python>=13.0.0", "numpy>=1.24"]
# ///

"""
Conjugate Gradient solver with cooperative multi-block synchronization

Solves ``A x = b`` for a random sparse tridiagonal SPD matrix ``A`` using
the Conjugate Gradient method. The entire CG iteration runs inside a
single kernel launched via ``cuLaunchCooperativeKernel``; each iteration
performs several ``cg::grid_group::sync()`` synchronizations to move
between phases without returning to the host.

The device-side building blocks are:

  * ``gpuSpMV`` -- sparse matrix-vector multiply (CSR format).
  * ``gpuSaxpy`` -- ``y = a*x + y``.
  * ``gpuDotProduct`` -- warp-shuffle reduce (`cg::reduce`) + atomicAdd across blocks.
  * ``gpuScaleVectorAndSaxpy`` and ``gpuCopyVector`` -- CG bookkeeping.

The high-level flow is the classic CG loop: r = b - A x; loop while
|r|^2 > tol^2 ... update p, alpha, x, r, and re-dot. Because everything
lives in one cooperative kernel, we avoid host round-trips between
iterations.

The sample is the only end-to-end numerical solver in ``/samples/cuda_bindings`` that
uses ``grid.sync()``. The simpler
[`samples/cuda_core/reductionMultiBlockCG/`](../../../cuda_core/reductionMultiBlockCG/)
uses the same underlying feature for a plain reduction.

Waives with exit code 2 on Darwin / QNX / armv7l, on devices without
Unified Memory, and on devices without Cooperative Kernel Launch support.
"""

import ctypes
import math
import platform
import sys
from pathlib import Path
from random import random

# Add samples/cuda_bindings/Utilities/ to the import path for shared bindings helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Utilities"))

try:
    import numpy as np
    from cuda_bindings_utils import KernelHelper, check_cuda_errors, find_cuda_device, requirement_not_met

    from cuda.bindings import driver as cuda
    from cuda.bindings import runtime as cudart
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


CG_KERNEL_SOURCE = """\
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


THREADS_PER_BLOCK = 512
UNSUPPORTED_SYSTEMS = {"Darwin", "QNX"}
UNSUPPORTED_MACHINES = {"armv7l"}


def _gen_tridiag(row_offsets, col_indices, values, n, nz):
    """Random symmetric tridiagonal matrix in CSR format."""
    row_offsets[0] = 0
    col_indices[0] = 0
    col_indices[1] = 0
    values[0] = float(random()) + 10.0
    values[1] = float(random())

    for row_idx in range(1, n):
        if row_idx > 1:
            row_offsets[row_idx] = row_offsets[row_idx - 1] + 3
        else:
            row_offsets[1] = 2

        start = (row_idx - 1) * 3 + 2
        col_indices[start] = row_idx - 1
        col_indices[start + 1] = row_idx
        if row_idx < n - 1:
            col_indices[start + 2] = row_idx + 1

        values[start] = values[start - 1]
        values[start + 1] = float(random()) + 10.0
        if row_idx < n - 1:
            values[start + 2] = float(random())
    row_offsets[n] = nz


def main():
    tol = 1e-5

    system_name = platform.system()
    if system_name in UNSUPPORTED_SYSTEMS:
        requirement_not_met(f"conjugateGradientMultiBlockCG is not supported on {system_name}")

    machine_name = platform.machine()
    if machine_name in UNSUPPORTED_MACHINES:
        requirement_not_met(f"conjugateGradientMultiBlockCG is not supported on {machine_name}")

    dev_id = find_cuda_device()
    device_prop = check_cuda_errors(cudart.cudaGetDeviceProperties(dev_id))

    if not device_prop.managedMemory:
        requirement_not_met("Unified Memory not supported on this device")

    if not device_prop.cooperativeLaunch:
        requirement_not_met(f"Selected GPU {dev_id} does not support Cooperative Kernel Launch")

    print(
        f"> GPU device has {device_prop.multiProcessorCount} Multi-Processors, "
        f"SM {device_prop.major}.{device_prop.minor} compute capability"
    )

    kernel_helper = KernelHelper(CG_KERNEL_SOURCE, dev_id)
    gpu_cg = kernel_helper.get_function(b"gpuConjugateGradient")

    # ---- Allocate a random tridiagonal SPD system in CSR format ----
    n = 1048576
    nz = (n - 2) * 3 + 4

    i = check_cuda_errors(cudart.cudaMallocManaged(np.dtype(np.int32).itemsize * (n + 1), cudart.cudaMemAttachGlobal))
    j = check_cuda_errors(cudart.cudaMallocManaged(np.dtype(np.int32).itemsize * nz, cudart.cudaMemAttachGlobal))
    val = check_cuda_errors(cudart.cudaMallocManaged(np.dtype(np.float32).itemsize * nz, cudart.cudaMemAttachGlobal))
    i_local = (ctypes.c_int * (n + 1)).from_address(i)
    j_local = (ctypes.c_int * nz).from_address(j)
    val_local = (ctypes.c_float * nz).from_address(val)
    _gen_tridiag(i_local, j_local, val_local, n, nz)

    x = check_cuda_errors(cudart.cudaMallocManaged(np.dtype(np.float32).itemsize * n, cudart.cudaMemAttachGlobal))
    rhs = check_cuda_errors(cudart.cudaMallocManaged(np.dtype(np.float32).itemsize * n, cudart.cudaMemAttachGlobal))
    dot_result = check_cuda_errors(cudart.cudaMallocManaged(np.dtype(np.float64).itemsize, cudart.cudaMemAttachGlobal))
    x_local = (ctypes.c_float * n).from_address(x)
    rhs_local = (ctypes.c_float * n).from_address(rhs)
    dot_result_local = ctypes.c_double.from_address(dot_result)
    dot_result_local.value = 0.0

    # ---- CG scratch vectors ----
    r = check_cuda_errors(cudart.cudaMallocManaged(np.dtype(np.float32).itemsize * n, cudart.cudaMemAttachGlobal))
    p = check_cuda_errors(cudart.cudaMallocManaged(np.dtype(np.float32).itemsize * n, cudart.cudaMemAttachGlobal))
    ax = check_cuda_errors(cudart.cudaMallocManaged(np.dtype(np.float32).itemsize * n, cudart.cudaMemAttachGlobal))
    r_local = (ctypes.c_float * n).from_address(r)

    check_cuda_errors(cudart.cudaDeviceSynchronize())

    start = check_cuda_errors(cudart.cudaEventCreate())
    stop = check_cuda_errors(cudart.cudaEventCreate())

    for idx in range(n):
        r_local[idx] = rhs_local[idx] = 1.0
        x_local[idx] = 0.0

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

    # Grid size: max active blocks per SM * number of SMs -- this saturates
    # the device with cooperating blocks so grid.sync() has real work to do.
    s_mem_size = np.dtype(np.float64).itemsize * ((THREADS_PER_BLOCK // 32) + 1)
    num_blocks_per_sm = check_cuda_errors(
        cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(gpu_cg, THREADS_PER_BLOCK, s_mem_size)
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
            gpu_cg,
            dim_grid.x,
            dim_grid.y,
            dim_grid.z,
            dim_block.x,
            dim_block.y,
            dim_block.z,
            s_mem_size,
            0,
            kernel_args,
        )
    )
    check_cuda_errors(cudart.cudaEventRecord(stop, 0))
    check_cuda_errors(cudart.cudaDeviceSynchronize())

    ms = check_cuda_errors(cudart.cudaEventElapsedTime(start, stop))
    residual = math.sqrt(dot_result_local.value)
    print(f"GPU Final, residual = {residual:e}, kernel execution time = {ms:.3f} ms")

    # Host reference: compute max_i | (A x)_i - b_i |
    err = 0.0
    for row_idx in range(n):
        rsum = 0.0
        for elem_idx in range(i_local[row_idx], i_local[row_idx + 1]):
            rsum += val_local[elem_idx] * x_local[j_local[elem_idx]]
        err = max(err, math.fabs(rsum - rhs_local[row_idx]))

    for handle in (i, j, val, x, rhs, r, p, ax, dot_result):
        check_cuda_errors(cudart.cudaFree(handle))
    check_cuda_errors(cudart.cudaEventDestroy(start))
    check_cuda_errors(cudart.cudaEventDestroy(stop))

    print(f"Test Summary: Error amount = {err:f}")
    if residual >= tol:
        print("conjugateGradientMultiBlockCG FAILED", file=sys.stderr)
        return 1

    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
