# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import math
import sys

import numpy as np
from common import common
from common.helper_cuda import check_cuda_errors, find_cuda_device_drv

from cuda.bindings import driver as cuda

vector_add_drv = """\
/* Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 3
 * of the programming guide with some additions like error checking.
 *
 */

// Device code
extern "C" __global__ void VecAdd_kernel(const float *A, const float *B, float *C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
        C[i] = A[i] + B[i];
}
"""


def main():
    n = 50000
    nbytes = n * np.dtype(np.float32).itemsize

    # Initialize
    check_cuda_errors(cuda.cuInit(0))
    cu_device = find_cuda_device_drv()
    # Create context
    cu_context = check_cuda_errors(cuda.cuCtxCreate(None, 0, cu_device))

    uva_supported = check_cuda_errors(
        cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, cu_device)
    )
    if not uva_supported:
        import pytest

        pytest.skip("Accessing pageable memory directly requires UVA")

    kernel_helper = common.KernelHelper(vector_add_drv, int(cu_device))
    _vec_add_kernel = kernel_helper.get_function(b"VecAdd_kernel")

    # Allocate input vectors h_A and h_B in host memory
    h_a = np.random.rand(n).astype(dtype=np.float32)
    h_b = np.random.rand(n).astype(dtype=np.float32)
    h_c = np.random.rand(n).astype(dtype=np.float32)

    # Allocate vectors in device memory
    d_a = check_cuda_errors(cuda.cuMemAlloc(nbytes))
    d_b = check_cuda_errors(cuda.cuMemAlloc(nbytes))
    d_c = check_cuda_errors(cuda.cuMemAlloc(nbytes))

    # Copy vectors from host memory to device memory
    check_cuda_errors(cuda.cuMemcpyHtoD(d_a, h_a, nbytes))
    check_cuda_errors(cuda.cuMemcpyHtoD(d_b, h_b, nbytes))

    if True:
        # Grid/Block configuration
        threads_per_block = 256
        blocks_per_grid = (n + threads_per_block - 1) / threads_per_block

        kernel_args = ((d_a, d_b, d_c, n), (None, None, None, ctypes.c_int))

        # Launch the CUDA kernel
        check_cuda_errors(
            cuda.cuLaunchKernel(
                _vec_add_kernel,
                blocks_per_grid,
                1,
                1,
                threads_per_block,
                1,
                1,
                0,
                0,
                kernel_args,
                0,
            )
        )
    else:
        pass

    # Copy result from device memory to host memory
    # h_C contains the result in host memory
    check_cuda_errors(cuda.cuMemcpyDtoH(h_c, d_c, nbytes))

    for i in range(n):
        sum_all = h_a[i] + h_b[i]
        if math.fabs(h_c[i] - sum_all) > 1e-7:
            break

    # Free device memory
    check_cuda_errors(cuda.cuMemFree(d_a))
    check_cuda_errors(cuda.cuMemFree(d_b))
    check_cuda_errors(cuda.cuMemFree(d_c))

    check_cuda_errors(cuda.cuCtxDestroy(cu_context))
    if i + 1 != n:
        print("Result = FAIL", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
