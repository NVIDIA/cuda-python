# Copyright 2021 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import ctypes
import math
import numpy as np
from cuda import cuda
from examples.common import common
from examples.common.helper_cuda import checkCudaErrors, findCudaDeviceDRV

vectorAddDrv = '''\
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
'''

def main():
    print("Vector Addition (Driver API)")
    N = 50000
    devID = 0
    size = N * np.dtype(np.float32).itemsize

    # Initialize
    checkCudaErrors(cuda.cuInit(0));

    cuDevice = findCudaDeviceDRV()
    # Create context
    cuContext = checkCudaErrors(cuda.cuCtxCreate(0, cuDevice))

    uvaSupported = checkCudaErrors(cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, cuDevice))
    if not uvaSupported:
        print("Accessing pageable memory directly requires UVA")
        return

    kernelHelper = common.KernelHelper(vectorAddDrv, int(cuDevice))
    _VecAdd_kernel = kernelHelper.getFunction(b'VecAdd_kernel')

    # Allocate input vectors h_A and h_B in host memory
    h_A = np.random.rand(size).astype(dtype=np.float32)
    h_B = np.random.rand(size).astype(dtype=np.float32)
    h_C = np.random.rand(size).astype(dtype=np.float32)

    # Allocate vectors in device memory
    d_A = checkCudaErrors(cuda.cuMemAlloc(size))
    d_B = checkCudaErrors(cuda.cuMemAlloc(size))
    d_C = checkCudaErrors(cuda.cuMemAlloc(size))

    # Copy vectors from host memory to device memory
    checkCudaErrors(cuda.cuMemcpyHtoD(d_A, h_A, size))
    checkCudaErrors(cuda.cuMemcpyHtoD(d_B, h_B, size))

    if True:
        # Grid/Block configuration
        threadsPerBlock = 256
        blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock

        kernelArgs = ((d_A, d_B, d_C, N),
                      (None, None, None, ctypes.c_int))

        # Launch the CUDA kernel
        checkCudaErrors(cuda.cuLaunchKernel(_VecAdd_kernel,
                                            blocksPerGrid, 1, 1,
                                            threadsPerBlock, 1, 1,
                                            0, cuda.CUstream(0),
                                            kernelArgs, 0))
    else:
        pass

    # Copy result from device memory to host memory
    # h_C contains the result in host memory
    checkCudaErrors(cuda.cuMemcpyDtoH(h_C, d_C, size))

    for i in range(N):
        sum_all = h_A[i] + h_B[i]
        if math.fabs(h_C[i] - sum_all) > 1e-7:
            break

    # Free device memory
    checkCudaErrors(cuda.cuMemFree(d_A))
    checkCudaErrors(cuda.cuMemFree(d_B))
    checkCudaErrors(cuda.cuMemFree(d_C))

    checkCudaErrors(cuda.cuCtxDestroy(cuContext))
    print("{}".format("Result = PASS" if i+1 == N else "Result = FAIL"))
    if i+1 != N:
        sys.exit(-1)

if __name__ == "__main__":
    main()
