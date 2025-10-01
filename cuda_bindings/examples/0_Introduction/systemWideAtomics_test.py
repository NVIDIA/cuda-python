# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import os
import sys

import numpy as np
from common import common
from common.helper_cuda import checkCudaErrors, findCudaDevice
from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart

systemWideAtomics = """\
#define LOOP_NUM 50

extern "C"
__global__ void atomicKernel(int *atom_arr) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = 0; i < LOOP_NUM; i++) {
        // Atomic addition
        atomicAdd_system(&atom_arr[0], 10);

        // Atomic exchange
        atomicExch_system(&atom_arr[1], tid);

        // Atomic maximum
        atomicMax_system(&atom_arr[2], tid);

        // Atomic minimum
        atomicMin_system(&atom_arr[3], tid);

        // Atomic increment (modulo 17+1)
        atomicInc_system((unsigned int *)&atom_arr[4], 17);

        // Atomic decrement
        atomicDec_system((unsigned int *)&atom_arr[5], 137);

        // Atomic compare-and-swap
        atomicCAS_system(&atom_arr[6], tid - 1, tid);

        // Bitwise atomic instructions

        // Atomic AND
        atomicAnd_system(&atom_arr[7], 2 * tid + 7);

        // Atomic OR
        atomicOr_system(&atom_arr[8], 1 << tid);

        // Atomic XOR
        atomicXor_system(&atom_arr[9], tid);
  }
}
"""

LOOP_NUM = 50


#! Compute reference data set
#! Each element is multiplied with the number of threads / array length
#! @param reference  reference data, computed but preallocated
#! @param idata      input data as provided to device
#! @param len        number of elements in reference / idata
def verify(testData, length):
    val = 0

    for i in range(length * LOOP_NUM):
        val += 10

    if val != testData[0]:
        print(f"atomicAdd failed val = {val} testData = {testData[0]}")
        return False

    val = 0
    found = False
    for i in range(length):
        # second element should be a member of [0, len)
        if i == testData[1]:
            found = True
            break

    if not found:
        print("atomicExch failed")
        return False

    val = -(1 << 8)

    for i in range(length):
        # third element should be len-1
        val = max(val, i)

    if val != testData[2]:
        print("atomicMax failed")
        return False

    val = 1 << 8

    for i in range(length):
        val = min(val, i)

    if val != testData[3]:
        print("atomicMin failed")
        return False

    limit = 17
    val = 0

    for i in range(length * LOOP_NUM):
        val = 0 if val >= limit else val + 1

    if val != testData[4]:
        print("atomicInc failed")
        return False

    limit = 137
    val = 0

    for i in range(length * LOOP_NUM):
        val = limit if (val == 0) or (val > limit) else val - 1

    if val != testData[5]:
        print("atomicDec failed")
        return False

    found = False

    for i in range(length):
        # seventh element should be a member of [0, len)
        if i == testData[6]:
            found = True
            break

    if not found:
        print("atomicCAS failed")
        return False

    val = 0xFF

    for i in range(length):
        # 8th element should be 1
        val &= 2 * i + 7

    if val != testData[7]:
        print("atomicAnd failed")
        return False

    # 9th element should be 0xff
    val = -1
    if val != testData[8]:
        print("atomicOr failed")
        return False

    val = 0xFF

    for i in range(length):
        # 11th element should be 0xff
        val ^= i

    if val != testData[9]:
        print("atomicXor failed")
        return False

    return True


def main():
    if os.name == "nt":
        print("Atomics not supported on Windows")
        return

    # set device
    dev_id = findCudaDevice()
    device_prop = checkCudaErrors(cudart.cudaGetDeviceProperties(dev_id))

    if not device_prop.managedMemory:
        # This samples requires being run on a device that supports Unified Memory
        print("Unified Memory not supported on this device")
        return

    computeMode = checkCudaErrors(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeMode, dev_id))
    if computeMode == cudart.cudaComputeMode.cudaComputeModeProhibited:
        # This sample requires being run with a default or process exclusive mode
        print("This sample requires a device in either default or process exclusive mode")
        return

    if device_prop.major < 6:
        print("Requires a minimum CUDA compute 6.0 capability, waiving testing.")
        return

    numThreads = 256
    numBlocks = 64
    numData = 10

    if device_prop.pageableMemoryAccess:
        print("CAN access pageable memory")
        atom_arr_h = (ctypes.c_int * numData)(0)
        atom_arr = ctypes.addressof(atom_arr_h)
    else:
        print("CANNOT access pageable memory")
        atom_arr = checkCudaErrors(
            cudart.cudaMallocManaged(np.dtype(np.int32).itemsize * numData, cudart.cudaMemAttachGlobal)
        )
        atom_arr_h = (ctypes.c_int * numData).from_address(atom_arr)

    for i in range(numData):
        atom_arr_h[i] = 0

    # To make the AND and XOR tests generate something other than 0...
    atom_arr_h[7] = atom_arr_h[9] = 0xFF

    kernelHelper = common.KernelHelper(systemWideAtomics, dev_id)
    _atomicKernel = kernelHelper.getFunction(b"atomicKernel")
    kernelArgs = ((atom_arr,), (ctypes.c_void_p,))
    checkCudaErrors(
        cuda.cuLaunchKernel(
            _atomicKernel,
            numBlocks,
            1,
            1,  # grid dim
            numThreads,
            1,
            1,  # block dim
            0,
            cuda.CU_STREAM_LEGACY,  # shared mem and stream
            kernelArgs,
            0,
        )
    )  # arguments
    # NOTE: Python doesn't have an equivalent system atomic operations
    # atomicKernel_CPU(atom_arr_h, numBlocks * numThreads)

    checkCudaErrors(cudart.cudaDeviceSynchronize())

    # Compute & verify reference solution
    testResult = verify(atom_arr_h, numThreads * numBlocks)

    if device_prop.pageableMemoryAccess:
        pass
    else:
        checkCudaErrors(cudart.cudaFree(atom_arr))

    print("systemWideAtomics completed, returned {}".format("OK" if testResult else "ERROR!"))
    if not testResult:
        sys.exit(-1)


if __name__ == "__main__":
    main()
