# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import os
import sys

import numpy as np
from common import common
from common.helper_cuda import check_cuda_errors, find_cuda_device

from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart

system_wide_atomics = """\
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
def verify(test_data, length):
    val = 0

    for i in range(length * LOOP_NUM):
        val += 10

    if val != test_data[0]:
        print(f"atomicAdd failed val = {val} test_data = {test_data[0]}")
        return False

    val = 0
    found = False
    for i in range(length):
        # second element should be a member of [0, len)
        if i == test_data[1]:
            found = True
            break

    if not found:
        print("atomicExch failed")
        return False

    val = -(1 << 8)

    for i in range(length):
        # third element should be len-1
        val = max(val, i)

    if val != test_data[2]:
        print("atomicMax failed")
        return False

    val = 1 << 8

    for i in range(length):
        val = min(val, i)

    if val != test_data[3]:
        print("atomicMin failed")
        return False

    limit = 17
    val = 0

    for i in range(length * LOOP_NUM):
        val = 0 if val >= limit else val + 1

    if val != test_data[4]:
        print("atomicInc failed")
        return False

    limit = 137
    val = 0

    for i in range(length * LOOP_NUM):
        val = limit if (val == 0) or (val > limit) else val - 1

    if val != test_data[5]:
        print("atomicDec failed")
        return False

    found = False

    for i in range(length):
        # seventh element should be a member of [0, len)
        if i == test_data[6]:
            found = True
            break

    if not found:
        print("atomicCAS failed")
        return False

    val = 0xFF

    for i in range(length):
        # 8th element should be 1
        val &= 2 * i + 7

    if val != test_data[7]:
        print("atomicAnd failed")
        return False

    # 9th element should be 0xff
    val = -1
    if val != test_data[8]:
        print("atomicOr failed")
        return False

    val = 0xFF

    for i in range(length):
        # 11th element should be 0xff
        val ^= i

    if val != test_data[9]:
        print("atomicXor failed")
        return False

    return True


def main():
    import pytest

    if os.name == "nt":
        pytest.skip("Atomics not supported on Windows")

    # set device
    dev_id = find_cuda_device()
    device_prop = check_cuda_errors(cudart.cudaGetDeviceProperties(dev_id))

    if not device_prop.managedMemory:
        pytest.skip("Unified Memory not supported on this device")

    compute_mode = check_cuda_errors(
        cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeMode, dev_id)
    )
    if compute_mode == cudart.cudaComputeMode.cudaComputeModeProhibited:
        pytest.skip("This sample requires a device in either default or process exclusive mode")

    if device_prop.major < 6:
        pytest.skip("Requires a minimum CUDA compute 6.0 capability")

    num_threads = 256
    num_blocks = 64
    num_data = 10

    if device_prop.pageableMemoryAccess:
        print("CAN access pageable memory")
        atom_arr_h = (ctypes.c_int * num_data)(0)
        atom_arr = ctypes.addressof(atom_arr_h)
    else:
        print("CANNOT access pageable memory")
        atom_arr = check_cuda_errors(
            cudart.cudaMallocManaged(np.dtype(np.int32).itemsize * num_data, cudart.cudaMemAttachGlobal)
        )
        atom_arr_h = (ctypes.c_int * num_data).from_address(atom_arr)

    for i in range(num_data):
        atom_arr_h[i] = 0

    # To make the AND and XOR tests generate something other than 0...
    atom_arr_h[7] = atom_arr_h[9] = 0xFF

    kernel_helper = common.KernelHelper(system_wide_atomics, dev_id)
    _atomic_kernel = kernel_helper.get_function(b"atomicKernel")
    kernel_args = ((atom_arr,), (ctypes.c_void_p,))
    check_cuda_errors(
        cuda.cuLaunchKernel(
            _atomic_kernel,
            num_blocks,
            1,
            1,  # grid dim
            num_threads,
            1,
            1,  # block dim
            0,
            cuda.CU_STREAM_LEGACY,  # shared mem and stream
            kernel_args,
            0,
        )
    )  # arguments
    # NOTE: Python doesn't have an equivalent system atomic operations
    # atomicKernel_CPU(atom_arr_h, numBlocks * numThreads)

    check_cuda_errors(cudart.cudaDeviceSynchronize())

    # Compute & verify reference solution
    test_result = verify(atom_arr_h, num_threads * num_blocks)

    if device_prop.pageableMemoryAccess:
        pass
    else:
        check_cuda_errors(cudart.cudaFree(atom_arr))

    if not test_result:
        print("systemWideAtomics completed with errors", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
