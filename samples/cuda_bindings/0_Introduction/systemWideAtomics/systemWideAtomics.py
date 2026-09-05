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
# dependencies = ["cuda-python>=13.4.0", "numpy>=1.24"]
# ///

"""
System-wide atomics on managed (unified) memory

Exercises the full ``*_system`` atomic API surface on ``cudaMallocManaged``
memory, which is coherently accessible from both the CPU and the GPU:

    atomicAdd_system, atomicExch_system, atomicMax_system, atomicMin_system,
    atomicInc_system, atomicDec_system, atomicCAS_system,
    atomicAnd_system,  atomicOr_system,   atomicXor_system

The kernel spins over ``LOOP_NUM`` iterations per thread, applying each
atomic against a 10-element shared array in managed memory. After the
kernel completes, the host runs the equivalent scalar reference and
verifies every slot.

Waives with exit code 2 when:

  * running on Windows (system-scope atomics aren't supported there for
    this managed-memory flavor), or
  * the device does not report Unified Memory support, or
  * compute mode is prohibited, or
  * compute capability is below 6.0 (the minimum for these intrinsics).
"""

import ctypes
import os
import sys
from pathlib import Path

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


LOOP_NUM = 50


SYSTEM_WIDE_ATOMICS_KERNEL = """\
#define LOOP_NUM 50

extern "C"
__global__ void atomicKernel(int *atom_arr) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = 0; i < LOOP_NUM; i++) {
        atomicAdd_system (&atom_arr[0], 10);
        atomicExch_system(&atom_arr[1], tid);
        atomicMax_system (&atom_arr[2], tid);
        atomicMin_system (&atom_arr[3], tid);
        atomicInc_system ((unsigned int *)&atom_arr[4], 17);
        atomicDec_system ((unsigned int *)&atom_arr[5], 137);
        atomicCAS_system (&atom_arr[6], tid - 1, tid);

        // Bitwise atomic instructions
        atomicAnd_system(&atom_arr[7], 2 * tid + 7);
        atomicOr_system (&atom_arr[8], 1 << tid);
        atomicXor_system(&atom_arr[9], tid);
    }
}
"""


def _verify(test_data, length):
    """Host reference computation, one atomic at a time."""
    # atomicAdd
    val = 10 * length * LOOP_NUM
    if val != test_data[0]:
        print(f"atomicAdd failed val={val} test_data={test_data[0]}")
        return False

    # atomicExch: some tid in [0, length)
    if not (0 <= test_data[1] < length):
        print("atomicExch failed")
        return False

    # atomicMax: length - 1
    if length - 1 != test_data[2]:
        print("atomicMax failed")
        return False

    # atomicMin: 0
    if test_data[3] != 0:
        print("atomicMin failed")
        return False

    # atomicInc modulo 17
    limit = 17
    val = 0
    for _ in range(length * LOOP_NUM):
        val = 0 if val >= limit else val + 1
    if val != test_data[4]:
        print("atomicInc failed")
        return False

    # atomicDec modulo 137
    limit = 137
    val = 0
    for _ in range(length * LOOP_NUM):
        val = limit if (val == 0) or (val > limit) else val - 1
    if val != test_data[5]:
        print("atomicDec failed")
        return False

    # atomicCAS: some tid in [0, length)
    if not (0 <= test_data[6] < length):
        print("atomicCAS failed")
        return False

    # atomicAnd against 0xff
    val = 0xFF
    for i in range(length):
        val &= 2 * i + 7
    if val != test_data[7]:
        print("atomicAnd failed")
        return False

    # atomicOr: filled with 1s
    if test_data[8] != -1:
        print("atomicOr failed")
        return False

    # atomicXor against 0xff
    val = 0xFF
    for i in range(length):
        val ^= i
    if val != test_data[9]:
        print("atomicXor failed")
        return False

    return True


def main():
    if os.name == "nt":
        requirement_not_met("System-wide atomics on managed memory are not supported on Windows")

    dev_id = find_cuda_device()
    device_prop = check_cuda_errors(cudart.cudaGetDeviceProperties(dev_id))

    if not device_prop.managedMemory:
        requirement_not_met("Unified Memory not supported on this device")

    compute_mode = check_cuda_errors(
        cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeMode, dev_id)
    )
    if compute_mode == cudart.cudaComputeMode.cudaComputeModeProhibited:
        requirement_not_met("This sample requires a device in either default or process-exclusive compute mode")

    if device_prop.major < 6:
        requirement_not_met("Requires a minimum CUDA compute capability 6.0")

    num_threads = 256
    num_blocks = 64
    num_data = 10

    # Prefer pageable memory when the driver supports it; otherwise fall back
    # to explicit cudaMallocManaged. Either way we hand the kernel a plain
    # int* whose contents are coherently visible from both the host and the
    # device.
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

    # Prime the AND / XOR seats with 0xff so the operations produce something
    # other than zero and are easy to verify against a reference.
    atom_arr_h[7] = atom_arr_h[9] = 0xFF

    kernel_helper = KernelHelper(SYSTEM_WIDE_ATOMICS_KERNEL, dev_id)
    atomic_kernel = kernel_helper.get_function(b"atomicKernel")
    kernel_args = ((atom_arr,), (ctypes.c_void_p,))
    check_cuda_errors(
        cuda.cuLaunchKernel(
            atomic_kernel,
            num_blocks,
            1,
            1,  # grid dim
            num_threads,
            1,
            1,  # block dim
            0,
            cuda.CU_STREAM_LEGACY,
            kernel_args,
            0,
        )
    )
    check_cuda_errors(cudart.cudaDeviceSynchronize())

    ok = _verify(atom_arr_h, num_threads * num_blocks)

    if not device_prop.pageableMemoryAccess:
        check_cuda_errors(cudart.cudaFree(atom_arr))

    if not ok:
        print("systemWideAtomics completed with errors", file=sys.stderr)
        return 1

    print("systemWideAtomics: all 10 system-scope atomic operations verified")
    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
