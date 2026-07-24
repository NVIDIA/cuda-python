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
CUDA Virtual Memory Management (VMM) API

Vector add C = A + B, but each device buffer is built by hand out of
per-device physical backings stitched into a single contiguous virtual
address range. Uses the Virtual Memory Management API in
``cuda.bindings.driver``:

    cuMemAddressReserve  -> reserve a chunk of virtual address space
    cuMemGetAllocationGranularity -> query alignment requirements
    cuMemCreate          -> create a physical backing on a specific device
    cuMemMap             -> map the backing into the reserved VA range
    cuMemSetAccess       -> grant read/write access to the mapping device(s)
    cuMemUnmap / cuMemAddressFree / cuMemRelease -> tear everything down

This is the *only* sample that teaches the CUDA VMM API. The allocation
is striped across all peer-capable devices ("backing devices") but is
accessed by a single mapping device (the current device), which is a
useful pattern for NUMA-aware placement across a multi-GPU machine.

Waives on macOS and on 32-bit / ARM SBSA / aarch64 configurations that do
not support VMM, and on devices that do not report
``VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED``.
"""

import ctypes
import math
import platform
import sys
from pathlib import Path

# Add samples/cuda_bindings/Utilities/ to the import path for shared bindings helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Utilities"))

try:
    import numpy as np
    from cuda_bindings_utils import KernelHelper, check_cuda_errors, find_cuda_device_drv, requirement_not_met

    from cuda.bindings import driver as cuda
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


VECTOR_ADD_MMAP_KERNEL = """\
extern "C" __global__ void VecAdd_kernel(const float *A, const float *B, float *C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}
"""


def _round_up(x, y):
    return int((x - 1) / y + 1) * y


def _get_backing_devices(cu_device):
    """Return ``[cu_device] + <every peer that supports VMM>``."""
    num_devices = check_cuda_errors(cuda.cuDeviceGetCount())

    backing_devices = [cu_device]
    for dev in range(num_devices):
        if int(dev) == int(cu_device):
            continue

        # Only peer-capable devices can back a shared VA range together.
        capable = check_cuda_errors(cuda.cuDeviceCanAccessPeer(cu_device, dev))
        if not capable:
            continue

        # The device also needs to support VMM to participate.
        attribute_val = check_cuda_errors(
            cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
                cu_device,
            )
        )
        if attribute_val == 0:
            continue

        backing_devices.append(cuda.CUdevice(dev))
    return backing_devices


def _simple_malloc_multi_device_mmap(size, resident_devices, mapping_devices, align=0):
    """Allocate ``size`` bytes striped across ``resident_devices`` and mapped
    into a single contiguous VA range accessible by ``mapping_devices``.

    Returns ``(status, dptr, allocation_size)``. ``allocation_size`` is the
    rounded-up size (caller uses it for the matching free).
    """
    min_granularity = 0

    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE

    # The stripe size must satisfy each participant's granularity.
    for device in resident_devices:
        prop.location.id = device
        status, granularity = cuda.cuMemGetAllocationGranularity(
            prop, cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )
        if status != cuda.CUresult.CUDA_SUCCESS:
            return status, None, None
        min_granularity = max(min_granularity, granularity)

    for device in mapping_devices:
        prop.location.id = device
        status, granularity = cuda.cuMemGetAllocationGranularity(
            prop, cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )
        if status != cuda.CUresult.CUDA_SUCCESS:
            return status, None, None
        min_granularity = max(min_granularity, granularity)

    # Round up so ``size`` splits evenly into stripes that each meet the granularity requirement.
    size = _round_up(size, len(resident_devices) * min_granularity)
    stripe_size = size // len(resident_devices)
    allocation_size = size

    # Reserve one contiguous VA range for the whole allocation.
    status, dptr = cuda.cuMemAddressReserve(size, align, cuda.CUdeviceptr(0), 0)
    if status != cuda.CUresult.CUDA_SUCCESS:
        _simple_free_multi_device_mmap(dptr, size)
        return status, None, None

    # For each backing device, create a physical allocation and map it into the reserved VA range.
    for idx in range(len(resident_devices)):
        prop.location.id = resident_devices[idx]

        status, allocation_handle = cuda.cuMemCreate(stripe_size, prop, 0)
        if status != cuda.CUresult.CUDA_SUCCESS:
            _simple_free_multi_device_mmap(dptr, size)
            return status, None, None

        # After cuMemMap, the physical handle can be released; the mapping
        # keeps the backing alive until the VA range is unmapped.
        (status,) = cuda.cuMemMap(int(dptr) + (stripe_size * idx), stripe_size, 0, allocation_handle, 0)
        (status2,) = cuda.cuMemRelease(allocation_handle)
        if status != cuda.CUresult.CUDA_SUCCESS:
            status = status2
        if status != cuda.CUresult.CUDA_SUCCESS:
            _simple_free_multi_device_mmap(dptr, size)
            return status, None, None

    # Grant each mapping device read/write access to the whole range.
    access_descriptors = [cuda.CUmemAccessDesc()] * len(mapping_devices)
    for idx in range(len(mapping_devices)):
        access_descriptors[idx].location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access_descriptors[idx].location.id = mapping_devices[idx]
        access_descriptors[idx].flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

    (status,) = cuda.cuMemSetAccess(dptr, size, access_descriptors, len(access_descriptors))
    if status != cuda.CUresult.CUDA_SUCCESS:
        _simple_free_multi_device_mmap(dptr, size)
        return status, None, None

    return (status, dptr, allocation_size)


def _simple_free_multi_device_mmap(dptr, size):
    status = cuda.cuMemUnmap(dptr, size)
    if status[0] != cuda.CUresult.CUDA_SUCCESS:
        return status
    status = cuda.cuMemAddressFree(dptr, size)
    if status[0] != cuda.CUresult.CUDA_SUCCESS:
        return status
    return status


def main():
    if platform.system() == "Darwin":
        requirement_not_met("vectorAddMmap is not supported on Mac OSX")
    if platform.machine() in {"armv7l", "aarch64", "sbsa"}:
        requirement_not_met(f"vectorAddMmap is not supported on {platform.machine()}")

    n = 50000
    size = n * np.dtype(np.float32).itemsize

    check_cuda_errors(cuda.cuInit(0))
    cu_device = find_cuda_device_drv()

    attribute_val = check_cuda_errors(
        cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
            cu_device,
        )
    )
    print(f"Device {cu_device} VIRTUAL ADDRESS MANAGEMENT SUPPORTED = {attribute_val}.")
    if not attribute_val:
        requirement_not_met(f"Device {cu_device} does not support Virtual Address Management")

    # The kernel launches on cu_device; the allocation stripes across every
    # peer-capable device that also supports VMM.
    mapping_devices = [cu_device]
    backing_devices = _get_backing_devices(cu_device)

    cu_context = check_cuda_errors(cuda.cuCtxCreate(None, 0, cu_device))
    kernel_helper = KernelHelper(VECTOR_ADD_MMAP_KERNEL, int(cu_device))
    vec_add_kernel = kernel_helper.get_function(b"VecAdd_kernel")

    h_a = np.random.rand(n).astype(dtype=np.float32)
    h_b = np.random.rand(n).astype(dtype=np.float32)
    h_c = np.zeros(n, dtype=np.float32)

    # ---- Allocate device memory via VMM ----
    # (allocation_size might be rounded up above `size` for granularity.)
    d_a, allocation_size = check_cuda_errors(_simple_malloc_multi_device_mmap(size, backing_devices, mapping_devices))
    d_b, _ = check_cuda_errors(_simple_malloc_multi_device_mmap(size, backing_devices, mapping_devices))
    d_c, _ = check_cuda_errors(_simple_malloc_multi_device_mmap(size, backing_devices, mapping_devices))

    check_cuda_errors(cuda.cuMemcpyHtoD(d_a, h_a, size))
    check_cuda_errors(cuda.cuMemcpyHtoD(d_b, h_b, size))

    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    kernel_args = ((d_a, d_b, d_c, n), (None, None, None, ctypes.c_int))
    check_cuda_errors(
        cuda.cuLaunchKernel(
            vec_add_kernel,
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

    check_cuda_errors(cuda.cuMemcpyDtoH(h_c, d_c, size))

    max_err = 0.0
    for i in range(n):
        max_err = max(max_err, math.fabs(h_c[i] - (h_a[i] + h_b[i])))

    check_cuda_errors(_simple_free_multi_device_mmap(d_a, allocation_size))
    check_cuda_errors(_simple_free_multi_device_mmap(d_b, allocation_size))
    check_cuda_errors(_simple_free_multi_device_mmap(d_c, allocation_size))
    check_cuda_errors(cuda.cuCtxDestroy(cu_context))

    if max_err > 1e-5:
        print(f"Result = FAIL (max error {max_err})", file=sys.stderr)
        return 1

    print(f"Result = PASS (max error {max_err:.3e} over {n} elements, striped across {len(backing_devices)} device(s))")
    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
