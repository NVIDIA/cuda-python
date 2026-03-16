# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import math
import platform
import sys

import numpy as np
from common import common
from common.helper_cuda import check_cuda_errors, find_cuda_device_drv

from cuda.bindings import driver as cuda

vector_add_mmap = """\
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


def round_up(x, y):
    return int((x - 1) / y + 1) * y


def get_backing_devices(cu_device):
    num_devices = check_cuda_errors(cuda.cuDeviceGetCount())

    backing_devices = [cu_device]
    for dev in range(num_devices):
        # The mapping device is already in the backingDevices vector
        if int(dev) == int(cu_device):
            continue

        # Only peer capable devices can map each others memory
        capable = check_cuda_errors(cuda.cuDeviceCanAccessPeer(cu_device, dev))
        if not capable:
            continue

        # The device needs to support virtual address management for the required apis to work
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


def simple_malloc_multi_device_mmap(size, resident_devices, mapping_devices, align=0):
    min_granularity = 0

    # Setup the properties common for all the chunks
    # The allocations will be device pinned memory.
    # This property structure describes the physical location where the memory will be allocated via cuMemCreate allong with additional properties
    # In this case, the allocation will be pinnded device memory local to a given device.
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE

    # Get the minimum granularity needed for the resident devices
    # (the max of the minimum granularity of each participating device)
    for device in resident_devices:
        prop.location.id = device
        status, granularity = cuda.cuMemGetAllocationGranularity(
            prop, cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )
        if status != cuda.CUresult.CUDA_SUCCESS:
            return status, None, None
        if min_granularity < granularity:
            min_granularity = granularity

    # Get the minimum granularity needed for the accessing devices
    # (the max of the minimum granularity of each participating device)
    for device in mapping_devices:
        prop.location.id = device
        status, granularity = cuda.cuMemGetAllocationGranularity(
            prop, cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )
        if status != cuda.CUresult.CUDA_SUCCESS:
            return status, None, None
        if min_granularity < granularity:
            min_granularity = granularity

    # Round up the size such that we can evenly split it into a stripe size tha meets the granularity requirements
    # Essentially size = N * residentDevices.size() * min_granularity is the requirement,
    # since each piece of the allocation will be stripeSize = N * min_granularity
    # and the min_granularity requirement applies to each stripeSize piece of the allocation.
    size = round_up(size, len(resident_devices) * min_granularity)
    stripe_size = size / len(resident_devices)

    # Return the rounded up size to the caller for use in the free
    allocation_size = size

    # Reserve the required contiguous VA space for the allocations
    status, dptr = cuda.cuMemAddressReserve(size, align, cuda.CUdeviceptr(0), 0)
    if status != cuda.CUresult.CUDA_SUCCESS:
        simple_free_multi_device_mmap(dptr, size)
        return status, None, None

    # Create and map the backings on each gpu
    # note: reusing CUmemAllocationProp prop from earlier with prop.type & prop.location.type already specified.
    for idx in range(len(resident_devices)):
        # Set the location for this chunk to this device
        prop.location.id = resident_devices[idx]

        # Create the allocation as a pinned allocation on this device
        status, allocation_handle = cuda.cuMemCreate(stripe_size, prop, 0)
        if status != cuda.CUresult.CUDA_SUCCESS:
            simple_free_multi_device_mmap(dptr, size)
            return status, None, None

        # Assign the chunk to the appropriate VA range and release the handle.
        # After mapping the memory, it can be referenced by virtual address.
        # Since we do not need to make any other mappings of this memory or export it,
        # we no longer need and can release the allocationHandle.
        # The allocation will be kept live until it is unmapped.
        (status,) = cuda.cuMemMap(int(dptr) + (stripe_size * idx), stripe_size, 0, allocation_handle, 0)

        # the handle needs to be released even if the mapping failed.
        (status2,) = cuda.cuMemRelease(allocation_handle)
        if status != cuda.CUresult.CUDA_SUCCESS:
            # cuMemRelease should not have failed here
            # as the handle was just allocated successfully
            # however return an error if it does.
            status = status2

        # Cleanup in case of any mapping failures.
        if status != cuda.CUresult.CUDA_SUCCESS:
            simple_free_multi_device_mmap(dptr, size)
            return status, None, None

    # Each accessDescriptor will describe the mapping requirement for a single device
    access_descriptors = [cuda.CUmemAccessDesc()] * len(mapping_devices)

    # Prepare the access descriptor array indicating where and how the backings should be visible.
    for idx in range(len(mapping_devices)):
        # Specify which device we are adding mappings for.
        access_descriptors[idx].location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access_descriptors[idx].location.id = mapping_devices[idx]

        # Specify both read and write access.
        access_descriptors[idx].flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

    # Apply the access descriptors to the whole VA range.
    (status,) = cuda.cuMemSetAccess(dptr, size, access_descriptors, len(access_descriptors))
    if status != cuda.CUresult.CUDA_SUCCESS:
        simple_free_multi_device_mmap(dptr, size)
        return status, None, None

    return (status, dptr, allocation_size)


def simple_free_multi_device_mmap(dptr, size):
    # Unmap the mapped virtual memory region
    # Since the handles to the mapped backing stores have already been released
    # by cuMemRelease, and these are the only/last mappings referencing them,
    # The backing stores will be freed.
    # Since the memory has been unmapped after this call, accessing the specified
    # va range will result in a fault (unitll it is remapped).
    status = cuda.cuMemUnmap(dptr, size)
    if status[0] != cuda.CUresult.CUDA_SUCCESS:
        return status

    # Free the virtual address region.  This allows the virtual address region
    # to be reused by future cuMemAddressReserve calls.  This also allows the
    # virtual address region to be used by other allocation made through
    # opperating system calls like malloc & mmap.
    status = cuda.cuMemAddressFree(dptr, size)
    if status[0] != cuda.CUresult.CUDA_SUCCESS:
        return status
    return status


def main():
    import pytest

    if platform.system() == "Darwin":
        pytest.skip("vectorAddMMAP is not supported on Mac OSX")

    if platform.machine() == "armv7l":
        pytest.skip("vectorAddMMAP is not supported on ARMv7")

    if platform.machine() == "aarch64":
        pytest.skip("vectorAddMMAP is not supported on aarch64")

    if platform.machine() == "sbsa":
        pytest.skip("vectorAddMMAP is not supported on sbsa")

    n = 50000
    size = n * np.dtype(np.float32).itemsize

    # Initialize
    check_cuda_errors(cuda.cuInit(0))

    cu_device = find_cuda_device_drv()

    # Check that the selected device supports virtual address management
    attribute_val = check_cuda_errors(
        cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
            cu_device,
        )
    )
    print(f"Device {cu_device} VIRTUAL ADDRESS MANAGEMENT SUPPORTED = {attribute_val}.")
    if not attribute_val:
        pytest.skip(f"Device {cu_device} doesn't support VIRTUAL ADDRESS MANAGEMENT.")

    # The vector addition happens on cuDevice, so the allocations need to be mapped there.
    mapping_devices = [cu_device]

    # Collect devices accessible by the mapping device (cuDevice) into the backingDevices vector.
    backing_devices = get_backing_devices(cu_device)

    # Create context
    cu_context = check_cuda_errors(cuda.cuCtxCreate(None, 0, cu_device))

    kernel_helper = common.KernelHelper(vector_add_mmap, int(cu_device))
    _vec_add_kernel = kernel_helper.get_function(b"VecAdd_kernel")

    # Allocate input vectors h_A and h_B in host memory
    h_a = np.random.rand(size).astype(dtype=np.float32)
    h_b = np.random.rand(size).astype(dtype=np.float32)
    h_c = np.random.rand(size).astype(dtype=np.float32)

    # Allocate vectors in device memory
    # note that a call to cuCtxEnablePeerAccess is not needed even though
    # the backing devices and mapping device are not the same.
    # This is because the cuMemSetAccess call explicitly specifies
    # the cross device mapping.
    # cuMemSetAccess is still subject to the constraints of cuDeviceCanAccessPeer
    # for cross device mappings (hence why we checked cuDeviceCanAccessPeer earlier).
    d_a, allocation_size = check_cuda_errors(simple_malloc_multi_device_mmap(size, backing_devices, mapping_devices))
    d_b, _ = check_cuda_errors(simple_malloc_multi_device_mmap(size, backing_devices, mapping_devices))
    d_c, _ = check_cuda_errors(simple_malloc_multi_device_mmap(size, backing_devices, mapping_devices))

    # Copy vectors from host memory to device memory
    check_cuda_errors(cuda.cuMemcpyHtoD(d_a, h_a, size))
    check_cuda_errors(cuda.cuMemcpyHtoD(d_b, h_b, size))

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

    # Copy result from device memory to host memory
    # h_C contains the result in host memory
    check_cuda_errors(cuda.cuMemcpyDtoH(h_c, d_c, size))

    # Verify result
    for i in range(n):
        sum_all = h_a[i] + h_b[i]
        if math.fabs(h_c[i] - sum_all) > 1e-7:
            break

    check_cuda_errors(simple_free_multi_device_mmap(d_a, allocation_size))
    check_cuda_errors(simple_free_multi_device_mmap(d_b, allocation_size))
    check_cuda_errors(simple_free_multi_device_mmap(d_c, allocation_size))

    check_cuda_errors(cuda.cuCtxDestroy(cu_context))

    if i + 1 != n:
        print("Result = FAIL", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
