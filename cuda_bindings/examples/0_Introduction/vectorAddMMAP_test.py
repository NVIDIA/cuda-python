# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import math
import platform
import sys

import numpy as np
from common import common
from common.helper_cuda import checkCudaErrors, findCudaDeviceDRV
from cuda.bindings import driver as cuda

vectorAddMMAP = """\
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


def getBackingDevices(cuDevice):
    num_devices = checkCudaErrors(cuda.cuDeviceGetCount())

    backingDevices = [cuDevice]
    for dev in range(num_devices):
        # The mapping device is already in the backingDevices vector
        if int(dev) == int(cuDevice):
            continue

        # Only peer capable devices can map each others memory
        capable = checkCudaErrors(cuda.cuDeviceCanAccessPeer(cuDevice, dev))
        if not capable:
            continue

        # The device needs to support virtual address management for the required apis to work
        attributeVal = checkCudaErrors(
            cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
                cuDevice,
            )
        )
        if attributeVal == 0:
            continue

        backingDevices.append(cuda.CUdevice(dev))
    return backingDevices


def simpleMallocMultiDeviceMmap(size, residentDevices, mappingDevices, align=0):
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
    for device in residentDevices:
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
    for device in mappingDevices:
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
    size = round_up(size, len(residentDevices) * min_granularity)
    stripeSize = size / len(residentDevices)

    # Return the rounded up size to the caller for use in the free
    allocationSize = size

    # Reserve the required contiguous VA space for the allocations
    status, dptr = cuda.cuMemAddressReserve(size, align, cuda.CUdeviceptr(0), 0)
    if status != cuda.CUresult.CUDA_SUCCESS:
        simpleFreeMultiDeviceMmap(dptr, size)
        return status, None, None

    # Create and map the backings on each gpu
    # note: reusing CUmemAllocationProp prop from earlier with prop.type & prop.location.type already specified.
    for idx in range(len(residentDevices)):
        # Set the location for this chunk to this device
        prop.location.id = residentDevices[idx]

        # Create the allocation as a pinned allocation on this device
        status, allocationHandle = cuda.cuMemCreate(stripeSize, prop, 0)
        if status != cuda.CUresult.CUDA_SUCCESS:
            simpleFreeMultiDeviceMmap(dptr, size)
            return status, None, None

        # Assign the chunk to the appropriate VA range and release the handle.
        # After mapping the memory, it can be referenced by virtual address.
        # Since we do not need to make any other mappings of this memory or export it,
        # we no longer need and can release the allocationHandle.
        # The allocation will be kept live until it is unmapped.
        (status,) = cuda.cuMemMap(int(dptr) + (stripeSize * idx), stripeSize, 0, allocationHandle, 0)

        # the handle needs to be released even if the mapping failed.
        (status2,) = cuda.cuMemRelease(allocationHandle)
        if status != cuda.CUresult.CUDA_SUCCESS:
            # cuMemRelease should not have failed here
            # as the handle was just allocated successfully
            # however return an error if it does.
            status = status2

        # Cleanup in case of any mapping failures.
        if status != cuda.CUresult.CUDA_SUCCESS:
            simpleFreeMultiDeviceMmap(dptr, size)
            return status, None, None

    # Each accessDescriptor will describe the mapping requirement for a single device
    accessDescriptors = [cuda.CUmemAccessDesc()] * len(mappingDevices)

    # Prepare the access descriptor array indicating where and how the backings should be visible.
    for idx in range(len(mappingDevices)):
        # Specify which device we are adding mappings for.
        accessDescriptors[idx].location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        accessDescriptors[idx].location.id = mappingDevices[idx]

        # Specify both read and write access.
        accessDescriptors[idx].flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

    # Apply the access descriptors to the whole VA range.
    (status,) = cuda.cuMemSetAccess(dptr, size, accessDescriptors, len(accessDescriptors))
    if status != cuda.CUresult.CUDA_SUCCESS:
        simpleFreeMultiDeviceMmap(dptr, size)
        return status, None, None

    return (status, dptr, allocationSize)


def simpleFreeMultiDeviceMmap(dptr, size):
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
    print("Vector Addition (Driver API)")

    if platform.system() == "Darwin":
        print("vectorAddMMAP is not supported on Mac OSX - waiving sample")
        return

    if platform.machine() == "armv7l":
        print("vectorAddMMAP is not supported on ARMv7 - waiving sample")
        return

    if platform.machine() == "aarch64":
        print("vectorAddMMAP is not supported on aarch64 - waiving sample")
        return

    if platform.machine() == "sbsa":
        print("vectorAddMMAP is not supported on sbsa - waiving sample")
        return

    N = 50000
    size = N * np.dtype(np.float32).itemsize

    # Initialize
    checkCudaErrors(cuda.cuInit(0))

    cuDevice = findCudaDeviceDRV()

    # Check that the selected device supports virtual address management
    attributeVal = checkCudaErrors(
        cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
            cuDevice,
        )
    )
    print(f"Device {cuDevice} VIRTUAL ADDRESS MANAGEMENT SUPPORTED = {attributeVal}.")
    if not attributeVal:
        print(f"Device {cuDevice} doesn't support VIRTUAL ADDRESS MANAGEMENT.")
        return

    # The vector addition happens on cuDevice, so the allocations need to be mapped there.
    mappingDevices = [cuDevice]

    # Collect devices accessible by the mapping device (cuDevice) into the backingDevices vector.
    backingDevices = getBackingDevices(cuDevice)

    # Create context
    cuContext = checkCudaErrors(cuda.cuCtxCreate(None, 0, cuDevice))

    kernelHelper = common.KernelHelper(vectorAddMMAP, int(cuDevice))
    _VecAdd_kernel = kernelHelper.getFunction(b"VecAdd_kernel")

    # Allocate input vectors h_A and h_B in host memory
    h_A = np.random.rand(size).astype(dtype=np.float32)
    h_B = np.random.rand(size).astype(dtype=np.float32)
    h_C = np.random.rand(size).astype(dtype=np.float32)

    # Allocate vectors in device memory
    # note that a call to cuCtxEnablePeerAccess is not needed even though
    # the backing devices and mapping device are not the same.
    # This is because the cuMemSetAccess call explicitly specifies
    # the cross device mapping.
    # cuMemSetAccess is still subject to the constraints of cuDeviceCanAccessPeer
    # for cross device mappings (hence why we checked cuDeviceCanAccessPeer earlier).
    d_A, allocationSize = checkCudaErrors(simpleMallocMultiDeviceMmap(size, backingDevices, mappingDevices))
    d_B, _ = checkCudaErrors(simpleMallocMultiDeviceMmap(size, backingDevices, mappingDevices))
    d_C, _ = checkCudaErrors(simpleMallocMultiDeviceMmap(size, backingDevices, mappingDevices))

    # Copy vectors from host memory to device memory
    checkCudaErrors(cuda.cuMemcpyHtoD(d_A, h_A, size))
    checkCudaErrors(cuda.cuMemcpyHtoD(d_B, h_B, size))

    # Grid/Block configuration
    threadsPerBlock = 256
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock

    kernelArgs = ((d_A, d_B, d_C, N), (None, None, None, ctypes.c_int))

    # Launch the CUDA kernel
    checkCudaErrors(
        cuda.cuLaunchKernel(
            _VecAdd_kernel,
            blocksPerGrid,
            1,
            1,
            threadsPerBlock,
            1,
            1,
            0,
            0,
            kernelArgs,
            0,
        )
    )

    # Copy result from device memory to host memory
    # h_C contains the result in host memory
    checkCudaErrors(cuda.cuMemcpyDtoH(h_C, d_C, size))

    # Verify result
    for i in range(N):
        sum_all = h_A[i] + h_B[i]
        if math.fabs(h_C[i] - sum_all) > 1e-7:
            break

    checkCudaErrors(simpleFreeMultiDeviceMmap(d_A, allocationSize))
    checkCudaErrors(simpleFreeMultiDeviceMmap(d_B, allocationSize))
    checkCudaErrors(simpleFreeMultiDeviceMmap(d_C, allocationSize))

    checkCudaErrors(cuda.cuCtxDestroy(cuContext))

    print("{}".format("Result = PASS" if i + 1 == N else "Result = FAIL"))
    if i + 1 != N:
        sys.exit(-1)


if __name__ == "__main__":
    main()
