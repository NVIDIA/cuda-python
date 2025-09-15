# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example demonstrating the VMMAllocatedMemoryResource for fine-grained memory management.

This example shows how to use CUDA's Virtual Memory Management APIs through the
VMMAllocatedMemoryResource class for advanced memory allocation scenarios.
"""

import sys

from cuda.core.experimental import Device, VMMAllocatedMemoryResource, Stream
from cuda.core.experimental._utils.cuda_utils import driver


def main():
    """Demonstrate VMMAllocatedMemoryResource usage."""
    try:
        # Get the default device
        device = Device()
        print(f"Using device {device.device_id}: {device.properties.name}")
        
        # Check if device supports virtual memory management
        err, vmm_supported = driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, 
            device.device_id
        )
        
        if err != driver.CUresult.CUDA_SUCCESS or not vmm_supported:
            print(f"Device {device.device_id} does not support virtual memory management.")
            print("This feature requires a modern GPU with compute capability 6.0 or higher.")
            sys.exit(1)
        
        print(f"Device {device.device_id} supports virtual memory management!")
        
        # Create a VMMAllocatedMemoryResource using the convenience method
        vmm_mr = device.create_vmm_memory_resource()
        print(f"Created VMMAllocatedMemoryResource for device {device.device_id}")
        
        # Optionally set it as the default memory resource for the device
        # device.memory_resource = vmm_mr
        
        # Create a stream for operations
        stream = Stream()
        
        # Allocate some memory using VMM
        sizes = [1024, 4096, 1024*1024]  # 1KB, 4KB, 1MB
        buffers = []
        
        print("\nAllocating buffers using VMM:")
        for i, size in enumerate(sizes):
            buffer = vmm_mr.allocate(size, stream)
            buffers.append(buffer)
            print(f"  Buffer {i+1}: {size:,} bytes at address 0x{int(buffer.handle):016x}")
            
            # Verify buffer properties
            assert buffer.is_device_accessible
            assert not buffer.is_host_accessible
            assert buffer.device_id == device.device_id
            assert buffer.memory_resource is vmm_mr
        
        # Demonstrate buffer copying
        if len(buffers) >= 2:
            print(f"\nCopying from buffer 1 to buffer 2...")
            # Note: In a real application, you would initialize buffer 1 with data first
            buffers[1].copy_from(buffers[0], stream=stream)
            stream.sync()  # Wait for copy to complete
            print("Copy completed!")
        
        # Clean up buffers
        print("\nCleaning up buffers:")
        for i, buffer in enumerate(buffers):
            buffer.close()
            print(f"  Buffer {i+1} deallocated")
        
        print("\nVMM memory management example completed successfully!")
        
        # Demonstrate advanced usage: custom allocation type
        print("\nDemonstrating custom allocation type:")
        try:
            # Create with managed memory type (if supported)
            vmm_mr_managed = device.create_vmm_memory_resource(
                driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_MANAGED
            )
            
            managed_buffer = vmm_mr_managed.allocate(4096, stream)
            print(f"  Managed buffer: 4096 bytes at address 0x{int(managed_buffer.handle):016x}")
            managed_buffer.close()
            print("  Managed buffer deallocated")
            
        except Exception as e:
            print(f"  Managed memory allocation failed: {e}")
            print("  This is expected on some systems/drivers")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
