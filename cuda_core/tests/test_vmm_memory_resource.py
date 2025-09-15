# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.core.experimental import Device, VMMAllocatedMemoryResource
from cuda.core.experimental._utils.cuda_utils import driver


class TestVMMAllocatedMemoryResource:
    def test_vmm_memory_resource_creation(self):
        """Test creating a VMMAllocatedMemoryResource."""
        device = Device()
        
        # Check if device supports VMM
        err, vmm_supported = driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, 
            device.device_id
        )
        if err != driver.CUresult.CUDA_SUCCESS or not vmm_supported:
            pytest.skip("Device does not support virtual memory management")
        
        mr = device.create_vmm_memory_resource()
        
        assert mr.device_id == device.device_id
        assert mr.is_device_accessible is True
        assert mr.is_host_accessible is False

    def test_vmm_memory_resource_allocation_deallocation(self):
        """Test allocating and deallocating memory with VMMAllocatedMemoryResource."""
        device = Device()
        
        # Check if device supports VMM
        err, vmm_supported = driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, 
            device.device_id
        )
        if err != driver.CUresult.CUDA_SUCCESS or not vmm_supported:
            pytest.skip("Device does not support virtual memory management")
        
        mr = device.create_vmm_memory_resource()
        
        # Test allocation
        size = 1024 * 1024  # 1 MB
        buffer = mr.allocate(size)
        
        assert buffer.size == size
        assert buffer.memory_resource is mr
        assert buffer.is_device_accessible is True
        assert buffer.is_host_accessible is False
        assert buffer.device_id == device.device_id
        
        # Test deallocation
        buffer.close()
        
        # Verify the buffer is closed
        assert buffer.handle is None

    def test_vmm_memory_resource_multiple_allocations(self):
        """Test multiple allocations with VMMAllocatedMemoryResource."""
        device = Device()
        
        # Check if device supports VMM
        err, vmm_supported = driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, 
            device.device_id
        )
        if err != driver.CUresult.CUDA_SUCCESS or not vmm_supported:
            pytest.skip("Device does not support virtual memory management")
        
        mr = device.create_vmm_memory_resource()
        
        # Allocate multiple buffers
        buffers = []
        for i in range(5):
            size = (i + 1) * 1024  # Different sizes
            buffer = mr.allocate(size)
            buffers.append(buffer)
            
            assert buffer.size == size
            assert buffer.memory_resource is mr
        
        # Deallocate all buffers
        for buffer in buffers:
            buffer.close()

    def test_vmm_memory_resource_with_different_allocation_types(self):
        """Test VMMAllocatedMemoryResource with different allocation types."""
        device = Device()
        
        # Check if device supports VMM
        err, vmm_supported = driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, 
            device.device_id
        )
        if err != driver.CUresult.CUDA_SUCCESS or not vmm_supported:
            pytest.skip("Device does not support virtual memory management")
        
        # Test with pinned allocation type (default)
        mr_pinned = device.create_vmm_memory_resource(
            driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        )
        
        buffer = mr_pinned.allocate(1024)
        assert buffer.size == 1024
        buffer.close()

    def test_vmm_memory_resource_invalid_device(self):
        """Test VMMAllocatedMemoryResource creation with invalid device."""
        # This should raise an error for an invalid device ID
        with pytest.raises((ValueError, RuntimeError, Exception)):  # Accept any exception for invalid device
            invalid_device = Device(0)  # Get a valid device first
            invalid_device._id = 999  # Hack to test invalid device
            invalid_device.create_vmm_memory_resource()

    def test_vmm_memory_resource_deallocate_untracked_pointer(self):
        """Test deallocating a pointer that wasn't allocated by this resource."""
        device = Device()
        
        # Check if device supports VMM
        err, vmm_supported = driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, 
            device.device_id
        )
        if err != driver.CUresult.CUDA_SUCCESS or not vmm_supported:
            pytest.skip("Device does not support virtual memory management")
        
        mr = device.create_vmm_memory_resource()
        
        # Try to deallocate a fake pointer
        with pytest.raises(ValueError, match="was not allocated by this memory resource"):
            mr.deallocate(0x12345678, 1024)
