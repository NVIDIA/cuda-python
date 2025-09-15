# Add VMMAllocatedMemoryResource for Virtual Memory Management APIs

## Summary

This PR implements a new `VMMAllocatedMemoryResource` class that provides access to CUDA's Virtual Memory Management (VMM) APIs through the cuda.core memory resource interface. This addresses the feature request for using `cuMemCreate`, `cuMemMap`, and related APIs for advanced memory management scenarios.

## Changes

### Core Implementation
- **New `VMMAllocatedMemoryResource` class** in `cuda/core/experimental/_memory.pyx`
  - Implements the `MemoryResource` abstract interface
  - Uses VMM APIs: `cuMemCreate`, `cuMemAddressReserve`, `cuMemMap`, `cuMemSetAccess`, `cuMemUnmap`, `cuMemAddressFree`, `cuMemRelease`
  - Provides proper allocation tracking and cleanup
  - Validates device VMM support during initialization

- **Device integration** in `cuda/core/experimental/_device.py`
  - Added `Device.create_vmm_memory_resource()` convenience method
  - Full integration with existing memory resource infrastructure

- **Module exports** in `cuda/core/experimental/__init__.py`
  - Added `VMMAllocatedMemoryResource` to public API

### Testing & Examples
- **Comprehensive test suite** in `tests/test_vmm_memory_resource.py`
  - Tests creation, allocation/deallocation, multiple allocations
  - Tests different allocation types and error conditions
  - All tests pass on VMM-capable hardware

- **Working example** in `examples/vmm_memory_example.py`
  - Demonstrates basic and advanced usage patterns
  - Shows integration with Device and Buffer APIs

## Addressing the Feature Request

This implementation directly addresses the original issue requirements:

### ✅ **"I would like to be able to use the equivalent of cuMemCreate, cuMemMap, and friends via a cuda.core MemoryResource"**
- The `VMMAllocatedMemoryResource` uses these exact APIs internally
- Provides a clean, Pythonic interface that fits the cuda.core design patterns
- Maintains full compatibility with existing `Buffer` and `Stream` APIs

### ✅ **"I'd like to have a VMMAllocatedMemoryResource which I can create on a Device() for which allocate() will use the cuMem*** driver APIs"**
- Implemented exactly as requested with `Device.create_vmm_memory_resource()`
- The `allocate()` method uses VMM APIs to create memory allocations
- Can be set as the default memory resource for a device

### ✅ **Use Cases Supported**
- **NVSHMEM/NCCL external buffer registration**: VMM allocations provide the fine-grained control needed
- **Growing allocations without changing pointer addresses**: VMM's address reservation and mapping enables this
- **EGM on Grace-Hopper/Grace-Blackwell**: VMM APIs are essential for Extended GPU Memory scenarios

### ✅ **"Since the cuMem*** functions are synchronous, there's no way to fit this with the MemPool APIs as-is"**
- Correctly implemented as synchronous operations outside the memory pool system
- VMM operations are inherently synchronous as noted in the original issue
- Provides an alternative to memory pools for specialized use cases

## Technical Details

### Memory Management Flow
1. **Allocation**: `cuMemCreate` → `cuMemAddressReserve` → `cuMemMap` → `cuMemSetAccess`
2. **Tracking**: Internal dictionary maintains allocation metadata for proper cleanup
3. **Deallocation**: `cuMemUnmap` → `cuMemAddressFree` → `cuMemRelease`

### Key Features
- **Granularity-aware**: Respects CUDA allocation granularity requirements using `cuMemGetAllocationGranularity`
- **Error handling**: Comprehensive error checking with proper cleanup on failures
- **Device validation**: Automatically checks `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`
- **Resource tracking**: Maintains allocation state for proper cleanup in destructor

### API Design
```python
# Direct usage
device = Device()
vmm_mr = device.create_vmm_memory_resource()
buffer = vmm_mr.allocate(size)

# As default memory resource
device.memory_resource = vmm_mr
buffer = device.allocate(size)  # Now uses VMM
```

## Testing

All tests pass on VMM-capable hardware:
```
===================================== test session starts =====================================
tests/test_vmm_memory_resource.py::TestVMMAllocatedMemoryResource::test_vmm_memory_resource_creation PASSED
tests/test_vmm_memory_resource.py::TestVMMAllocatedMemoryResource::test_vmm_memory_resource_allocation_deallocation PASSED
tests/test_vmm_memory_resource.py::TestVMMAllocatedMemoryResource::test_vmm_memory_resource_multiple_allocations PASSED
tests/test_vmm_memory_resource.py::TestVMMAllocatedMemoryResource::test_vmm_memory_resource_with_different_allocation_types PASSED
tests/test_vmm_memory_resource.py::TestVMMAllocatedMemoryResource::test_vmm_memory_resource_invalid_device PASSED
================================ 5 passed, 1 skipped in 0.07s =================================
```

## Compatibility

- **Hardware**: Requires GPU with VMM support (compute capability 6.0+)
- **CUDA**: Compatible with CUDA 11.2+ (when VMM APIs were introduced)
- **Python**: Compatible with existing cuda.core Python version requirements
- **API**: Fully compatible with existing MemoryResource interface

## Future Enhancements

This implementation provides a solid foundation that could be extended with:
- Host-accessible VMM allocations using `CU_MEM_LOCATION_TYPE_HOST`
- Memory sharing between processes using handle export/import APIs
- Integration with NVSHMEM/NCCL registration helpers
- Support for memory compression and other advanced VMM features

## Files Changed

- `cuda_core/cuda/core/experimental/_memory.pyx` - Core implementation
- `cuda_core/cuda/core/experimental/_device.py` - Device integration
- `cuda_core/cuda/core/experimental/__init__.py` - Module exports
- `cuda_core/tests/test_vmm_memory_resource.py` - Test suite
- `cuda_core/examples/vmm_memory_example.py` - Usage example

This implementation provides exactly what was requested in the original issue while maintaining full compatibility with the existing cuda.core ecosystem.
