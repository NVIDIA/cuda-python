# Sample: Stream-Ordered Memory Allocation (Python)

## Description

Demonstrates the raw stream-ordered allocation API in
``cuda.bindings.runtime``:

- ``cudaMallocAsync(nbytes, stream)`` /
  ``cudaFreeAsync(ptr, stream)`` — allocate and free on a stream, ordered
  with any surrounding kernels or copies
- ``cudaDeviceGetDefaultMemPool(dev)`` — grab the default memory pool
- ``cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, ...)`` —
  ask the pool to *retain* memory across frees instead of releasing it to
  the OS

Two demos are run back to back on a 1M-element vector add:

1. **Basic** — plain ``cudaMallocAsync`` / launch / ``cudaFreeAsync``. With
   the default release threshold of zero, the pool may release chunks back
   to the OS between iterations.
2. **Post-sync** — set the release threshold to ``UINT64_MAX`` so
   ``cudaFreeAsync`` keeps the pool "warm", then time the loop with CUDA
   events to show the steady-state cost.

This is the low-level counterpart to the high-level
[`samples/cuda_core/memoryResources/`](../../../cuda_core/memoryResources/) sample, whose
``DeviceMemoryResource`` sits on top of the same pool but hides the
attribute knobs.

Waives on Darwin (Metal-only) and on GPUs without memory-pool support.

## What You'll Learn

- Stream-ordered device memory: `cudaMallocAsync` and `cudaFreeAsync`
- Retrieving the device's default memory pool with `cudaDeviceGetDefaultMemPool`
- Tuning `cudaMemPoolAttrReleaseThreshold` for retain-across-frees behavior
- Timing GPU work with `cudaEventCreate` / `cudaEventRecord` / `cudaEventSynchronize` / `cudaEventElapsedTime`

## Key Libraries

- [`cuda.bindings`](https://nvidia.github.io/cuda-python/cuda-bindings/latest/) - driver + runtime bindings
- `numpy` - host arrays and reference computation

## Key APIs

### From `cuda.bindings.runtime`

- `cudaSetDevice`, `cudaStreamCreateWithFlags`, `cudaStreamDestroy`, `cudaStreamSynchronize`
- `cudaMallocAsync`, `cudaFreeAsync`, `cudaMemcpyAsync`
- `cudaDeviceGetDefaultMemPool`, `cudaMemPoolSetAttribute`, `cudaMemPoolAttr`
- `cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`

### From `cuda.bindings.driver`

- `cuLaunchKernel`
- `CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED`

## Requirements

### Hardware

- NVIDIA GPU that supports memory pools (all modern discrete GPUs since Volta)

### Software

- CUDA Toolkit 11.3 or newer (feature gate for `MEMORY_POOLS_SUPPORTED`)
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)
- `numpy`

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python streamOrderedAllocation.py
python streamOrderedAllocation.py --device=1
```

## Expected Output

Timings depend on GPU and CPU.

```
Starting basicStreamOrderedAllocation()
> Checking the results from vectorAddGPU() ...
Starting streamOrderedAllocationPostSync()
Total elapsed time = 12.345 ms over 20 iterations
> Checking the results from vectorAddGPU() ...
Both stream-ordered allocation variants verified.
Done
```

## Files

- `streamOrderedAllocation.py` - Python implementation using `cuda.bindings`
- `README.md` - This file
- `requirements.txt` - Sample dependencies
- `../../Utilities/cuda_bindings_utils.py` - Shared bindings helpers (imported by this sample)

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.bindings` runtime API](https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/runtime.html)
- [`samples/cuda_core/memoryResources/`](../../../cuda_core/memoryResources/) - the high-level `cuda.core` equivalent
- [CUDA Runtime API — Stream Ordered Memory Allocator](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html)
