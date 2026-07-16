# Sample: System-wide Atomics on Managed Memory (Python)

## Description

Exercises the full ``*_system`` atomic API surface on ``cudaMallocManaged``
memory, which is coherently accessible from both the CPU and the GPU:

- `atomicAdd_system`
- `atomicExch_system`
- `atomicMax_system` / `atomicMin_system`
- `atomicInc_system` / `atomicDec_system`
- `atomicCAS_system`
- `atomicAnd_system` / `atomicOr_system` / `atomicXor_system`

The kernel spins over `LOOP_NUM` iterations per thread, applying each
operation against a 10-element shared array. The host then runs the
equivalent scalar reference computation and verifies every slot.

This is the only sample in `/samples` that teaches **system-wide** atomic
operations (as opposed to device-scope ones like the histogram sample) or
the full atomic surface (Exch / Min / Max / Inc / Dec / CAS / And / Or / Xor).

Waives with exit code 2 when:

- running on Windows (system-scope atomics on managed memory aren't
  supported for this flavor there), or
- the device does not report Unified Memory support, or
- compute mode is prohibited, or
- Compute Capability is below 6.0 (minimum for these intrinsics).

## What You'll Learn

- Using every `atomic*_system` intrinsic in a single kernel
- Managed memory basics: coherent host / device access without explicit copies
- The two paths for host-visible memory in NVML-modern drivers:
  - Passing a pageable host allocation directly when
    `pageableMemoryAccess=True` is reported on the device
  - Falling back to `cudaMallocManaged` on hardware without pageable access
- Guarding a sample on compute capability, compute mode, and platform

## Key Libraries

- [`cuda.bindings`](https://nvidia.github.io/cuda-python/cuda-bindings/latest/) - driver + runtime bindings
- `numpy` - dtype sizes and array plumbing

## Key APIs

### From `cuda.bindings.runtime`

- `cudaGetDeviceProperties` (for `managedMemory`, `pageableMemoryAccess`, `major`)
- `cudaDeviceGetAttribute` (for compute mode)
- `cudaMallocManaged` / `cudaFree`
- `cudaDeviceSynchronize`
- `cudaMemAttachGlobal`

### From `cuda.bindings.driver`

- `cuLaunchKernel`
- `CU_STREAM_LEGACY`

### Kernel-side

- `atomicAdd_system`, `atomicExch_system`, `atomicMax_system`,
  `atomicMin_system`, `atomicInc_system`, `atomicDec_system`,
  `atomicCAS_system`, `atomicAnd_system`, `atomicOr_system`,
  `atomicXor_system`

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 6.0 or higher and Unified Memory support

### Software

- Linux (system-scope atomics on managed memory are not supported on Windows here)
- CUDA Toolkit 13.0 or newer
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)
- `numpy`

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python systemWideAtomics.py
python systemWideAtomics.py --device=1     # use a specific GPU
```

## Expected Output

```
CAN access pageable memory
systemWideAtomics: all 10 system-scope atomic operations verified
Done
```

(Or `CANNOT access pageable memory` on hardware that falls back to
`cudaMallocManaged`.)

## Files

- `systemWideAtomics.py` - Python implementation using `cuda.bindings`
- `README.md` - This file
- `requirements.txt` - Sample dependencies
- `../../Utilities/cuda_bindings_utils.py` - Shared bindings helpers (imported by this sample)

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [CUDA C++ Programming Guide — Atomic Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions)
- [Scope of Atomic Operations](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#scope-of-atomic-operations)
