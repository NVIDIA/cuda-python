# Sample: Conjugate Gradient with Cooperative Multi-Block Sync (Python)

## Description

Solves ``A x = b`` for a random sparse tridiagonal symmetric-positive-
definite matrix ``A`` (1M x 1M in CSR format) using the Conjugate Gradient
method. **The entire CG iteration runs inside a single kernel** launched
via ``cuLaunchCooperativeKernel``; each iteration uses
``cg::grid_group::sync()`` to move between phases without returning to the
host.

The device-side building blocks are:

- `gpuSpMV` -- sparse matrix-vector multiply (CSR)
- `gpuSaxpy` -- `y = a*x + y`
- `gpuDotProduct` -- warp-shuffle reduce (`cg::reduce`) + `atomicAdd`
  across blocks
- `gpuScaleVectorAndSaxpy`, `gpuCopyVector` -- CG bookkeeping

This is the only end-to-end numerical solver in `/samples/cuda_bindings` that uses
grid-level cooperative synchronization. The simpler
[`samples/cuda_core/reductionMultiBlockCG/`](../../../cuda_core/reductionMultiBlockCG/) uses the
same underlying feature for a plain reduction.

Waives with exit code 2 on Darwin / QNX / armv7l, on devices without
Unified Memory, and on devices without Cooperative Kernel Launch
support.

## What You'll Learn

- Launching a cooperative kernel with `cuLaunchCooperativeKernel`
- Grid-level synchronization inside a kernel via `cg::grid_group::sync()`
- Sizing the grid to saturate the device with cooperating blocks using
  `cuOccupancyMaxActiveBlocksPerMultiprocessor`
- Warp-tile reductions with `cg::reduce` and `cg::tiled_partition<32>`
- Multi-kernel-in-one-launch design for iterative solvers
- Managed (unified) memory shared between host initialization and device
  compute

## Key Libraries

- [`cuda.bindings`](https://nvidia.github.io/cuda-python/cuda-bindings/latest/) - driver + runtime bindings
- `numpy` - dtype sizes

## Key APIs

### From `cuda.bindings.runtime`

- `cudaMallocManaged` / `cudaFree`
- `cudaGetDeviceProperties` (for `managedMemory`, `cooperativeLaunch`,
  `multiProcessorCount`)
- `cudaEventCreate` / `cudaEventRecord` / `cudaEventElapsedTime` / `cudaEventDestroy`
- `cudaDeviceSynchronize`

### From `cuda.bindings.driver`

- `cuLaunchCooperativeKernel`
- `cuOccupancyMaxActiveBlocksPerMultiprocessor`

### Kernel-side

- `cooperative_groups::this_grid()`, `cg::grid_group::sync()`,
  `cg::grid_group::thread_rank()`, `cg::grid_group::size()`
- `cg::tiled_partition<32>(cta)`, `cg::reduce(tile, x, cg::plus<T>())`

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 7.0+ that supports Cooperative Kernel
  Launch and Unified Memory (all discrete Pascal-and-later GPUs)

### Software

- Linux (not supported on Darwin, QNX, or armv7l)
- CUDA Toolkit 13.0 or newer
- Python 3.10 or newer
- `cuda-python` (>=13.4.0)
- `numpy`

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python conjugateGradientMultiBlockCG.py
python conjugateGradientMultiBlockCG.py --device=1
```

## Expected Output

Timings depend on GPU.

```
> GPU device has 128 Multi-Processors, SM 8.9 compute capability
GPU Final, residual = 3.72e-06, kernel execution time = 4.201 ms
Test Summary: Error amount = 0.000012
Done
```

## Files

- `conjugateGradientMultiBlockCG.py` - Python implementation using `cuda.bindings`
- `README.md` - This file
- `requirements.txt` - Sample dependencies
- `../../Utilities/cuda_bindings_utils.py` - Shared bindings helpers (imported by this sample)

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.bindings` driver API](https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/driver.html)
- [`samples/cuda_core/reductionMultiBlockCG/`](../../../cuda_core/reductionMultiBlockCG/) - simpler cooperative-launch demo
- [CUDA C++ Programming Guide — Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)
