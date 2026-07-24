# Sample: Global → Shared Async Copy (memcpy_async) (Python)

## Description

Eight matrix-multiply kernels sharing the same launch harness, benchmarked
against a naive tiled baseline. The variants explore progressively more
sophisticated ways to hide the global → shared load latency using
``cuda::memcpy_async`` plus the ``cuda::pipeline`` / ``cuda::barrier``
primitives:

- `AsyncCopyMultiStageLargeChunk`
- `AsyncCopyLargeChunk`
- `AsyncCopyLargeChunkAWBarrier` — arrive/wait barrier variant
- `AsyncCopyMultiStageSharedState`
- `AsyncCopyMultiStage`
- `AsyncCopySingleStage`
- `Naive` — baseline
- `NaiveLargeChunk` — baseline

The sample selects a variant via ``--kernel=N`` (default 0), computes the
reference ``C = A * B`` on the host, and reports GFLOPS. This is the only
sample in ``/samples/cuda_bindings`` that teaches ``cuda::memcpy_async``,
``cuda::pipeline``, or the arrive/wait barrier patterns — the low-level
async-copy machinery that ``cuda.core``'s newer APIs sit on top of.

## What You'll Learn

- Loading global memory into shared memory asynchronously with
  ``cuda::memcpy_async``
- Multi-stage pipelining with ``cuda::pipeline`` and
  ``cuda::pipeline_shared_state`` so different pipeline stages can compute
  and load in parallel
- Arrive/wait barriers with ``cuda::barrier`` for producer/consumer
  synchronization inside a block
- Large-chunk async copies with vectorized load sizes
- Direct comparison against the classic hand-rolled tiled matmul

## Key Libraries

- [`cuda.bindings`](https://nvidia.github.io/cuda-python/cuda-bindings/latest/) - driver + runtime bindings
- `numpy` - host reference computation and matrix plumbing

## Key APIs

### From `cuda.bindings.driver`

- `cuLaunchKernel`

### From `cuda.bindings.runtime`

- `cudaMalloc` / `cudaFree` / `cudaMemcpy` / `cudaMemset`
- `cudaEventCreate` / `cudaEventRecord` / `cudaEventElapsedTime`

### Kernel-side (headers pulled in by NVRTC via `KernelHelper`)

- `cuda/barrier`, `cuda/pipeline`, `cooperative_groups`,
  `cooperative_groups/reduce`
- `cuda::memcpy_async(barrier|pipeline|group, dst, src, size)`
- `cuda::pipeline_shared_state`, `cuda::pipeline`, `producer_acquire`,
  `producer_commit`, `consumer_wait`, `consumer_release`
- `cuda::barrier::arrive` / `arrive_and_wait`

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 7.0 or higher (Volta+). Some kernels
  require SM 7.0 explicitly for `cuda::memcpy_async` support.

### Software

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
python globalToShmemAsyncCopy.py                     # default variant
python globalToShmemAsyncCopy.py --kernel=7          # NaiveLargeChunk
python globalToShmemAsyncCopy.py --wA=1024 --wB=1024 # bigger matrices
```

## Expected Output

Throughput depends on GPU.

```
[globalToShmemAsyncCopy]
GPU Device 0: "NVIDIA GeForce RTX 4090" with compute capability 8.9
MatrixA(1024,1024), MatrixB(1024,1024)
Running kernel = 0 - AsyncCopyMultiStageLargeChunk
Performance= 4123.45 GFlop/s, Time= 0.520 msec, Size= ... Ops, WorkgroupSize= 256 threads/block
Result = PASS
```

## Files

- `globalToShmemAsyncCopy.py` - Python implementation using `cuda.bindings`
- `README.md` - This file
- `requirements.txt` - Sample dependencies
- `../../Utilities/cuda_bindings_utils.py` - Shared bindings helpers (imported by this sample)

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`samples/cuda_core/matrixMulSharedMem/`](../../../cuda_core/matrixMulSharedMem/) - basic tiled GEMM (no async copy)
- [CUDA C++ Programming Guide — Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies)
- [`libcudacxx` `cuda::memcpy_async`](https://nvidia.github.io/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html)
- [`libcudacxx` `cuda::pipeline`](https://nvidia.github.io/libcudacxx/extended_api/synchronization_primitives/pipeline.html)
