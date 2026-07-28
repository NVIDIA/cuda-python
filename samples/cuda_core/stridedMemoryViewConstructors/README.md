# Sample: StridedMemoryView Constructors (Python)

## Description

`cuda.core.utils.StridedMemoryView` is the type a library uses when it
wants to accept "any array-like object" and describe its memory layout
without dictating whether the caller uses NumPy, CuPy, PyTorch, DLPack, or a
raw `cuda.core.Buffer`.

This sample walks through the four explicit `from_*` constructors and
verifies each round-trips its input via DLPack:

1. `from_array_interface(host_numpy_array)` — NumPy's `__array_interface__`
2. `from_dlpack(any_array, stream_ptr=...)` — the DLPack protocol (host or device)
3. `from_cuda_array_interface(gpu_array, stream_ptr=...)` — `__cuda_array_interface__`
4. `from_buffer(buf, shape=..., strides=..., dtype=...)` — a raw
   `cuda.core.Buffer` plus explicit layout metadata

Companion samples show how a library actually *consumes* these views:
[`stridedMemoryViewCpu/`](../stridedMemoryViewCpu/) dispatches to a
JIT-compiled CPU function via `cffi`, and
[`stridedMemoryViewGpu/`](../stridedMemoryViewGpu/) dispatches to an NVRTC
kernel.

## What You'll Learn

- Constructing a `StridedMemoryView` from each of the four supported inputs
- Inspecting a view's `shape`, `dtype`, `size`, and `is_device_accessible`
- Building your own explicit layout (`shape`/`strides`/`dtype`) for a raw
  `Buffer`
- Round-tripping a view via DLPack back to NumPy or CuPy

## Key Libraries

- [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/) - `Device`, `Buffer`, `StridedMemoryView`
- `numpy` (>=2.1) - host arrays and DLPack round-trip
- `cupy` - GPU arrays and DLPack / CAI round-trip

## Key APIs

### From `cuda.core.utils`

- `StridedMemoryView.from_array_interface(obj)`
- `StridedMemoryView.from_dlpack(obj, stream_ptr=-1)` (host) or `stream_ptr=stream.handle` (device)
- `StridedMemoryView.from_cuda_array_interface(obj, stream_ptr=stream.handle)`
- `StridedMemoryView.from_buffer(buf, shape=..., strides=..., dtype=...)`

### From `cuda.core`

- `Device.memory_resource.allocate(nbytes, stream=...)` - obtain a `Buffer` for the `from_buffer` demo

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 5.0 or higher

### Software

- CUDA Toolkit 13.0 or newer (matches `cuda-python` 13.x)
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)
- `cuda-core` (>=1.0.0)
- `cupy-cuda13x` (>=14.0.0)
- `numpy` (>=2.1) — the DLPack round-trip requires NumPy 2.1+

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python stridedMemoryViewConstructors.py
```

## Expected Output

```
[1] from_array_interface(host_numpy_array)
  host_view: shape=(3, 4), dtype=int16, size=12 (host-only)

[2] from_dlpack(host_array, stream_ptr=-1)
  host_dlpack_view: shape=(3, 4), dtype=int16, size=12 (host-only)

[3] from_dlpack(gpu_array) and from_cuda_array_interface(gpu_array)
  dlpack_view (gpu): shape=(3, 4), dtype=float32, size=12 (device-accessible)
  cai_view (gpu): shape=(3, 4), dtype=float32, size=12 (device-accessible)

[4] from_buffer(buf, shape=..., strides=..., dtype=...)
  buffer_view: shape=(3, 4), dtype=float32, size=12 (device-accessible)

Constructed StridedMemoryView from array_interface, DLPack, CAI, and Buffer inputs.
Done
```

## Files

- `stridedMemoryViewConstructors.py` - Python implementation
- `README.md` - This file
- `requirements.txt` - Sample dependencies

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.core` StridedMemoryView API](https://nvidia.github.io/cuda-python/cuda-core/latest/api.html#cuda.core.utils.StridedMemoryView)
- [DLPack Protocol](https://dmlc.github.io/dlpack/latest/)
- [`__cuda_array_interface__`](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html)
- [`stridedMemoryViewCpu/`](../stridedMemoryViewCpu/) - CPU-side foreign-function consumer
- [`stridedMemoryViewGpu/`](../stridedMemoryViewGpu/) - GPU-side foreign-function consumer
