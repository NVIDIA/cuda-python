# Sample: Raw NVRTC + Driver API SAXPY (Python)

## Description

Walks through the raw NVRTC + driver-API pipeline for compiling and
launching a CUDA kernel. This is the "under the hood" companion to the
higher-level [`samples/cuda_core/jitLtoLinking/`](../../../cuda_core/jitLtoLinking/) sample:
``jitLtoLinking`` uses ``cuda.core.Program`` and ``cuda.core.Linker``;
this sample makes the individual NVRTC + ``cuModule*`` calls those
higher-level abstractions wrap.

The full flow is:

```
nvrtc.create_program  ->  nvrtc.compile_program  ->  nvrtc.get_program_log
                      ->  nvrtc.get_cubin / nvrtc.get_ptx
cuModuleLoadData    ->  cuModuleGetFunction("saxpy")
cuLaunchKernel      ->  cuStreamSynchronize
cuModuleUnload      ->  cuCtxDestroy
```

The kernel is the standard single-precision AXPY: ``out = a * x + y``.

## What You'll Learn

- The complete NVRTC compile pipeline: create program, compile with
  options, retrieve the log, retrieve CUBIN (or PTX on older NVRTC)
- Loading a compiled module with `cuModuleLoadData` and resolving a kernel
  entry point with `cuModuleGetFunction`
- Passing typed kernel arguments through `cuLaunchKernel` using a
  `(values, ctypes)` pair
- Async copies + stream synchronization for host / device transfers
- Correctly tearing down all driver-owned resources

## Key Libraries

- [`cuda.bindings`](https://nvidia.github.io/cuda-python/cuda-bindings/latest/) - low-level driver + NVRTC bindings
- `numpy` - host-side input/output buffers

## Key APIs

### From `cuda.bindings._v2.nvrtc`

- `create_program` / `compile_program`
- `get_program_log`
- `get_cubin`
- `get_ptx`
- `version`

### From `cuda.bindings.driver`

- `cuInit`, `cuDeviceGet`, `cuDeviceGetAttribute`
- `cuCtxCreate` / `cuCtxDestroy`
- `cuModuleLoadData` / `cuModuleGetFunction` / `cuModuleUnload`
- `cuMemAlloc` / `cuMemFree`
- `cuStreamCreate` / `cuStreamSynchronize` / `cuStreamDestroy`
- `cuMemcpyHtoDAsync` / `cuMemcpyDtoHAsync`
- `cuLaunchKernel`

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 5.0 or higher

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
python jitProgram.py
```

## Expected Output

```
SAXPY through raw NVRTC + driver API verified.
Done
```

(An empty compile log line may print before the verification message.)

## Files

- `jitProgram.py` - Python implementation using `cuda.bindings._v2.nvrtc` + `cuda.bindings.driver`
- `README.md` - This file
- `requirements.txt` - Sample dependencies

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.bindings` NVRTC API](https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/nvrtc.html)
- [`cuda.bindings` driver API](https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/driver.html)
- [`samples/cuda_core/jitLtoLinking/`](../../../cuda_core/jitLtoLinking/) - the high-level `cuda.core` equivalent
