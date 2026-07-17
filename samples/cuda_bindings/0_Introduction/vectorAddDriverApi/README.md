# Sample: Vector Add via Driver API (Python)

## Description

The "hello world" of the raw CUDA Driver API in Python. Compute `C = A + B`
element-wise on the GPU using ``cuda.bindings.driver`` end-to-end:

```
cuInit -> cuDeviceGet -> cuCtxCreate
      -> NVRTC compile + cuModuleLoadData + cuModuleGetFunction
      -> cuMemAlloc -> cuMemcpyHtoD -> cuLaunchKernel -> cuMemcpyDtoH -> cuMemFree
      -> cuCtxDestroy
```

This is the low-level equivalent of the higher-level
[`samples/cuda_core/vectorAdd/`](../../../cuda_core/vectorAdd/) sample, which does the same
computation with the ``cuda.core`` API. Both are useful:

- Use ``samples/cuda_core/vectorAdd/`` when you want an idiomatic, Pythonic launch.
- Use this sample to understand what those higher-level abstractions do
  under the hood, or when you need direct control over the driver API
  (custom contexts, explicit stream/module lifetime, etc.).

The sample also queries ``CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING`` and
waives with exit code 2 when the device does not support UVA (a
prerequisite for most driver-API patterns).

## What You'll Learn

- The full raw driver-API launch flow (`cuInit` through `cuCtxDestroy`)
- Explicit CUDA context management with `cuCtxCreate` / `cuCtxDestroy`
- Compiling a kernel string at runtime with NVRTC via `KernelHelper`
- Querying a device attribute (`cuDeviceGetAttribute`)
- Passing kernel arguments to `cuLaunchKernel` as a ``(values, ctypes)`` pair

## Key Libraries

- [`cuda.bindings`](https://nvidia.github.io/cuda-python/cuda-bindings/latest/) - low-level driver / NVRTC bindings
- `numpy` - host-side input/output buffers

## Key APIs

### From `cuda.bindings.driver`

- `cuInit` / `cuDeviceGet` / `cuDeviceGetAttribute`
- `cuCtxCreate` / `cuCtxDestroy`
- `cuMemAlloc` / `cuMemFree` / `cuMemcpyHtoD` / `cuMemcpyDtoH`
- `cuLaunchKernel`
- `CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING`

### From `cuda_bindings_utils` (`samples/cuda_bindings/Utilities/`)

- `KernelHelper(source, dev_id)` - NVRTC compile + load
- `check_cuda_errors(result)` - unwrap `(status, *values)` tuples
- `find_cuda_device_drv()` - pick a device (honors ``--device=<id>``)

## Requirements

### Hardware

- NVIDIA GPU with UVA support (all discrete NVIDIA GPUs since Kepler)

### Software

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
python vectorAddDriverApi.py
python vectorAddDriverApi.py --device=1     # use a specific GPU
```

## Expected Output

```
Result = PASS (max error 0.000e+00 over 50000 elements)
Done
```

## Files

- `vectorAddDriverApi.py` - Python implementation using `cuda.bindings.driver`
- `README.md` - This file
- `requirements.txt` - Sample dependencies
- `../../Utilities/cuda_bindings_utils.py` - Shared bindings helpers (imported by this sample)

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.bindings` driver API](https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/driver.html)
- [`samples/cuda_core/vectorAdd/`](../../../cuda_core/vectorAdd/) - the high-level `cuda.core` equivalent
- [CUDA Driver API Reference](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)
