# Sample: Kernel Timing with device clock() (Python)

## Description

Time a reduction kernel using the CUDA device-side ``clock()`` intrinsic,
compiled at runtime with NVRTC and launched through the driver API in
``cuda.bindings``.

Each thread block records the SM cycle counter at kernel entry and exit;
the host then reports the average cycles-per-block spent in the reduction.

The sample simultaneously demonstrates **dynamic shared memory**: the
kernel declares ``extern __shared__ float shared[]`` and the host passes
the byte size at launch time via ``cuLaunchKernel``'s ``sharedMemBytes``
argument (the 7th argument), so the actual size of the shared array is
determined at launch time rather than at compile time.

Waives on 32-bit ARM (``armv7l``), where the sample isn't supported.

## What You'll Learn

- Reading the SM cycle counter from a kernel via the CUDA ``clock()`` intrinsic
- Sizing an ``extern __shared__`` array at launch time via ``sharedMemBytes``
- The bare-metal driver-API launch flow: compile with NVRTC, get a
  ``CUfunction``, call ``cuLaunchKernel``
- Copying host / device buffers with ``cuMemAlloc`` / ``cuMemcpyHtoD`` /
  ``cuMemcpyDtoH`` / ``cuMemFree``

## Key Libraries

- [`cuda.bindings`](https://nvidia.github.io/cuda-python/cuda-bindings/latest/) - low-level driver / NVRTC bindings
- `numpy` - host-side input/output buffers

## Key APIs

### From `cuda.bindings.driver`

- `cuMemAlloc` / `cuMemFree` - device memory alloc/free
- `cuMemcpyHtoD` / `cuMemcpyDtoH` - synchronous host/device copies
- `cuLaunchKernel(func, gx, gy, gz, bx, by, bz, sharedMemBytes, stream, args, extra)` - launch a kernel
- `cuCtxSynchronize` - wait for all outstanding device work

### From `cuda_bindings_utils` (`samples/cuda_bindings/Utilities/`)

- `KernelHelper(source, dev_id)` - compile a source string with NVRTC and load the module
- `check_cuda_errors(result)` - unwrap ``(status, *values)`` tuples returned by cuda.bindings
- `find_cuda_device()` - pick a device (honors ``--device=<id>``)

### Kernel-side

- ``clock_t clock()`` - device-side SM cycle counter
- ``extern __shared__ float shared[]`` - dynamically-sized shared memory

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 5.0 or higher
- Not supported on ARMv7

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
python clockNvrtc.py
python clockNvrtc.py --device=1     # use a specific GPU
```

## Expected Output

Cycle counts are hardware-dependent.

```
Average clocks/block = 8734.0
Done
```

## Files

- `clockNvrtc.py` - Python implementation using `cuda.bindings`
- `README.md` - This file
- `requirements.txt` - Sample dependencies
- `../../Utilities/cuda_bindings_utils.py` - Shared bindings helpers (imported by this sample)

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.bindings` driver API](https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/driver.html)
- [CUDA C++ Programming Guide — Time Function](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#time-function)
