# Sample: Simple Multi-GPU (Python)

## Description

Compile and launch two different kernels on two GPUs concurrently using
`cuda.core`. Each GPU has its own `Program`, `Stream`, and CuPy-allocated
buffers. There is no peer-to-peer access, no MPI, and no host-side
communication between the two GPUs beyond starting and waiting for them.

- GPU 0 computes `c = a + b` (float32).
- GPU 1 computes `z = x - y` (float32).

The sample also demonstrates the **`StreamAdaptor`** idiom for bridging a
foreign stream (CuPy's current stream) into `cuda.core`. That lets us make
our `cuda.core` stream wait on the stream that CuPy used to initialize the
input buffers, so the kernels see fully-materialized data.

## What You'll Learn

- Managing multiple `Device` contexts in the same Python process
- Compiling one `Program` per device
- Building an ad-hoc `StreamAdaptor` that implements `__cuda_stream__` to
  bridge CuPy's current stream into `cuda.core`
- Ordering `cuda.core` work against a foreign stream via `Stream.wait()`
- Running independent kernel launches concurrently on two GPUs

## Key Libraries

- [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/) - Pythonic access to CUDA runtime, programs, and streams
- `cupy` - GPU array library used for buffer allocation and result verification

## Key APIs

### From `cuda.core`

- `system.get_num_devices()` - count of visible CUDA devices
- `Device(device_id)` / `Device.set_current()` - select and activate a device
- `Device.create_stream()` - create a `cuda.core` stream on the current device
- `Device.create_stream(adaptor)` - wrap a foreign stream via the `__cuda_stream__` protocol
- `Stream.wait(other_stream)` - order this stream after another stream
- `Program`, `ProgramOptions`, `LaunchConfig`, `launch` - standard compile / launch flow

### From CuPy

- `cp.cuda.get_current_stream()` - the CuPy stream that owns recent allocations
- `cp.random.default_rng().random(...)` - GPU random buffers on the current device

## Requirements

### Hardware

- **At least 2 CUDA-capable devices** in the system. The sample waives
  itself when only one GPU is visible.
- Compute Capability 7.0 or higher on both devices.

### Software

- CUDA Toolkit 13.0 or newer (matches `cuda-python` 13.x)
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)
- `cuda-core` (>=1.0.0)
- `cupy-cuda13x` (>=14.0.0)

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python simpleMultiGpu.py
```

The sample has no CLI arguments; it always uses device 0 and device 1.

## Expected Output

```
GPU 0: vector_add on 50000 elements verified
GPU 1: vector_sub on 50000 elements verified
Done
```

On single-GPU systems the sample waives:

```
This sample requires at least 2 CUDA-capable devices (found 1). Waiving.
```

## Files

- `simpleMultiGpu.py` - Python implementation using `cuda.core`
- `README.md` - This file
- `requirements.txt` - Sample dependencies

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.core` API](https://nvidia.github.io/cuda-python/cuda-core/latest/)
- [CuPy Documentation](https://docs.cupy.dev/)
