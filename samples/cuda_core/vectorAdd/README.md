# Sample: Vector Addition (Python)

## Description

Run your first GPU kernel: add two vectors element-wise on the GPU using the [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/) API with runtime compilation.

The sample compiles a *templated* kernel and instantiates it for two
element types in a single `Program.compile()` call (via
`name_expressions`), then runs two phases:

1. `vectorAdd<float>` against CuPy-allocated input and output buffers.
2. `vectorAdd<double>` where the output buffer is allocated through
   `Device.allocate()` and passed straight to the kernel. That raw
   `Buffer` is then wrapped as a zero-copy CuPy view for verification.

Phase 2 is the pattern to reach for when you want the kernel to write into
memory you own directly, without depending on CuPy's allocator for the
output.

## What You'll Learn

- Writing CUDA kernels in C++ with template support
- Runtime compilation of CUDA kernels from Python
- Requesting multiple template instantiations via `name_expressions`
- Using `cuda.core` for device management, programs, and launches
- Configuring and launching kernels with grid and block dimensions
- Using CuPy for GPU memory management
- Allocating your own output buffer with `Device.allocate()` and wrapping
  it as a CuPy view
- Verifying GPU results against CPU computation

## Key Libraries

- [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/) — Pythonic access to CUDA runtime and compilation
- `cupy` — GPU array library for Python

## Key APIs

### From `cuda.core`

- `Device` — Initialize and manage CUDA device
- `Device.allocate(nbytes, stream=...)` — Allocate a raw `Buffer` from the device pool
- `Program` — Create program from kernel source code
- `Program.compile("cubin", name_expressions=(...))` — Emit specific template instantiations
- `ProgramOptions` — Set compilation options (C++ standard, architecture)
- `LaunchConfig` — Configure kernel launch parameters
- `launch` — Execute kernel on specified stream

Import stable symbols from the top-level package (not `cuda.core.experimental`). See the [cuda.core documentation](https://nvidia.github.io/cuda-python/cuda-core/latest/).

### From CuPy

- `cp.random.rand()` — Generate random arrays on GPU
- `cp.empty()` — Allocate uninitialized GPU arrays
- `cp.allclose()` — Verify results with tolerance

### From `cuda_samples_utils`

- `verify_array_result()` — Verify computation results

## Kernel Techniques

- **1D Grid-Stride Loop** — Handle arbitrary array sizes with fixed grid
- **Template Programming** — Generic kernel for different data types
- **Bounds Checking** — Prevent out-of-bounds memory access

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 7.0 or higher
- Minimum GPU memory: 512 MB

### Software

- CUDA Toolkit 13.0 or newer (matches `cuda-python` 13.x)
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)
- `cuda-core` (>=1.0.0)
- `cupy-cuda13x` (>=14.0.0)

## Installation

Install the required packages from requirements.txt:

```bash
cd /path/to/cuda-python/samples/cuda_core/vectorAdd
pip install -r requirements.txt
```

The requirements.txt installs:

- `cuda-python` (>=13.0.0)
- `cuda-core` (>=1.0.0)
- `cupy-cuda13x` (>=14.0.0)

## How to Run

### Basic usage

```bash
cd samples/cuda_core/vectorAdd
python vectorAdd.py
```

### With custom parameters

```bash
# Custom vector size
python vectorAdd.py --elements 1000000

# Use specific GPU
python vectorAdd.py --device 1

# Skip verification for benchmarking
python vectorAdd.py --no-verify
```

## Expected Output

```
[Vector addition using CUDA Core API]
Device: <Your GPU Name>
Compute Capability: sm_<XX>
Compiling kernel 'vectorAdd<float>' and 'vectorAdd<double>'...
Kernels compiled successfully

[1] vectorAdd<float> on 50000 CuPy-allocated elements
  CUDA kernel launch with 196 blocks of 256 threads
  Verifying result...

[2] vectorAdd<double> on 25000 elements with device.allocate() output
  CUDA kernel launch with 98 blocks of 256 threads
  Verifying result...

Test PASSED

Done
```

**Note:** Device name and compute capability will vary based on your GPU.

## Files

- `vectorAdd.py` — Python implementation using cuda.core API
- `README.md` — This file
- `requirements.txt` — Sample dependencies
- `../Utilities/cuda_samples_utils.py` — Common utilities (imported by this sample)

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [cuda.core API](https://nvidia.github.io/cuda-python/cuda-core/latest/)
- [CuPy Documentation](https://docs.cupy.dev/)
