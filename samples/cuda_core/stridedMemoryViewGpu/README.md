# Sample: StridedMemoryView + Foreign GPU Kernel (Python)

## Description

GPU counterpart to [`../stridedMemoryViewCpu/`](../stridedMemoryViewCpu/).
Same library-style entry point (a Python function decorated with
`@args_viewable_as_strided_memory((0,))`) but this time the wrapped
function dispatches to a **CUDA kernel** compiled at runtime with NVRTC
via `cuda.core.Program`.

The kernel signature

```c
__global__ void inplace_plus_arange_N(int* data, size_t N);
```

adds `i` to `data[i]`. Because `StridedMemoryView` carries a raw device
pointer (`view.ptr`) plus shape/dtype/device-accessibility metadata, the
library can launch the kernel without ever knowing that the caller used
CuPy (or a DLPack capsule, or a raw `Buffer`, or anything else).

The constructor tour lives in
[`../stridedMemoryViewConstructors/`](../stridedMemoryViewConstructors/).

## What You'll Learn

- Building a library entry point with `@args_viewable_as_strided_memory((0,))`
  that dispatches to a CUDA kernel
- Ordering a `StridedMemoryView` against a caller's data stream by passing
  `view = arr.view(work_stream.handle)`
- Reading `view.ptr` inside the library and using it as the kernel's `int*`
  argument via `launch(...)`
- Compiling and launching a kernel via `cuda.core.Program` / NVRTC

## Key Libraries

- [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/) - `Device`, `Program`, `LaunchConfig`, `launch`, `StridedMemoryView`, `args_viewable_as_strided_memory`
- `cupy` - GPU array used as the caller-side input
- `numpy` - scalar kernel arguments

## Key APIs

### From `cuda.core`

- `Program`, `ProgramOptions`, `Program.compile("cubin")` - NVRTC compile pipeline
- `LaunchConfig`, `launch(stream, config, kernel, ...)` - kernel launch
- `Device`, `Device.create_stream()` - device context and stream management

### From `cuda.core.utils`

- `@args_viewable_as_strided_memory((argument_index,))` - decorator materializing arguments as `StridedMemoryView`
- `StridedMemoryView.ptr`, `.shape`, `.dtype`, `.is_device_accessible` - metadata read by the library

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 5.0 or higher

### Software

- CUDA Toolkit 13.0 or newer (matches `cuda-python` 13.x)
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)
- `cuda-core` (>=1.0.0)
- `cupy-cuda13x` (>=14.0.0)
- `numpy` (>=2.3.2)

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python stridedMemoryViewGpu.py
```

## Expected Output

```
before: arr_gpu[:10]=array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)
after:  arr_gpu[:10]=array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10], dtype=int32)
Done
```

## Files

- `stridedMemoryViewGpu.py` - Python implementation
- `README.md` - This file
- `requirements.txt` - Sample dependencies

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.core` StridedMemoryView API](https://nvidia.github.io/cuda-python/cuda-core/latest/api.html#cuda.core.utils.StridedMemoryView)
- [`stridedMemoryViewConstructors/`](../stridedMemoryViewConstructors/) - the four `from_*` constructors
- [`stridedMemoryViewCpu/`](../stridedMemoryViewCpu/) - CPU-side foreign-function consumer
