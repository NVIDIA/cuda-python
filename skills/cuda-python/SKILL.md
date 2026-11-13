---
name: cuda-python
description: >-
  Use for any task or question involving CUDA Python Core libraries: cuda-core, cuda-bindings
  and cuda-pathfinder.
  Use when asked to write, port, review, debug Python code that compiles and launches
  its own CUDA kernels (Program, launch, LaunchConfig, Device, Stream, Buffer, Linker, CUDA graphs).
  When porting PyCUDA or old CUDA bindings APIs to cuda.core and cuda.bindings.
  When interacting with passing CuPy, PyTorch or Numba arrays and streams to a custom kernel.
  When locating CUDA headers or libraries from Python, and for install questions about
  cuda-python, cuda-core or conda.
  Use before hand-rolling cuda.bindings driver / nvrtc / nvml calls for something cuda.core provides.
  Not for plain CuPy, Numba or PyTorch array code, or for cuda.compute (CCCL).
license: Apache-2.0
metadata:
  author: NVIDIA
  tags: [cuda, cuda-python, cuda-core, cuda-bindings, cuda-pathfinder]
---

# CUDA Python: cuda.core, cuda.bindings, cuda.pathfinder

`cuda-python` is a metapackage that covers:
- `cuda.core`: the Pythonic API
- `cuda.bindings`: 1:1 wrappers of the CUDA C APIs
- `cuda.pathfinder`: locates CUDA libraries and headers

## Install

- pip: driver only, no CUDA Toolkit: `pip install "cuda-core[cu13]"` (or `[cu12]`).
  The extra pulls `cuda-bindings[all]`, which brings the NVRTC, nvJitLink and NVVM wheels.
- conda: `conda install -c conda-forge cuda-python cuda-version=13` (or `cuda-core`, or `cuda-version=12`).
  One `cuda-core` release supports both CUDA majors; there are no `-cu12` / `-cu13` package names.


## Use cuda.core first

`cuda.core` is the Pythonic API to all CUDA APIs and should be used for
all new Python code as far as the API is available, only fall back to
`cuda-bindings` when needed.
`cuda.core` raises exceptions and manages handles for you, while `cuda.bindings` `driver`, `runtime` and `nvrtc` calls return `(err, ...)` tuples, so an unchecked failure is silent.

| Task | cuda.core API |
|---|---|
| Compile CUDA C++ at runtime | `Program(src, code_type="c++", options=ProgramOptions(...)).compile("cubin")`, `.get_kernel(name)` |
| Device and pinned memory | `dev.allocate(nbytes, stream=s)`, `PinnedMemoryResource()`, `Buffer.copy_to` / `copy_from` / `fill` |
| CUDA graphs | `stream.create_graph_builder()` (types in `cuda.core.graph`) |
| Link with LTO | `ProgramOptions(link_time_optimization=True)`, `.compile("ltoir")`, `Linker(a, b, options=LinkerOptions(link_time_optimization=True)).link("cubin")` |
| GPU monitoring (NVML) | `cuda.core.system`: `system.Device.get_all_devices()`, `.name`, `.uuid`, `.memory_info`, `.temperature.get_sensor()`, `.utilization.gpu`, `system.get_kernel_mode_driver_version()` |
| OpenGL interop | `GraphicsResource.from_gl_buffer` / `from_gl_image`, `.map(stream=s)` |

## Compile, copy, launch

A `cuda.core` program usually follows the same four steps: make a device current, compile the kernel,
move data into device memory, and launch on a stream. Everything submitted to a stream is
asynchronous, so results are only valid after `stream.sync()`.

```python
import numpy as np

from cuda.core import Device, LaunchConfig, PinnedMemoryResource, Program, launch

# extern "C" keeps the kernel name unmangled so get_kernel("scale") can find it.
code = r"""
extern "C" __global__ void scale(float* x, float a, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) { x[i] *= a; }
}
"""

# 1. Device and stream. set_current() comes first, and in every thread that uses the GPU.
dev = Device()
dev.set_current()
stream = dev.create_stream()

# 2. Compile with NVRTC. "cubin" is machine code for this GPU and always loads.
kernel = Program(code, code_type="c++").compile("cubin").get_kernel("scale")

# 3. Allocate pinned host memory and device memory.
n = 1 << 20
h_x = PinnedMemoryResource().allocate(n * 4, stream=stream)
d_x = dev.allocate(n * 4, stream=stream)

# Host data goes through a pinned `Buffer` that can be viewed from Numpy using `np.from_dlpack`
x = np.from_dlpack(h_x).view(np.float32)  # zero-copy NumPy view of the pinned buffer
x[:] = np.arange(n)

# 4. Copy in, launch, copy out. Scalars are sized NumPy types that match the kernel signature.
h_x.copy_to(d_x, stream=stream)
launch(stream, LaunchConfig(grid=(n + 255) // 256, block=256), kernel, d_x, np.float32(2.5), np.uint64(n))
d_x.copy_to(h_x, stream=stream)
stream.sync()  # x now holds the result
```

## Kernel arguments

`launch()` doesn't check the kernel signature, a missmatch will generate zeros or garbage output.

- Pass sized scalars: `np.float32` for `float`, `np.float64` for `double`, `np.int32` for `int`, `np.uint64` for `size_t`.
- Pass pointers as a `Buffer` or an `int` address (`arr.data.ptr`, `tensor.data_ptr()`, `view.ptr`).
- Arrays, tensors and structs (`ctypes.Structure`, NumPy structured scalars) raise `TypeError`.

## 3rd party libraries: CuPy, PyTorch and Numba

A stream from `dev.create_stream()` is not ordered with the other library's stream, so a kernel can
run before a preceding call from another library has finished. Share one stream instead of synchronizing:
`s = dev.create_stream(cp.cuda.get_current_stream())`. Their stream objects work directly.

`StridedMemoryView` reads arrays from CuPy, PyTorch and Numba. Validate before passing `view.ptr` to a kernel:

```python
view = StridedMemoryView.from_any_interface(arr, stream_ptr=stream.handle)
c_strides = tuple(int(np.prod(view.shape[i + 1 :])) for i in range(len(view.shape)))
ok = view.is_device_accessible and view.dtype == np.float32 and view.strides in (None, c_strides)
```

## CUDA graphs

Capture a batch of launches once and replay it with a single driver call. Use it for loops of many
small kernels, where the host-side launch overhead dominates.

```python
gb = stream.create_graph_builder().begin_building()
for _ in range(100):
    launch(gb, config, kernel, buf, n)  # recorded, not executed; arguments are fixed at capture
graph = gb.end_building().complete()
graph.launch(stream)  # replays all 100 launches
stream.sync()
```

## Headers and libraries: cuda.pathfinder

Never hard-code a cuda location such as: `/usr/local/cuda`.
`cuda.pathfinder` searches pip wheels, conda and `CUDA_PATH` for the right location.

```python
from cuda.pathfinder import find_nvidia_header_directory

include = [find_nvidia_header_directory("cudart"), find_nvidia_header_directory("cccl")]  # None if missing
options = ProgramOptions(std="c++17", include_path=include)
```

## Migration

| Old | Now |
|---|---|
| `from cuda import cuda, nvrtc`, `cuda.cudart` (removed in 13.0) | `cuda.bindings.driver`, `.nvrtc`, `.runtime`; port new code to `cuda.core` |
| `cuda.core.experimental` (removed in 1.0), `from cuda.core import GraphBuilder` | `cuda.core`, `cuda.core.utils`, `cuda.core.graph` |
| PyCUDA `autoinit`, `SourceModule`, `mem_alloc`, `pagelocked_empty`, `memcpy_htod_async`, `Event.time_since` | `Device().set_current()`, `Program` (add `extern "C"`), `dev.allocate`, pinned `Buffer`, `Buffer.copy_to`, `end - start` |

## References

- Docs: https://nvidia.github.io/cuda-python/cuda-core/latest/ (`api.html`, `release.html`)
- Source: if you can clone repositories, clone https://github.com/NVIDIA/cuda-python in a temp location to validate APIs and produce more accurate code.
