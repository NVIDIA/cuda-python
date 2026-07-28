# Sample: StridedMemoryView + Foreign CPU Function (Python)

## Description

Model a library that accepts "any array-like object" and dispatches to a
JIT-compiled **CPU** implementation. `cuda.core.utils.StridedMemoryView`
+ the `@args_viewable_as_strided_memory` decorator hide the caller's array
protocol from the library, so the library only needs to know:

- how many elements are in the array (`view.shape`)
- what type they are (`view.dtype`)
- whether they are on the host or the device (`view.is_device_accessible`)
- and a raw pointer to the data (`view.ptr`)

The sample compiles a small C function

```c
void inplace_plus_arange_N(int* data, size_t N);   // data[i] += i
```

via `cffi` at runtime, wraps it in a decorated Python function, and calls
it against a NumPy array. Result verified against
`np.arange(1024, dtype=np.int32)`.

The GPU counterpart lives in
[`../stridedMemoryViewGpu/`](../stridedMemoryViewGpu/); the constructor
tour lives in
[`../stridedMemoryViewConstructors/`](../stridedMemoryViewConstructors/).

## What You'll Learn

- Building a library entry point with `@args_viewable_as_strided_memory((0,))`
- Reading `view.ptr`, `view.shape`, `view.dtype`,
  `view.is_device_accessible` inside the library
- JIT-compiling a small C function with `cffi` and calling it through the
  view's raw pointer

## Key Libraries

- [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/) - `StridedMemoryView`, `args_viewable_as_strided_memory`
- `cffi` - JIT-compile the C function
- `numpy` - test array and reference computation

## Key APIs

### From `cuda.core.utils`

- `StridedMemoryView` - describes the layout of an array-like object
- `@args_viewable_as_strided_memory((argument_index,))` - decorator that
  materializes the annotated arguments as `StridedMemoryView` instances
  inside the wrapped function

### From `cffi`

- `FFI()`, `FFI.cdef()`, `FFI.set_source()`, `FFI.compile(tmpdir=...)` - JIT-compile a shared library
- `FFI.cast("int*", ptr)` - reinterpret an integer pointer as a typed C pointer

## Requirements

### Hardware

- CPU only. No GPU work is performed on the array itself; a valid CUDA
  install is only needed because `cuda.core` is imported.

### Software

- CUDA Toolkit 13.0 or newer (matches `cuda-python` 13.x)
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)
- `cuda-core` (>=1.0.0)
- `numpy` (>=2.3.2)
- `cffi`
- `setuptools` (needed by `cffi` for the JIT build step)
- A working C++ compiler on `PATH`

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python stridedMemoryViewCpu.py
```

## Expected Output

```
before: arr_cpu[:10]=array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)
after:  arr_cpu[:10]=array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)
Done
```

## Files

- `stridedMemoryViewCpu.py` - Python implementation
- `README.md` - This file
- `requirements.txt` - Sample dependencies

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.core` StridedMemoryView API](https://nvidia.github.io/cuda-python/cuda-core/latest/api.html#cuda.core.utils.StridedMemoryView)
- [`stridedMemoryViewConstructors/`](../stridedMemoryViewConstructors/) - the four `from_*` constructors
- [`stridedMemoryViewGpu/`](../stridedMemoryViewGpu/) - GPU counterpart using NVRTC
- [`cffi` documentation](https://cffi.readthedocs.io/)
