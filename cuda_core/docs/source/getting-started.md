# Overview

## What is `cuda core`?

`cuda.core` provides a Pythonic interface to the CUDA runtime and other functionality,
including:

- Compiling and launching CUDA kernels
- Asynchronous concurrent execution with CUDA graphs, streams and events
- Coordinating work across multiple CUDA devices
- Allocating, transferring, and managing device memory
- Runtime linking of device code with Link-Time Optimization (LTO)
- and much more!

Rather than providing 1:1 equivalents of the CUDA driver and runtime APIs
(for that, see [`cuda.bindings`][bindings]), `cuda.core` provides high-level constructs such as:

- {class}`Device <cuda.core.experimental.Device>` class for GPU device operations and context management.
- {class}`Buffer <cuda.core.experimental.Buffer>` and {class}`MemoryResource <cuda.core.experimental.MemoryResource>` classes for memory allocation and management.
- {class}`Program <cuda.core.experimental.Program>` for JIT compilation of CUDA kernels.
- {class}`GraphBuilder <cuda.core.experimental.GraphBuilder>` for building and executing CUDA graphs.
- {class}`Stream <cuda.core.experimental.Stream>` and {class}`Event <cuda.core.experimental.Event>` for asynchronous execution and timing.

## Example: Compiling and Launching a CUDA kernel

To get a taste for `cuda.core`, let's walk through a simple example that compiles and launches a vector addition kernel.
You can find the complete example in [`vector_add.py`][vector_add_example].

First, we define a string containing the CUDA C++ kernel. Note that this is a templated kernel:

```python
# compute c = a + b
code = """
template<typename T>
__global__ void vector_add(const T* A,
                           const T* B,
                           T* C,
                           size_t N) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i=tid; i<N; i+=gridDim.x*blockDim.x) {
        C[tid] = A[tid] + B[tid];
    }
}
"""
```

Next, we create a {class}`Device <cuda.core.experimental.Device>` object
and a corresponding {class}`Stream <cuda.core.experimental.Stream>`.
Don't forget to use {meth}`Device.set_current() <cuda.core.experimental.Device.set_current>`!

```python
from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch

dev = Device()
dev.set_current()
s = dev.create_stream()
```

Next, we compile the CUDA C++ kernel from earlier using the {class}`Program <cuda.core.experimental.Program>` class.
The result of the compilation  is saved as a CUBIN.
Note the use of the `name_expressions` parameter to the {meth}`Program.compile() <cuda.core.experimental.Program.compile>` method to specify which kernel template instantiations to compile:

```python
arch = "".join(f"{i}" for i in dev.compute_capability)
program_options = ProgramOptions(std="c++17", arch=f"sm_{arch}")
prog = Program(code, code_type="c++", options=program_options)
mod = prog.compile("cubin", name_expressions=("vector_add<float>",))
```

Next, we retrieve the compiled kernel from the CUBIN and prepare the arguments and kernel configuration.
We're using [CuPy][cupy] arrays as inputs for this example, but you can use PyTorch tensors too
(we show how to do this in one of our [examples][examples]).

```python
ker = mod.get_kernel("vector_add<float>")

# Prepare input/output arrays (using CuPy)
size = 50000
a = rng.random(size, dtype=cp.float32)
b = rng.random(size, dtype=cp.float32)
c = cp.empty_like(a)

# Configure launch parameters
block = 256
grid = (size + block - 1) // block
config = LaunchConfig(grid=grid, block=block)
```

Finally, we use the {func}`launch <cuda.core.experimental.launch>` function to execute our kernel on the specified stream with the given configuration and arguments. Note the use of `.data.ptr` to get the pointer to the array data.

```python
launch(s, config, ker, a.data.ptr, b.data.ptr, c.data.ptr, cp.uint64(size))
s.sync()
```

This example demonstrates one of the core workflows enabled by `cuda.core`: compiling and launching CUDA code.
Note the clean, Pythonic interface, and absence of any direct calls to the CUDA runtime/driver APIs.

## Examples and Recipes

As we mentioned before, `cuda.core` can do much more than just compile and launch kernels.

The best way to explore and learn the different features `cuda.core` is through
our [`examples`][examples]. Find one that matches your use-case, and modify it to fit your needs!


[bindings]: https://nvidia.github.io/cuda-python/cuda-bindings/latest/
[cai]: https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
[cupy]: https://cupy.dev/
[dlpack]: https://dmlc.github.io/dlpack/latest/
[examples]: https://github.com/NVIDIA/cuda-python/tree/main/cuda_core/examples
[vector_add_example]: https://github.com/NVIDIA/cuda-python/tree/main/cuda_core/examples/vector_add.py
