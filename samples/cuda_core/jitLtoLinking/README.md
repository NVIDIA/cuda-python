# JIT Compilation and Link-Time Optimization (Python)

## Description

This sample demonstrates how to build a kernel out of two independently
compiled translation units and link them at runtime with
`cuda.core.Linker`. This is the pattern a library would use to accept
user-supplied device code as a plug-in without recompiling its own
kernels from scratch.

The sample runs the same program in three phases:

1. **PTX linking** - each module is compiled with
   `ProgramOptions(relocatable_device_code=True)` down to PTX, and the
   `Linker` emits a final cubin. The two modules stay independently
   compiled (no cross-module inlining).
2. **Link-Time Optimization (LTO)** - each module is compiled with
   `ProgramOptions(link_time_optimization=True)` down to LTO IR, and the
   `Linker` is configured with `LinkerOptions(link_time_optimization=True)`
   so the optimizer runs again across both modules, typically matching
   the code generation of a single-source build.
3. **Plug-in swap** - the *same* library main kernel is re-linked
   against a *different* `user_transform` implementation, producing a
   new cubin without re-compiling the library main kernel. This
   demonstrates the runtime-plug-in pattern that motivates the whole
   library / user split.

The "main" kernel `apply_transform` calls a `user_transform` device
function that lives in a separate source string, and the results of every
mode are verified against a NumPy reference.

## What You'll Learn

- Compiling multiple `Program` objects into PTX or LTO IR
- Linking independent object codes into a single cubin with `Linker`
- Choosing between `relocatable_device_code` and `link_time_optimization`
- How a library's main kernel can call into user-supplied device code
- When to prefer LTO over plain PTX linking
- Swapping the plug-in module without rebuilding the library main kernel's
  intermediate representation

## Key Libraries

- [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/) - Pythonic access to CUDA runtime, programs, and the JIT linker
- `cupy` - input and output buffers on the GPU
- `numpy` - reference computation on the host

## Key APIs

### From `cuda.core`

- `ProgramOptions(relocatable_device_code=True)` + `Program.compile("ptx")` - produce relocatable PTX
- `ProgramOptions(link_time_optimization=True)` + `Program.compile("ltoir")` - produce LTO IR
- `Linker(*object_codes, options=LinkerOptions(...))` - create a JIT linker over multiple object codes
- `LinkerOptions(link_time_optimization=True)` - opt into LTO during linking
- `Linker.link("cubin")` - produce a loadable module
- `ObjectCode.get_kernel(name)` - fetch a kernel from the linked module

### From `cuda_samples_utils`

- `print_gpu_info()` - print device name and compute capability

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 7.0 or higher

### Software

- CUDA Toolkit 13.0 or newer (matches `cuda-python` 13.x)
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)
- `cuda-core` (>=1.0.0)
- `cupy-cuda13x` (>=14.0.0)

## Installation

Install the required packages from `requirements.txt`:

```bash
cd /path/to/cuda-python/samples/cuda_core/jitLtoLinking
pip install -r requirements.txt
```

The `requirements.txt` installs:

- `cuda-python` (>=13.0.0)
- `cuda-core` (>=1.0.0)
- `cupy-cuda13x` (>=14.0.0)

## How to Run

### Basic usage

```bash
cd samples/cuda_core/jitLtoLinking
python jitLtoLinking.py
```

### With custom parameters

```bash
# Larger element count
python jitLtoLinking.py --elements 1048576

# Use a specific GPU
python jitLtoLinking.py --device 1
```

## Expected Output

```
Device: <Your GPU Name>
Compute Capability: <X.Y>

[1] PTX linking (no LTO)
  [ptx] result verified against NumPy reference

[2] LTO linking (link-time optimization)
  [lto] result verified against NumPy reference

[3] Plug-in swap: re-link the same main kernel with a different user_transform
  [swap] result verified against NumPy reference

PTX, LTO, and plug-in-swap linked kernels all produced matching results. Done
```

**Note:** Device name and compute capability will vary based on your GPU.

## Files

- `jitLtoLinking.py` - Python implementation using `cuda.core.Linker`
- `README.md` - This file
- `requirements.txt` - Sample dependencies
- `../Utilities/cuda_samples_utils.py` - Common utilities (imported by this sample)

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.core` compilation API](https://nvidia.github.io/cuda-python/cuda-core/latest/api.html#cuda-compilation-toolchain)
- [NVIDIA nvJitLink documentation](https://docs.nvidia.com/cuda/nvjitlink/index.html)
