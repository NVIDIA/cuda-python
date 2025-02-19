# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# ################################################################################
#
# This demo aims to illustrate two takeaways:
#
#   1. The similarity between CPU and GPU JIT-compilation with C++ sources
#   2. How to use StridedMemoryView to interface with foreign C/C++ functions
#
# To facilitate this demo, we use cffi (https://cffi.readthedocs.io/) for the CPU
# path, which can be easily installed from pip or conda following their instructions.
# We also use NumPy/CuPy as the CPU/GPU array container.
#
# ################################################################################

import importlib
import string
import sys

try:
    from cffi import FFI
except ImportError:
    print("cffi is not installed, the CPU example will be skipped", file=sys.stderr)
    FFI = None
try:
    import cupy as cp
except ImportError:
    print("cupy is not installed, the GPU example will be skipped", file=sys.stderr)
    cp = None
import numpy as np

from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch
from cuda.core.experimental.utils import StridedMemoryView, args_viewable_as_strided_memory

# ################################################################################
#
# Usually this entire code block is in a separate file, built as a Python extension
# module that can be imported by users at run time. For illustrative purposes we
# use JIT compilation to make this demo self-contained.
#
# Here we assume an in-place operation, equivalent to the following NumPy code:
#
#   >>> arr = ...
#   >>> assert arr.dtype == np.int32
#   >>> assert arr.ndim == 1
#   >>> arr += np.arange(arr.size, dtype=arr.dtype)
#
# is implemented for both CPU and GPU at low-level, with the following C function
# signature:
func_name = "inplace_plus_arange_N"
func_sig = f"void {func_name}(int* data, size_t N)"

# Here is a concrete (very naive!) implementation on CPU:
if FFI:
    cpu_code = string.Template(r"""
    extern "C"
    $func_sig {
        for (size_t i = 0; i < N; i++) {
            data[i] += i;
        }
    }
    """).substitute(func_sig=func_sig)
    # This is cffi's way of JIT compiling & loading a CPU function. cffi builds an
    # extension module that has the Python binding to the underlying C function.
    # For more details, please refer to cffi's documentation.
    cpu_prog = FFI()
    cpu_prog.cdef(f"{func_sig};")
    cpu_prog.set_source(
        "_cpu_obj",
        cpu_code,
        source_extension=".cpp",
        extra_compile_args=["-std=c++11"],
    )
    cpu_prog.compile()
    cpu_func = getattr(importlib.import_module("_cpu_obj.lib"), func_name)

# Here is a concrete (again, very naive!) implementation on GPU:
if cp:
    gpu_code = string.Template(r"""
    extern "C"
    __global__ $func_sig {
        const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t stride_size = gridDim.x * blockDim.x;
        for (size_t i = tid; i < N; i += stride_size) {
            data[i] += i;
        }
    }
    """).substitute(func_sig=func_sig)

    # To know the GPU's compute capability, we need to identify which GPU to use.
    dev = Device(0)
    dev.set_current()
    arch = "".join(f"{i}" for i in dev.compute_capability)
    gpu_prog = Program(gpu_code, code_type="c++", options=ProgramOptions(arch=f"sm_{arch}", std="c++11"))
    mod = gpu_prog.compile(target_type="cubin")
    gpu_ker = mod.get_kernel(func_name)

# Now we are prepared to run the code from the user's perspective!
#
# ################################################################################


# Below, as a user we want to perform the said in-place operation on either CPU
# or GPU, by calling the corresponding function implemented "elsewhere" (done above).


# We assume the 0-th argument supports either DLPack or CUDA Array Interface (both
# of which are supported by StridedMemoryView).
@args_viewable_as_strided_memory((0,))
def my_func(arr, work_stream):
    # Create a memory view over arr (assumed to be a 1D array of int32). The stream
    # ordering is taken care of, so that arr can be safely accessed on our work
    # stream (ordered after a data stream on which arr is potentially prepared).
    view = arr.view(work_stream.handle if work_stream else -1)
    assert isinstance(view, StridedMemoryView)
    assert len(view.shape) == 1
    assert view.dtype == np.int32

    size = view.shape[0]
    # DLPack also supports host arrays. We want to know if the array data is
    # accessible from the GPU, and dispatch to the right routine accordingly.
    if view.is_device_accessible:
        block = 256
        grid = (size + block - 1) // block
        config = LaunchConfig(grid=grid, block=block, stream=work_stream)
        launch(gpu_ker, config, view.ptr, np.uint64(size))
        # Here we're being conservative and synchronize over our work stream,
        # assuming we do not know the data stream; if we know then we could
        # just order the data stream after the work stream here, e.g.
        #
        #   data_stream.wait(work_stream)
        #
        # without an expensive synchronization (with respect to the host).
        work_stream.sync()
    else:
        cpu_func(cpu_prog.cast("int*", view.ptr), size)


# This takes the CPU path
if FFI:
    # Create input array on CPU
    arr_cpu = np.zeros(1024, dtype=np.int32)
    print(f"before: {arr_cpu[:10]=}")

    # Run the workload
    my_func(arr_cpu, None)

    # Check the result
    print(f"after: {arr_cpu[:10]=}")
    assert np.allclose(arr_cpu, np.arange(1024, dtype=np.int32))


# This takes the GPU path
if cp:
    s = dev.create_stream()
    # Create input array on GPU
    arr_gpu = cp.ones(1024, dtype=cp.int32)
    print(f"before: {arr_gpu[:10]=}")

    # Run the workload
    my_func(arr_gpu, s)

    # Check the result
    print(f"after: {arr_gpu[:10]=}")
    assert cp.allclose(arr_gpu, 1 + cp.arange(1024, dtype=cp.int32))
    s.close()
