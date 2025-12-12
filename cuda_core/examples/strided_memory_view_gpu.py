# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This demo illustrates:
#
#   1. The similarity between CPU and GPU JIT-compilation with C++ sources
#   2. How to use StridedMemoryView to interface with foreign C/C++ functions
#
# This demo uses cffi (https://cffi.readthedocs.io/) for the CPU path, which can be
# easily installed from pip or conda following their instructions.
#
# ################################################################################

import string
import sys

try:
    import cupy as cp
except ImportError:
    print("cupy is not installed, the GPU example will be skipped", file=sys.stderr)
    cp = None
import numpy as np
from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch
from cuda.core.utils import StridedMemoryView, args_viewable_as_strided_memory

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

# Now we are prepared to run the code from the user's perspective!
#
# ################################################################################


# Below, as a user we want to perform the said in-place operation on either CPU
# or GPU, by calling the corresponding function implemented "elsewhere" (done above).


# We assume the 0-th argument supports either DLPack or CUDA Array Interface (both
# of which are supported by StridedMemoryView).
@args_viewable_as_strided_memory((0,))
def my_func(arr, work_stream, gpu_ker):
    # Create a memory view over arr (assumed to be a 1D array of int32). The stream
    # ordering is taken care of, so that arr can be safely accessed on our work
    # stream (ordered after a data stream on which arr is potentially prepared).
    view = arr.view(work_stream.handle if work_stream else -1)
    assert isinstance(view, StridedMemoryView)
    assert len(view.shape) == 1
    assert view.dtype == np.int32
    assert view.is_device_accessible

    size = view.shape[0]
    # DLPack also supports host arrays. We want to know if the array data is
    # accessible from the GPU, and dispatch to the right routine accordingly.
    block = 256
    grid = (size + block - 1) // block
    config = LaunchConfig(grid=grid, block=block)
    launch(work_stream, config, gpu_ker, view.ptr, np.uint64(size))
    # Here we're being conservative and synchronize over our work stream,
    # assuming we do not know the data stream; if we know then we could
    # just order the data stream after the work stream here, e.g.
    #
    #   data_stream.wait(work_stream)
    #
    # without an expensive synchronization (with respect to the host).
    work_stream.sync()


def run():
    global my_func
    if not cp:
        return None
    # Here is a concrete (very naive!) implementation on GPU:
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
    gpu_prog = Program(gpu_code, code_type="c++", options=ProgramOptions(arch=f"sm_{dev.arch}", std="c++11"))
    mod = gpu_prog.compile(target_type="cubin")
    gpu_ker = mod.get_kernel(func_name)

    s = dev.create_stream()
    try:
        # Create input array on GPU
        arr_gpu = cp.ones(1024, dtype=cp.int32)
        print(f"before: {arr_gpu[:10]=}")

        # Run the workload
        my_func(arr_gpu, s, gpu_ker)

        # Check the result
        print(f"after: {arr_gpu[:10]=}")
        assert cp.allclose(arr_gpu, 1 + cp.arange(1024, dtype=cp.int32))
    finally:
        s.close()


if __name__ == "__main__":
    run()
