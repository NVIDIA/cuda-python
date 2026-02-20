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

import importlib
import shutil
import string
import sys
import tempfile

try:
    from cffi import FFI
except ImportError:
    print("cffi is not installed, the CPU example will be skipped", file=sys.stderr)
    FFI = None
import numpy as np
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


# Below, as a user we want to perform the said in-place operation on a CPU
# or GPU, by calling the corresponding function implemented "elsewhere"
# (in the body of run function).


# We assume the 0-th argument supports either DLPack or CUDA Array Interface (both
# of which are supported by StridedMemoryView).
@args_viewable_as_strided_memory((0,))
def my_func(arr):
    global cpu_func
    global cpu_prog
    # Create a memory view over arr (assumed to be a 1D array of int32). The stream
    # ordering is taken care of, so that arr can be safely accessed on our work
    # stream (ordered after a data stream on which arr is potentially prepared).
    view = arr.view(-1)
    assert isinstance(view, StridedMemoryView)
    assert len(view.shape) == 1
    assert view.dtype == np.int32
    assert not view.is_device_accessible

    size = view.shape[0]
    # DLPack also supports host arrays. We want to know if the array data is
    # accessible from the GPU, and dispatch to the right routine accordingly.
    cpu_func(cpu_prog.cast("int*", view.ptr), size)


def main():
    global my_func
    if not FFI:
        return
    # Here is a concrete (very naive!) implementation on CPU:
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
    temp_dir = tempfile.mkdtemp()
    saved_sys_path = sys.path.copy()
    try:
        cpu_prog.compile(tmpdir=temp_dir)

        sys.path.append(temp_dir)
        cpu_func = getattr(importlib.import_module("_cpu_obj.lib"), func_name)

        # Create input array on CPU
        arr_cpu = np.zeros(1024, dtype=np.int32)
        print(f"before: {arr_cpu[:10]=}")

        # Run the workload
        my_func(arr_cpu)

        # Check the result
        print(f"after: {arr_cpu[:10]=}")
        assert np.allclose(arr_cpu, np.arange(1024, dtype=np.int32))
    finally:
        sys.path = saved_sys_path
        # to allow FFI module to unload, we delete references to
        # to cpu_func
        del cpu_func, my_func
        # clean up temp directory
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
