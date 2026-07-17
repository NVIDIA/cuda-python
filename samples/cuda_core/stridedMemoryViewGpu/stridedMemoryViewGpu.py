# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# /// script
# dependencies = ["cuda-python>=13.0.0", "cuda-core>=1.0.0", "cupy-cuda13x>=14.0.0", "numpy>=2.3.2"]
# ///

"""
StridedMemoryView + foreign GPU kernel via NVRTC

The GPU-side counterpart to :file:`stridedMemoryViewCpu.py`. Same public
Python entry point (``my_func`` decorated with
``@args_viewable_as_strided_memory((0,))``), but this time the decorated
function dispatches to a CUDA kernel compiled at runtime with NVRTC.

The C-side operation is:

    __global__ void inplace_plus_arange_N(int* data, size_t N);

which sets ``data[i] += i`` for each ``i`` in ``[0, N)``. The decorator
lifts the caller's CuPy array into a ``StridedMemoryView`` that carries a
device pointer, a dtype, and a shape. The library never needs to know that
the caller used CuPy -- it just launches the kernel through the view's
``ptr``.

Together with `stridedMemoryViewCpu/` this is the "practical" side of the
StridedMemoryView story: the library never needs to know which array
protocol its caller uses.
"""

import string
import sys

try:
    import cupy as cp
    import numpy as np

    from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch
    from cuda.core.utils import StridedMemoryView, args_viewable_as_strided_memory
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


# ---------------------------------------------------------------------------
# The kernel that plays the role of the "foreign" library entry point.
# ---------------------------------------------------------------------------
func_name = "inplace_plus_arange_N"
func_sig = f"void {func_name}(int* data, size_t N)"


# ---------------------------------------------------------------------------
# The library-facing entry point.
#
# ``@args_viewable_as_strided_memory((0,))`` says "take argument 0 and make
# it available as a StridedMemoryView on the callee side". Stream ordering
# is handled by the decorator: we pass ``work_stream.handle`` so the view
# waits until the caller's data has landed before the kernel reads it.
# ---------------------------------------------------------------------------
@args_viewable_as_strided_memory((0,))
def my_func(arr, work_stream, kernel):
    """In-place ``arr += arange(len(arr))`` via a foreign GPU kernel."""
    view = arr.view(work_stream.handle if work_stream else -1)
    assert isinstance(view, StridedMemoryView)
    assert len(view.shape) == 1
    assert view.dtype == np.int32
    assert view.is_device_accessible

    size = view.shape[0]
    block = 256
    grid = (size + block - 1) // block
    config = LaunchConfig(grid=grid, block=block)
    launch(work_stream, config, kernel, view.ptr, np.uint64(size))
    # Conservative: synchronize the work stream so the caller can observe
    # the write immediately. If we knew the caller's data stream we could
    # instead do ``data_stream.wait(work_stream)`` and avoid the host sync.
    work_stream.sync()


def main():
    # ---- Compile the GPU function ----
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

    dev = Device(0)
    dev.set_current()
    prog = Program(
        gpu_code,
        code_type="c++",
        options=ProgramOptions(arch=f"sm_{dev.arch}", std="c++11"),
    )
    mod = prog.compile(target_type="cubin")
    kernel = mod.get_kernel(func_name)

    stream = dev.create_stream()
    try:
        # ---- Call the library through the decorated wrapper ----
        arr_gpu = cp.ones(1024, dtype=cp.int32)
        print(f"before: {arr_gpu[:10]=}")

        my_func(arr_gpu, stream, kernel)

        print(f"after:  {arr_gpu[:10]=}")
        assert cp.allclose(arr_gpu, 1 + cp.arange(1024, dtype=cp.int32))
        print("Done")
        return 0
    finally:
        stream.close()


if __name__ == "__main__":
    sys.exit(main())
