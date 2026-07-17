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
# dependencies = ["cuda-python>=13.0.0", "cuda-core>=1.0.0", "numpy>=2.3.2", "cffi", "setuptools"]
# ///

"""
StridedMemoryView + foreign CPU function via cffi

This sample models a library that accepts "any array-like object" and
dispatches to a JIT-compiled CPU implementation. The library exposes a
single decorator, ``@args_viewable_as_strided_memory``, which turns the
selected function arguments into ``StridedMemoryView`` instances at call
time -- regardless of whether the caller passed a NumPy array, a CuPy
array, a DLPack capsule, or a raw ``Buffer``.

The C-side operation:

    void inplace_plus_arange_N(int* data, size_t N);

adds ``i`` to ``data[i]`` for each ``i`` in ``[0, N)``. It is compiled to a
native shared library via ``cffi`` at runtime, then invoked from Python
through the pointer that ``StridedMemoryView`` provides.

Together with `stridedMemoryViewGpu/` this is the "practical" side of the
StridedMemoryView story: the library never needs to know which array
protocol its caller uses.
"""

import importlib
import string
import sys
import tempfile
from contextlib import contextmanager

try:
    import numpy as np
    from cffi import FFI

    from cuda.core.utils import StridedMemoryView, args_viewable_as_strided_memory
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


# ---------------------------------------------------------------------------
# The C function we JIT compile. In a real library this lives in a compiled
# extension module and is imported at run time; here we build it in a temp
# directory to keep the sample self-contained.
# ---------------------------------------------------------------------------
func_name = "inplace_plus_arange_N"
func_sig = f"void {func_name}(int* data, size_t N)"


# ---------------------------------------------------------------------------
# The library-facing entry point.
#
# ``@args_viewable_as_strided_memory((0,))`` says "take argument 0 and make
# it available as a StridedMemoryView on the callee side". Stream ordering
# is handled by the decorator so the library can safely access the memory
# on its own work stream.
# ---------------------------------------------------------------------------
@args_viewable_as_strided_memory((0,))
def my_func(arr, cpu_prog, cpu_func):
    """In-place ``arr += arange(len(arr))`` via a foreign CPU function."""
    view = arr.view(-1)  # -1: no stream ordering required (host memory)
    assert isinstance(view, StridedMemoryView)
    assert len(view.shape) == 1
    assert view.dtype == np.int32
    assert not view.is_device_accessible

    size = view.shape[0]
    # Convert the raw pointer inside the view into a cffi ``int*`` and call
    # the compiled function. This is the only line that touches the pointer.
    cpu_func(cpu_prog.cast("int*", view.ptr), size)


# ---------------------------------------------------------------------------
# JIT compilation of the C function via cffi.
# ---------------------------------------------------------------------------
def _create_cpu_program():
    cpu_code = string.Template(r"""
    extern "C"
    $func_sig {
        for (size_t i = 0; i < N; i++) {
            data[i] += i;
        }
    }
    """).substitute(func_sig=func_sig)
    cpu_prog = FFI()
    cpu_prog.cdef(f"{func_sig};")
    cpu_prog.set_source(
        "_cpu_obj",
        cpu_code,
        source_extension=".cpp",
        extra_compile_args=["-std=c++11"],
    )
    return cpu_prog


@contextmanager
def _compiled_cpu_func(cpu_prog, temp_dir):
    saved_sys_path = sys.path.copy()
    try:
        cpu_prog.compile(tmpdir=temp_dir)
        sys.path.append(temp_dir)
        cpu_func = getattr(importlib.import_module("_cpu_obj.lib"), func_name)
        yield cpu_func
    finally:
        sys.path = saved_sys_path
        # Ensure cffi modules are unloadable before the temp dir is removed.
        sys.modules.pop("_cpu_obj.lib", None)
        sys.modules.pop("_cpu_obj", None)


def main():
    cpu_prog = _create_cpu_program()
    with tempfile.TemporaryDirectory() as temp_dir, _compiled_cpu_func(cpu_prog, temp_dir) as cpu_func:
        arr_cpu = np.zeros(1024, dtype=np.int32)
        print(f"before: {arr_cpu[:10]=}")
        my_func(arr_cpu, cpu_prog, cpu_func)
        print(f"after:  {arr_cpu[:10]=}")
        assert np.allclose(arr_cpu, np.arange(1024, dtype=np.int32))
        print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
