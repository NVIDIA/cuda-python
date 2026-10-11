# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared kernel compilation helpers for graph tests."""

import ctypes

import numpy as np
import pytest

from cuda.bindings import nvrtc
from cuda.core import Device, Program, ProgramOptions
from cuda.core._utils.cuda_utils import NVRTCError, handle_return

# NVRTC diagnostic phrases that indicate cudaGraphConditionalHandle itself is unknown
# to the compiler (older NVRTC builds predate the type). Matched narrowly so a
# genuine compile error (syntax error, etc.) is not hidden as a skip.
# Phrase #1 is the exact diagnostic observed on this machine's NVRTC; phrase #2
# is a common clang/NVRTC wording for an unknown type, not verified against the
# cudaGraphConditionalHandle case on an old NVRTC build.
_COND_HANDLE_UNKNOWN = (
    'identifier "cudaGraphConditionalHandle" is undefined',
    'unknown type name "cudaGraphConditionalHandle"',
)


def skip_if_nvrtc_lacks_conditional_handle(exc):
    """Skip when *exc* means NVRTC predates cudaGraphConditionalHandle.

    Catches only the documented "type unknown" cases; a genuine compile
    error (syntax error, etc.) re-raises so a real bug is not hidden as a skip.
    """
    msg = str(exc)
    if any(phrase in msg for phrase in _COND_HANDLE_UNKNOWN):
        nvrtc_version = handle_return(nvrtc.nvrtcVersion())
        pytest.skip(f"NVRTC version {nvrtc_version} does not support conditionals")
    raise


def compile_common_kernels():
    """Compile basic kernels for graph tests.

    Returns a module with:
    - empty_kernel: does nothing
    - add_one: increments an int pointer by 1
    - write_launch_dims: encodes the launch dimensions in an int
    """
    code = """
    __global__ void empty_kernel() {}
    __global__ void add_one(int *a) { *a += 1; }
    __global__ void write_launch_dims(int *a) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *a = gridDim.x * 1000 + blockDim.x;
        }
    }
    """
    arch = "".join(f"{i}" for i in Device().compute_capability)
    program_options = ProgramOptions(std="c++17", arch=f"sm_{arch}")
    prog = Program(code, code_type="c++", options=program_options)
    mod = prog.compile(
        "cubin",
        name_expressions=("empty_kernel", "add_one", "write_launch_dims"),
    )
    return mod


def compile_conditional_kernels(cond_type):
    """Compile kernels for conditional graph tests.

    Args:
        cond_type: The type of the condition value (bool, np.bool_, ctypes.c_bool, or int)

    Returns a module with:
    - empty_kernel: does nothing
    - add_one: increments an int pointer by 1
    - set_handle: sets a conditional handle value
    - loop_kernel: decrements a counter and updates a conditional handle
    """
    if cond_type in (bool, np.bool_, ctypes.c_bool):
        cond_type_str = "bool"
    elif cond_type is int:
        cond_type_str = "unsigned int"
    else:
        raise ValueError("Unsupported cond_type")

    code = """
    extern "C" __device__ __cudart_builtin__ void CUDARTAPI cudaGraphSetConditional(cudaGraphConditionalHandle handle,
                                                                                    unsigned int value);
    __global__ void empty_kernel() {}
    __global__ void add_one(int *a) { *a += 1; }
    __global__ void set_handle(cudaGraphConditionalHandle handle, $cond_type_str value) {
        cudaGraphSetConditional(handle, value);
    }
    __global__ void loop_kernel(cudaGraphConditionalHandle handle)
    {
        static int count = 10;
        cudaGraphSetConditional(handle, --count ? 1 : 0);
    }
    """.replace("$cond_type_str", cond_type_str)
    arch = "".join(f"{i}" for i in Device().compute_capability)
    program_options = ProgramOptions(std="c++17", arch=f"sm_{arch}")
    prog = Program(code, code_type="c++", options=program_options)
    try:
        mod = prog.compile("cubin", name_expressions=("empty_kernel", "add_one", "set_handle", "loop_kernel"))
    except NVRTCError as exc:
        skip_if_nvrtc_lacks_conditional_handle(exc)
        raise
    return mod


def compile_parallel_kernels():
    """Compile kernels for parallel graph tests.

    Returns a module with:
    - affine: computes *a = *a * m + b
    - combine: computes *s = (*a << 16) | (*b & 0xFFFF)
    - reduce: computes a sum.
    """
    code = """
    __global__ void affine(int *a, int m, int b) { *a = *a * m + b; }
    __global__ void combine(int *s, int *a, int *b) { *s = (*a << 16) | (*b & 0xFFFF); }
    __global__ void reduce(int *out, int *in, size_t n) { for(size_t i=0; i<n; ++i) { *out += in[i]; } }
    """
    arch = "".join(f"{i}" for i in Device().compute_capability)
    program_options = ProgramOptions(std="c++17", arch=f"sm_{arch}")
    prog = Program(code, code_type="c++", options=program_options)
    mod = prog.compile("cubin", name_expressions=("affine", "combine", "reduce"))
    return mod
