# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Shared kernel compilation helpers for graph tests."""

import ctypes

import numpy as np
import pytest

try:
    from cuda.bindings import nvrtc
except ImportError:
    from cuda import nvrtc

from cuda.core import Device, Program, ProgramOptions
from cuda.core._utils.cuda_utils import NVRTCError, handle_return


def compile_common_kernels():
    """Compile basic kernels for graph tests.

    Returns a module with:
    - empty_kernel: does nothing
    - add_one: increments an int pointer by 1
    """
    code = """
    __global__ void empty_kernel() {}
    __global__ void add_one(int *a) { *a += 1; }
    """
    arch = "".join(f"{i}" for i in Device().compute_capability)
    program_options = ProgramOptions(std="c++17", arch=f"sm_{arch}")
    prog = Program(code, code_type="c++", options=program_options)
    mod = prog.compile("cubin", name_expressions=("empty_kernel", "add_one"))
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
    except NVRTCError as e:
        with pytest.raises(NVRTCError, match='error: identifier "cudaGraphConditionalHandle" is undefined'):
            raise e
        nvrtcVersion = handle_return(nvrtc.nvrtcVersion())
        pytest.skip(f"NVRTC version {nvrtcVersion} does not support conditionals")
    return mod
