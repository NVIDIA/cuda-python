# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import importlib
import random
import subprocess
import sys
from pathlib import Path

import pytest
from cuda.bindings import driver, runtime
from cuda.bindings._internal.utils import get_c_compiler
from cuda.bindings.utils import get_cuda_native_handle, get_minimal_required_cuda_ver_from_ptx_ver, get_ptx_ver

have_cufile = importlib.util.find_spec("cuda.bindings.cufile") is not None

ptx_88_kernel = r"""
.version 8.8
.target sm_75
.address_size 64

	// .globl	empty_kernel

.visible .entry empty_kernel()
{
	ret;
}
"""


ptx_72_kernel = r"""
.version  7.2
.target sm_75
.address_size 64

	// .globl	empty_kernel

.visible .entry empty_kernel()
{
	ret;
}
"""


@pytest.mark.parametrize(
    "kernel,actual_ptx_ver,min_cuda_ver", ((ptx_88_kernel, "8.8", 12090), (ptx_72_kernel, "7.2", 11020))
)
def test_ptx_utils(kernel, actual_ptx_ver, min_cuda_ver):
    ptx_ver = get_ptx_ver(kernel)
    assert ptx_ver == actual_ptx_ver
    cuda_ver = get_minimal_required_cuda_ver_from_ptx_ver(ptx_ver)
    assert cuda_ver == min_cuda_ver


@pytest.mark.parametrize(
    "target",
    (
        driver.CUcontext,
        driver.CUstream,
        driver.CUevent,
        driver.CUmodule,
        driver.CUlibrary,
        driver.CUfunction,
        driver.CUkernel,
        driver.CUgraph,
        driver.CUgraphNode,
        driver.CUgraphExec,
        driver.CUmemoryPool,
        runtime.cudaStream_t,
        runtime.cudaEvent_t,
        runtime.cudaGraph_t,
        runtime.cudaGraphNode_t,
        runtime.cudaGraphExec_t,
        runtime.cudaMemPool_t,
    ),
)
def test_get_handle(target):
    ptr = random.randint(1, 1024)  # noqa: S311
    obj = target(ptr)
    handle = get_cuda_native_handle(obj)
    assert handle == ptr


@pytest.mark.parametrize(
    "target",
    (
        (1, 2, 3, 4),
        [5, 6],
        {},
        None,
    ),
)
def test_get_handle_error(target):
    with pytest.raises(TypeError) as e:
        handle = get_cuda_native_handle(target)


@pytest.mark.parametrize(
    "module",
    # Top-level modules for external Python use
    [
        "driver",
        "nvjitlink",
        "nvrtc",
        "nvvm",
        "runtime",
        *(["cufile"] if have_cufile else []),
    ],
)
def test_cyclical_imports(module):
    subprocess.check_call(  # noqa: S603
        [sys.executable, Path(__file__).parent / "utils" / "check_cyclical_import.py", f"cuda.bindings.{module}"],
    )


def test_get_c_compiler():
    c_compiler = get_c_compiler()
    prefix = ("GCC", "Clang", "MSVC", "Unknown")
    assert sum(c_compiler.startswith(p) for p in prefix) == 1
