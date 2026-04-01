# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import atexit

from cuda.bindings import driver as cuda
from cuda.bindings import nvrtc

_ctx = None
_device = None
_persistent_ptrs: list[int] = []
_modules: list = []


def assert_drv(err) -> None:
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Cuda Error: {err}")


def ensure_context() -> int:
    global _ctx, _device
    if _ctx is not None:
        return _ctx

    (err,) = cuda.cuInit(0)
    assert_drv(err)

    err, device = cuda.cuDeviceGet(0)
    assert_drv(err)
    _device = device

    err, ctx = cuda.cuCtxCreate(None, 0, device)
    assert_drv(err)
    _ctx = ctx
    return ctx


def alloc_persistent(size: int) -> int:
    ensure_context()
    err, ptr = cuda.cuMemAlloc(size)
    assert_drv(err)
    _persistent_ptrs.append(ptr)
    return ptr


def compile_and_load(kernel_source: str) -> int:
    """Compile CUDA C source and returns the CUmodule handle """
    ensure_context()

    err, major = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _device
    )
    assert_drv(err)
    err, minor = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, _device
    )
    assert_drv(err)

    err, prog = nvrtc.nvrtcCreateProgram(
        kernel_source.encode(), b"benchmark_kernel.cu", 0, [], []
    )
    assert_drv(err)

    arch_flag = f"--gpu-architecture=sm_{major}{minor}".encode()
    (err,) = nvrtc.nvrtcCompileProgram(prog, 2, [b"--fmad=false", arch_flag])

    # check for compile errors
    err_log, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
    assert_drv(err_log)
    log = b" " * log_size
    (err_log,) = nvrtc.nvrtcGetProgramLog(prog, log)
    assert_drv(err_log)
    assert_drv(err)

    err, cubin_size = nvrtc.nvrtcGetCUBINSize(prog)
    assert_drv(err)
    cubin = b" " * cubin_size
    (err,) = nvrtc.nvrtcGetCUBIN(prog, cubin)
    assert_drv(err)

    err, module = cuda.cuModuleLoadData(cubin)
    assert_drv(err)
    _modules.append(module)
    return module


def cleanup() -> None:
    global _ctx
    for ptr in reversed(_persistent_ptrs):
        (err,) = cuda.cuMemFree(ptr)
        assert_drv(err)
    _persistent_ptrs.clear()

    for module in reversed(_modules):
        (err,) = cuda.cuModuleUnload(module)
        assert_drv(err)
    _modules.clear()

    if _ctx is None:
        return
    (err,) = cuda.cuCtxDestroy(_ctx)
    assert_drv(err)
    _ctx = None


atexit.register(cleanup)
