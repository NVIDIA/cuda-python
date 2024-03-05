# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import pytest
from cuda import cuda, cudart, nvrtc
import numpy as np

def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError('Cuda Error: {}'.format(err))
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError('Cudart Error: {}'.format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError('Nvrtc Error: {}'.format(err))
    else:
        raise RuntimeError('Unknown error type: {}'.format(err))

@pytest.fixture
def init_cuda():
    # Initialize
    err, = cuda.cuInit(0)
    ASSERT_DRV(err)
    err, device = cuda.cuDeviceGet(0)
    ASSERT_DRV(err)
    err, ctx = cuda.cuCtxCreate(0, device)
    ASSERT_DRV(err)

    # create stream
    err, stream = cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING.value)
    ASSERT_DRV(err)

    yield device, ctx, stream

    err, = cuda.cuStreamDestroy(stream)
    ASSERT_DRV(err)
    err, = cuda.cuCtxDestroy(ctx)
    ASSERT_DRV(err)

@pytest.fixture
def load_module():
    module = None
    def _load_module(kernel_string, device):
        nonlocal module
        # Get module
        err, major = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)
        ASSERT_DRV(err)
        err, minor = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)
        ASSERT_DRV(err)

        err, prog = nvrtc.nvrtcCreateProgram(str.encode(kernel_string), b'kernelString.cu', 0, [], [])
        ASSERT_DRV(err)
        opts = [b'--fmad=false', bytes('--gpu-architecture=sm_' + str(major) + str(minor), 'ascii')]
        err, = nvrtc.nvrtcCompileProgram(prog, 2, opts)

        err_log, logSize = nvrtc.nvrtcGetProgramLogSize(prog)
        ASSERT_DRV(err_log)
        log = b' ' * logSize
        err_log, = nvrtc.nvrtcGetProgramLog(prog, log)
        ASSERT_DRV(err_log)
        result = log.decode()
        if len(result) > 1:
            print(result)

        ASSERT_DRV(err)
        err, cubinSize = nvrtc.nvrtcGetCUBINSize(prog)
        ASSERT_DRV(err)
        cubin = b' ' * cubinSize
        err, = nvrtc.nvrtcGetCUBIN(prog, cubin)
        ASSERT_DRV(err)
        cubin = np.char.array(cubin)
        err, module = cuda.cuModuleLoadData(cubin)
        ASSERT_DRV(err)

        return module

    yield _load_module

    err, = cuda.cuModuleUnload(module)
    ASSERT_DRV(err)
