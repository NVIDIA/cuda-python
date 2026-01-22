# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import os

import numpy as np
from common.helper_cuda import checkCudaErrors
from cuda.bindings import driver as cuda
from cuda.bindings import nvrtc
from cuda.bindings import runtime as cudart
from cuda.pathfinder._utils.env_vars import get_cuda_home_or_path


def pytest_skipif_cuda_include_not_found():
    import pytest

    cuda_home = get_cuda_home_or_path()
    if cuda_home is None:
        pytest.skip("CUDA_HOME/CUDA_PATH not set")
    cuda_include = os.path.join(cuda_home, "include")
    if not os.path.exists(cuda_include):
        pytest.skip(f"$CUDA_HOME/include does not exist: '{cuda_include}'")


def pytest_skipif_compute_capability_too_low(devID, required_cc_major_minor):
    import pytest

    cc_major = checkCudaErrors(
        cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, devID)
    )
    cc_minor = checkCudaErrors(
        cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, devID)
    )
    have_cc_major_minor = (cc_major, cc_minor)
    if have_cc_major_minor < required_cc_major_minor:
        pytest.skip(f"cudaDevAttrComputeCapability too low: {have_cc_major_minor=!r}, {required_cc_major_minor=!r}")


class KernelHelper:
    def __init__(self, code, devID):
        prog = checkCudaErrors(nvrtc.nvrtcCreateProgram(str.encode(code), b"sourceCode.cu", 0, None, None))

        cuda_home = get_cuda_home_or_path()
        assert cuda_home is not None
        cuda_include = os.path.join(cuda_home, "include")
        assert os.path.isdir(cuda_include)
        include_dirs = [cuda_include]
        cccl_include = os.path.join(cuda_include, "cccl")
        if os.path.isdir(cccl_include):
            include_dirs.insert(0, cccl_include)

        # Initialize CUDA
        checkCudaErrors(cudart.cudaFree(0))

        major = checkCudaErrors(
            cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, devID)
        )
        minor = checkCudaErrors(
            cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, devID)
        )
        _, nvrtc_minor = checkCudaErrors(nvrtc.nvrtcVersion())
        use_cubin = nvrtc_minor >= 1
        prefix = "sm" if use_cubin else "compute"
        arch_arg = bytes(f"--gpu-architecture={prefix}_{major}{minor}", "ascii")

        opts = [
            b"--fmad=true",
            arch_arg,
            b"--std=c++17",
            b"-default-device",
        ]
        for inc_dir in include_dirs:
            opts.append(f"--include-path={inc_dir}".encode())

        try:
            checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
        except RuntimeError as err:
            logSize = checkCudaErrors(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * logSize
            checkCudaErrors(nvrtc.nvrtcGetProgramLog(prog, log))
            print(log.decode())
            print(err)
            exit(-1)

        if use_cubin:
            dataSize = checkCudaErrors(nvrtc.nvrtcGetCUBINSize(prog))
            data = b" " * dataSize
            checkCudaErrors(nvrtc.nvrtcGetCUBIN(prog, data))
        else:
            dataSize = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
            data = b" " * dataSize
            checkCudaErrors(nvrtc.nvrtcGetPTX(prog, data))

        self.module = checkCudaErrors(cuda.cuModuleLoadData(np.char.array(data)))

    def getFunction(self, name):
        return checkCudaErrors(cuda.cuModuleGetFunction(self.module, name))
