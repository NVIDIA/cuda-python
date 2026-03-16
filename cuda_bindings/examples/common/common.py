# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE


import numpy as np
from common.helper_cuda import check_cuda_errors

from cuda import pathfinder
from cuda.bindings import driver as cuda
from cuda.bindings import nvrtc
from cuda.bindings import runtime as cudart


def pytest_skipif_compute_capability_too_low(dev_id, required_cc_major_minor):
    import pytest

    cc_major = check_cuda_errors(
        cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, dev_id)
    )
    cc_minor = check_cuda_errors(
        cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, dev_id)
    )
    have_cc_major_minor = (cc_major, cc_minor)
    if have_cc_major_minor < required_cc_major_minor:
        pytest.skip(f"cudaDevAttrComputeCapability too low: {have_cc_major_minor=!r}, {required_cc_major_minor=!r}")


class KernelHelper:
    def __init__(self, code, dev_id):
        include_dirs = []
        for libname in ("cudart", "cccl"):
            hdr_dir = pathfinder.find_nvidia_header_directory(libname)
            if hdr_dir is None:
                import pytest

                pytest.skip(f'pathfinder.find_nvidia_header_directory("{libname}") returned None')
            include_dirs.append(hdr_dir)

        prog = check_cuda_errors(nvrtc.nvrtcCreateProgram(str.encode(code), b"sourceCode.cu", 0, None, None))

        # Initialize CUDA
        check_cuda_errors(cudart.cudaFree(0))

        major = check_cuda_errors(
            cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, dev_id)
        )
        minor = check_cuda_errors(
            cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, dev_id)
        )
        _, nvrtc_minor = check_cuda_errors(nvrtc.nvrtcVersion())
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
            check_cuda_errors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
        except RuntimeError as err:
            log_size = check_cuda_errors(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * log_size
            check_cuda_errors(nvrtc.nvrtcGetProgramLog(prog, log))
            import sys

            print(log.decode(), file=sys.stderr)
            print(err, file=sys.stderr)
            sys.exit(1)

        if use_cubin:
            data_size = check_cuda_errors(nvrtc.nvrtcGetCUBINSize(prog))
            data = b" " * data_size
            check_cuda_errors(nvrtc.nvrtcGetCUBIN(prog, data))
        else:
            data_size = check_cuda_errors(nvrtc.nvrtcGetPTXSize(prog))
            data = b" " * data_size
            check_cuda_errors(nvrtc.nvrtcGetPTX(prog, data))

        self.module = check_cuda_errors(cuda.cuModuleLoadData(np.char.array(data)))

    def get_function(self, name):
        return check_cuda_errors(cuda.cuModuleGetFunction(self.module, name))
