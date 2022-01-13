# Copyright 2021-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import ctypes
import numpy as np
import os
from cuda import cuda, cudart, nvrtc
from examples.common.helper_cuda import checkCudaErrors

class KernelHelper:
    def __init__(self, code, devID):
        prog = checkCudaErrors(nvrtc.nvrtcCreateProgram(str.encode(code), b'sourceCode.cu', 0, [], []))
        CUDA_HOME = os.getenv('CUDA_HOME')
        if CUDA_HOME == None:
            raise RuntimeError('Environment variable CUDA_HOME is not defined')
        include_dirs = os.path.join(CUDA_HOME, 'include')

        # Initialize CUDA
        checkCudaErrors(cudart.cudaFree(0))

        major = checkCudaErrors(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, devID))
        minor = checkCudaErrors(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, devID))
        _, nvrtc_minor = checkCudaErrors(nvrtc.nvrtcVersion())
        use_cubin = (nvrtc_minor >= 1)
        prefix = 'sm' if use_cubin else 'compute'
        arch_arg = bytes(f'--gpu-architecture={prefix}_{major}{minor}', 'ascii')

        try:
            opts = [b'--fmad=true', arch_arg, '--include-path={}'.format(include_dirs).encode('UTF-8'),
                    b'--std=c++11', b'-default-device']
            checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
        except RuntimeError as err:
            logSize = checkCudaErrors(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b' ' * logSize
            checkCudaErrors(nvrtc.nvrtcGetProgramLog(prog, log))
            print(log.decode())
            print(err)
            exit(-1)

        if use_cubin:
            dataSize = checkCudaErrors(nvrtc.nvrtcGetCUBINSize(prog))
            data = b' ' * dataSize
            checkCudaErrors(nvrtc.nvrtcGetCUBIN(prog, data))
        else:
            dataSize = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
            data = b' ' * dataSize
            checkCudaErrors(nvrtc.nvrtcGetPTX(prog, data))

        self.module = checkCudaErrors(cuda.cuModuleLoadData(np.char.array(data)))

    def getFunction(self, name):
        return checkCudaErrors(cuda.cuModuleGetFunction(self.module, name))
