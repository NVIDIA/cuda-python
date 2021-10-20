# Copyright 2021 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
from cuda import cuda, cudart, nvrtc
from examples.common.helper_string import getCmdLineArgumentInt, checkCmdLineFlag

def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))

def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]

def findCudaDevice():
    devID = 0
    if checkCmdLineFlag("device="):
        devID = getCmdLineArgumentInt("device=")
    checkCudaErrors(cudart.cudaSetDevice(devID))
    return devID

def findCudaDeviceDRV():
    devID = 0
    if checkCmdLineFlag("device="):
        devID = getCmdLineArgumentInt("device=")
    checkCudaErrors(cuda.cuInit(0))
    cuDevice = checkCudaErrors(cuda.cuDeviceGet(devID))
    return cuDevice
