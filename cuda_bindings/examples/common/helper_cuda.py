# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from common.helper_string import check_cmd_line_flag, get_cmd_line_argument_int

from cuda.bindings import driver as cuda
from cuda.bindings import nvrtc
from cuda.bindings import runtime as cudart


def _cuda_get_error_enum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError(f"Unknown error type: {error}")


def check_cuda_errors(result):
    if result[0].value:
        raise RuntimeError(f"CUDA error code={result[0].value}({_cuda_get_error_enum(result[0])})")
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def find_cuda_device():
    dev_id = 0
    if check_cmd_line_flag("device="):
        dev_id = get_cmd_line_argument_int("device=")
    check_cuda_errors(cudart.cudaSetDevice(dev_id))
    return dev_id


def find_cuda_device_drv():
    dev_id = 0
    if check_cmd_line_flag("device="):
        dev_id = get_cmd_line_argument_int("device=")
    check_cuda_errors(cuda.cuInit(0))
    cu_device = check_cuda_errors(cuda.cuDeviceGet(dev_id))
    return cu_device
