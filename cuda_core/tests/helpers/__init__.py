# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os

from cuda.core._utils.cuda_utils import handle_return
from cuda.pathfinder import get_cuda_path_or_home
from cuda_python_test_helpers import *

CUDA_PATH = get_cuda_path_or_home()
CUDA_INCLUDE_PATH = None
CCCL_INCLUDE_PATHS = None
if CUDA_PATH is not None:
    path = os.path.join(CUDA_PATH, "include")
    if os.path.isdir(path):
        CUDA_INCLUDE_PATH = path
        CCCL_INCLUDE_PATHS = (path,)
        path = os.path.join(path, "cccl")
        if os.path.isdir(path):
            CCCL_INCLUDE_PATHS = (path,) + CCCL_INCLUDE_PATHS


@functools.cache
def supports_ipc_mempool(device_id: int | object) -> bool:
    """Return True if mempool IPC via POSIX file descriptor is supported.

    Uses cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES)
    to check for CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR support. Does not
    require an active CUDA context.

    Only CUDA_ERROR_NOT_SUPPORTED (the documented "attribute not
    available" result) is treated as "unsupported"; other driver errors
    (invalid device, deinitialized driver) propagate so a real
    bug is not hidden as "unsupported".
    """
    if IS_WSL:
        return False

    # Lazy import to avoid hard dependency when not running GPU tests
    from cuda.bindings import driver  # type: ignore

    # Initialize CUDA
    handle_return(driver.cuInit(0))

    # Resolve device id from int or Device-like object
    dev_id = int(getattr(device_id, "device_id", device_id))

    # Query supported mempool handle types bitmask. Inspect the raw CUresult
    # so only the documented "not available" case is treated as unsupported.
    attr = driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES
    result, mask = driver.cuDeviceGetAttribute(attr, dev_id)
    if result == driver.CUresult.CUDA_ERROR_NOT_SUPPORTED:
        return False
    handle_return((result, mask))  # raise CUDAError for other driver errors

    # Check POSIX FD handle type support via bitmask
    posix_fd = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    return (int(mask) & int(posix_fd)) != 0
