# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import pathlib
import sys
from typing import Union

from cuda.core._utils.cuda_utils import handle_return

CUDA_PATH = os.environ.get("CUDA_PATH")
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


helpers_root = pathlib.Path(__file__).resolve().parents[3] / "cuda_python_test_helpers"
if helpers_root.is_dir() and str(helpers_root) not in sys.path:
    # Prefer the in-repo helpers over any installed copy.
    sys.path.insert(0, str(helpers_root))

from cuda_python_test_helpers import *  # noqa: E402, F403


@functools.cache
def supports_ipc_mempool(device_id: Union[int, object]) -> bool:
    """Return True if mempool IPC via POSIX file descriptor is supported.

    Uses cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES)
    to check for CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR support. Does not
    require an active CUDA context.
    """
    if IS_WSL:  # noqa: F405
        return False

    try:
        # Lazy import to avoid hard dependency when not running GPU tests
        try:
            from cuda.bindings import driver  # type: ignore
        except Exception:
            from cuda import cuda as driver  # type: ignore

        # Initialize CUDA
        handle_return(driver.cuInit(0))

        # Resolve device id from int or Device-like object
        dev_id = int(getattr(device_id, "device_id", device_id))

        # Query supported mempool handle types bitmask
        attr = driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES
        mask = handle_return(driver.cuDeviceGetAttribute(attr, dev_id))

        # Check POSIX FD handle type support via bitmask
        posix_fd = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        return (int(mask) & int(posix_fd)) != 0
    except Exception:
        return False
