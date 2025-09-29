# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from typing import Union


def _detect_wsl() -> bool:
    try:
        with open("/proc/sys/kernel/osrelease") as f:
            data = f.read().lower()
        if "microsoft" in data or "wsl" in data:
            return True
    except Exception:
        pass
    return any(os.environ.get(k) for k in ("WSL_DISTRO_NAME", "WSL_INTEROP"))


IS_WSL: bool = _detect_wsl()

skip_on_wsl = pytest.mark.skipif(IS_WSL, reason="WSL does not support this test")


def supports_ipc_mempool(device_id: Union[int, object]) -> bool:
    """Return True if the driver accepts creating an IPC-enabled mempool.

    Attempts to create a mempool with POSIX FD handle type using the CUDA driver.
    Returns False if the operation is rejected or raises.
    """
    try:
        # Lazy import to avoid hard dependency when not running GPU tests
        try:
            from cuda.bindings import driver
        except Exception:
            from cuda import cuda as driver  # type: ignore

        # Build location for the provided device id (int or Device-like with device_id)
        dev_id = getattr(device_id, "device_id", device_id)
        loc = driver.CUmemLocation()
        loc.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        loc.id = int(dev_id)

        props = driver.CUmemPoolProps()
        props.allocType = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        props.handleTypes = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        props.location = loc
        props.maxSize = 2_097_152
        props.win32SecurityAttributes = 0
        props.usage = 0

        res, pool = driver.cuMemPoolCreate(props)
        if int(res.value) != 0:
            return False
        # Destroy created pool to avoid leaks
        try:
            driver.cuMemPoolDestroy(pool)
        except Exception:
            pass
        return True
    except Exception:
        return False


