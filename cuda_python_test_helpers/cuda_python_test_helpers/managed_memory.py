# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import cache

import pytest

try:
    from cuda.bindings import driver  # type: ignore
except Exception:
    from cuda import cuda as driver  # type: ignore


def _resolve_device_id(device) -> int:
    if device is None:
        return 0
    if hasattr(device, "device_id"):
        return int(device.device_id)
    try:
        return int(device)
    except Exception:
        return 0


def _cu_init_ok() -> bool:
    try:
        (err,) = driver.cuInit(0)
    except Exception:
        return False
    return err == driver.CUresult.CUDA_SUCCESS


@cache
def _get_concurrent_managed_access(device_id: int) -> int | None:
    if not _cu_init_ok():
        return None
    try:
        attr = driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
    except Exception:
        return None
    try:
        err, value = driver.cuDeviceGetAttribute(attr, device_id)
    except Exception:
        return None
    if err != driver.CUresult.CUDA_SUCCESS:
        return None
    return int(value)


def managed_memory_skip_reason(device=None) -> str | None:
    """Return a skip reason when managed memory should be avoided."""
    if not hasattr(driver, "cuMemAllocManaged"):
        return "cuMemAllocManaged is unavailable; treating concurrent managed access as disabled"
    device_id = _resolve_device_id(device)
    value = _get_concurrent_managed_access(device_id)
    if value is None:
        return "Unable to query CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS"
    if value == 0:
        return "CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS == 0"
    return None


def skip_if_concurrent_managed_access_disabled(device=None) -> None:
    reason = managed_memory_skip_reason(device)
    if reason:
        pytest.skip(reason)
