# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import sys

import pytest

from cuda.bindings import driver, runtime


def is_windows_mcdm_device(device=0):
    if sys.platform != "win32":
        return False
    import cuda.bindings.nvml as nvml

    device_id = int(getattr(device, "device_id", device))
    (err,) = driver.cuInit(0)
    if err != driver.CUresult.CUDA_SUCCESS:
        return False
    err, pci_bus_id = driver.cuDeviceGetPCIBusId(13, device_id)
    if err != driver.CUresult.CUDA_SUCCESS:
        return False
    pci_bus_id = pci_bus_id.split(b"\x00", 1)[0].decode("ascii")
    nvml.init_v2()
    try:
        handle = nvml.device_get_handle_by_pci_bus_id_v2(pci_bus_id)
        current, _ = nvml.device_get_driver_model_v2(handle)
        return current == nvml.DriverModel.DRIVER_MCDM
    finally:
        nvml.shutdown()


def xfail_if_mempool_oom(err_or_exc, api_name=None, device=0):
    if api_name is not None and not isinstance(api_name, str):
        device = api_name
        api_name = None

    is_oom = err_or_exc in (
        driver.CUresult.CUDA_ERROR_OUT_OF_MEMORY,
        runtime.cudaError_t.cudaErrorMemoryAllocation,
    ) or "CUDA_ERROR_OUT_OF_MEMORY" in str(err_or_exc)

    if not is_oom:
        return
    try:
        is_windows_mcdm = is_windows_mcdm_device(device)
    except Exception:
        # If MCDM detection fails, leave the primary test failure visible.
        return
    if not is_windows_mcdm:
        return

    api_context = f"{api_name} " if api_name else ""
    pytest.xfail(f"{api_context}could not reserve VA for mempool operations on Windows MCDM")
