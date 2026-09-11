# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest

import cuda.bindings.driver as cuda
from cuda.bindings import nvml

from .conftest import NVMLInitializer


def get_nvml_device_names():
    result = []
    with NVMLInitializer():
        # uses NVML Library to get the device count, device id and device pci id
        num_devices = nvml.device_get_count_v2()
        for idx in range(num_devices):
            handle = nvml.device_get_handle_by_index_v2(idx)
            name = nvml.device_get_name(handle)
            try:
                info = nvml.device_get_pci_info_v3(handle)
            except nvml.NotSupportedError:
                bus_id = -1
            else:
                bus_id = info.bus
            assert isinstance(bus_id, int)
            assert isinstance(name, str)
            result.append({"name": name, "id": bus_id})

    return result


def get_cuda_device_names(sort_by_bus_id=True):
    result = []

    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device_count = cuda.cuDeviceGetCount()
    assert err == cuda.CUresult.CUDA_SUCCESS

    for dev in range(device_count):
        size = 256
        err, name = cuda.cuDeviceGetName(size, dev)
        name = name.split(b"\x00")[0].decode()
        assert err == cuda.CUresult.CUDA_SUCCESS

        err, pci_bus_id = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev)
        assert err == cuda.CUresult.CUDA_SUCCESS
        assert isinstance(pci_bus_id, int)

        result.append({"name": name, "id": pci_bus_id})

    if sort_by_bus_id:
        result = sorted(result, key=lambda k: k["id"])

    return result


def test_cuda_device_order():
    cuda_devices = get_cuda_device_names()
    nvml_devices = get_nvml_device_names()

    for kind in ("Orin", "Thor"):
        if any(kind in device["name"] for device in nvml_devices):
            pytest.skip(f"Skipping test on {kind}, which has non-standard device naming")

    def compare(cuda_device, nvml_device):
        return cuda_device["name"] == nvml_device["name"] and (
            nvml_device["id"] == -1 or cuda_device["id"] == nvml_device["id"]
        )

    assert len(cuda_devices) <= len(nvml_devices)
    for cuda_device in cuda_devices:
        assert any(compare(cuda_device, nvml_device) for nvml_device in nvml_devices)
