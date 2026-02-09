# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

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
            info = nvml.device_get_pci_info_v3(handle)
            assert isinstance(info.bus, int)
            assert isinstance(name, str)
            result.append({"name": name, "id": info.bus})

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


def test_cuda_device_order(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    cuda_devices = get_cuda_device_names()
    nvml_devices = get_nvml_device_names()

    assert cuda_devices == nvml_devices, "CUDA and NVML device lists do not match"
