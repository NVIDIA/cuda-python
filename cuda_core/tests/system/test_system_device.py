# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402

from .conftest import skip_if_nvml_unsupported

pytestmark = skip_if_nvml_unsupported

import os
import re
import sys

import pytest
from cuda.core import system
from cuda.core.system import device as system_device


if system.HAS_WORKING_NVML:
    from cuda.bindings import _nvml as nvml

    if system.get_num_devices() == 0:
        pytest.skip("No GPUs available to run device tests", allow_module_level=True)


def test_device_index_handle():
    for device in system.Device.get_all_devices():
        assert isinstance(device.handle, int)


def test_device_architecture():
    for device in system.Device.get_all_devices():
        device_arch = device.architecture

        assert isinstance(device_arch, system_device.DeviceArchitecture)
        if sys.version_info < (3, 12):
            assert device_arch.id in nvml.DeviceArch.__members__.values()
        else:
            assert device_arch.id in nvml.DeviceArch


def test_device_bar1_memory():
    for device in system.Device.get_all_devices():
        bar1_memory_info = device.bar1_memory_info
        free, total, used = (
            bar1_memory_info.free,
            bar1_memory_info.total,
            bar1_memory_info.used,
        )

        assert isinstance(bar1_memory_info, system_device.BAR1MemoryInfo)
        assert isinstance(free, int)
        assert isinstance(total, int)
        assert isinstance(used, int)

        assert free >= 0
        assert total >= 0
        assert used >= 0
        assert free + used == total


def test_device_cpu_affinity():
    skip_reasons = set()
    for device in system.Device.get_all_devices():
        try:
            affinity = device.cpu_affinity
        except nvml.NotSupportedError:
            skip_reasons.add(f"CPU affinity not supported on {device}")
        else:
            assert isinstance(affinity, list)
            os.sched_setaffinity(0, affinity)
            assert os.sched_getaffinity(0) == set(affinity)
    if skip_reasons:
        pytest.skip(" ; ".join(skip_reasons))


def test_device_cuda_compute_capability():
    for device in system.Device.get_all_devices():
        cuda_compute_capability = device.cuda_compute_capability
        assert isinstance(cuda_compute_capability, tuple)
        assert len(cuda_compute_capability) == 2
        assert all([isinstance(i, int) for i in cuda_compute_capability])
        assert 3 <= cuda_compute_capability[0] <= 99
        assert 0 <= cuda_compute_capability[1] <= 9


def test_device_memory():
    for device in system.Device.get_all_devices():
        memory_info = device.memory_info
        free, total, used, reserved = memory_info.free, memory_info.total, memory_info.used, memory_info.reserved

        assert isinstance(memory_info, system_device.MemoryInfo)
        assert isinstance(free, int)
        assert isinstance(total, int)
        assert isinstance(used, int)
        assert isinstance(reserved, int)

        assert free >= 0
        assert total >= 0
        assert used >= 0
        assert reserved >= 0
        assert free + used + reserved == total


def test_device_name():
    for device in system.Device.get_all_devices():
        name = device.name
        assert isinstance(name, str)
        assert len(name) > 0


def test_device_pci_info():
    for device in system.Device.get_all_devices():
        pci_info = device.pci_info
        assert isinstance(pci_info, system_device.PciInfo)

        assert isinstance(pci_info.bus_id, str)
        assert re.match("[a-f0-9]{8}:[a-f0-9]{2}:[a-f0-9]{2}.[a-f0-9]", pci_info.bus_id.lower())
        bus_id_domain = int(pci_info.bus_id.split(":")[0], 16)
        bus_id_bus = int(pci_info.bus_id.split(":")[1], 16)
        bus_id_device = int(pci_info.bus_id.split(":")[2][:2], 16)

        assert isinstance(pci_info.domain, int)
        assert 0x00 <= pci_info.domain <= 0xFFFFFFFF
        assert pci_info.domain == bus_id_domain

        assert isinstance(pci_info.bus, int)
        assert 0x00 <= pci_info.bus <= 0xFF
        assert pci_info.bus == bus_id_bus

        assert isinstance(pci_info.device, int)
        assert 0x00 <= pci_info.device <= 0xFF
        assert pci_info.device == bus_id_device

        assert isinstance(pci_info.vendor_id, int)
        assert 0x0000 <= pci_info.vendor_id <= 0xFFFF

        assert isinstance(pci_info.device_id, int)
        assert 0x0000 <= pci_info.device_id <= 0xFFFF


def test_device_serial():
    skip_reasons = set()
    for device in system.Device.get_all_devices():
        try:
            serial = device.serial
        except nvml.NotSupportedError:
            skip_reasons.add(f"Device serial not supported by device {device}")
        else:
            assert isinstance(serial, str)
            assert len(serial) > 0

    if skip_reasons:
        pytest.skip(" ; ".join(skip_reasons))


def test_device_uuid():
    for device in system.Device.get_all_devices():
        uuid = device.uuid
        assert isinstance(uuid, str)

        # Expands to GPU-8hex-4hex-4hex-4hex-12hex, where 8hex means 8 consecutive
        # hex characters, e.g.: "GPU-abcdef12-abcd-0123-4567-1234567890ab"
