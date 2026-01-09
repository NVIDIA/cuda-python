# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402

from .conftest import skip_if_nvml_unsupported

pytestmark = skip_if_nvml_unsupported

import array
import os
import re
import sys

import helpers
import pytest
from cuda.core import system
from cuda.core.system import _device as system_device

if system.CUDA_BINDINGS_NVML_IS_COMPATIBLE:
    from cuda.bindings import _nvml as nvml


@pytest.fixture(autouse=True, scope="module")
def check_gpu_available():
    if not system.CUDA_BINDINGS_NVML_IS_COMPATIBLE or system.get_num_devices() == 0:
        pytest.skip("No GPUs available to run device tests", allow_module_level=True)


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
        except system.NotSupportedError:
            skip_reasons.add(f"CPU affinity not supported on '{device.name}'")
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
        except system.NotSupportedError:
            skip_reasons.add(f"Device serial not supported by device '{device.name}'")
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


@pytest.mark.parametrize(
    "params",
    [
        {
            "input": [1152920405096267775, 0],
            "output": [i for i in range(20)] + [i + 40 for i in range(20)],
        },
        {
            "input": [17293823668613283840, 65535],
            "output": [i + 20 for i in range(20)] + [i + 60 for i in range(20)],
        },
        {"input": [18446744073709551615, 0], "output": [i for i in range(64)]},
        {"input": [0, 18446744073709551615], "output": [i + 64 for i in range(64)]},
    ],
)
def test_unpack_bitmask(params):
    assert system_device._unpack_bitmask(array.array("Q", params["input"])) == params["output"]


def test_unpack_bitmask_single_value():
    with pytest.raises(TypeError):
        system_device._unpack_bitmask(1)


def test_device_brand():
    for device in system.Device.get_all_devices():
        brand = device.brand
        assert isinstance(brand, system_device.BrandType)
        assert isinstance(brand.name, str)
        assert isinstance(brand.value, int)


def test_device_pci_bus_id():
    for device in system.Device.get_all_devices():
        pci_bus_id = device.pci_info.bus_id
        assert isinstance(pci_bus_id, str)

        new_device = system.Device(pci_bus_id=device.pci_info.bus_id)
        assert new_device.index == device.index


@pytest.mark.skipif(helpers.IS_WSL or helpers.IS_WINDOWS, reason="Device attributes not supported on WSL or Windows")
def test_device_attributes():
    for device in system.Device.get_all_devices():
        attributes = device.attributes
        assert isinstance(attributes, system_device.DeviceAttributes)

        assert isinstance(attributes.multiprocessor_count, int)
        assert attributes.multiprocessor_count > 0

        assert isinstance(attributes.shared_copy_engine_count, int)
        assert isinstance(attributes.shared_decoder_count, int)
        assert isinstance(attributes.shared_encoder_count, int)
        assert isinstance(attributes.shared_jpeg_count, int)
        assert isinstance(attributes.shared_ofa_count, int)
        assert isinstance(attributes.gpu_instance_slice_count, int)
        assert isinstance(attributes.compute_instance_slice_count, int)
        assert isinstance(attributes.memory_size_mb, int)
        assert attributes.memory_size_mb > 0


def test_c2c_mode_enabled():
    skip_reasons = set()
    for device in system.Device.get_all_devices():
        try:
            is_enabled = device.is_c2c_mode_enabled
        except nvml.NotSupportedError:
            skip_reasons.add(f"C2C mode info not supported on {device}")
        else:
            assert isinstance(is_enabled, bool)
    if skip_reasons:
        pytest.skip(" ; ".join(skip_reasons))


@pytest.mark.skipif(helpers.IS_WSL or helpers.IS_WINDOWS, reason="Persistence mode not supported on WSL or Windows")
def test_persistence_mode_enabled():
    for device in system.Device.get_all_devices():
        is_enabled = device.persistence_mode_enabled
        assert isinstance(is_enabled, bool)
