# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from collections import namedtuple

import pytest
from cuda.bindings import _nvml as nvml


class NVMLInitializer:
    def __init__(self):
        pass

    def __enter__(self):
        nvml.init_v2()

    def __exit__(self, exception_type, exception, trace):
        nvml.shutdown()


@pytest.fixture
def nvml_init():
    with NVMLInitializer():
        yield


@pytest.fixture(scope="session", autouse=True)
def device_info():
    dev_count = None
    bus_id_to_board_details = {}

    with NVMLInitializer():
        dev_count = nvml.device_get_count_v2()

        # Store some details for each device now when we know NVML is in known state
        for i in range(dev_count):
            try:
                dev = nvml.device_get_handle_by_index_v2(i)
            except nvml.NoPermissionError:
                continue
            pci_info = nvml.device_get_pci_info_v3(dev)

            name = nvml.device_get_name(dev)
            # Get architecture name ex: Ampere, Kepler
            arch_id = nvml.device_get_architecture(dev)
            # 1 = NVML_DEVICE_ARCH_KEPLER and 12 = NVML_DEVICE_ARCH_COUNT
            assert 1 <= arch_id <= 12, "Architecture not found, presumably something newer"
            # arch_name = (utils.nvml_architecture_name.get(archID)).split("_")[-1]
            # archName = archName[0] + archName[1:].lower()

            BoardCfg = namedtuple("BoardCfg", "name, ids_arr")
            board = BoardCfg(name, ids_arr=[(pci_info.pci_device_id, pci_info.pci_sub_system_id)])

            try:
                serial = nvml.device_get_serial(dev)
            except:
                serial = None

            bus_id = pci_info.bus_id
            device_id = pci_info.device_
            uuid = nvml.device_get_uuid(dev)

            BoardDetails = namedtuple("BoardDetails", "name, board, arch_id, bus_id, device_id, serial")
            bus_id_to_board_details[uuid] = BoardDetails(name, board, arch_id, bus_id, device_id, serial)

    return bus_id_to_board_details


def get_devices(device_info):
    for uuid in list(device_info.keys()):
        try:
            yield nvml.device_get_handle_by_uuid(uuid)
        except nvml.NoPermissionError:
            continue  # ignore devices that can't be accessed


@pytest.fixture
def for_all_devices(device_info):
    with NVMLInitializer():
        unique_devices = set()
        for device_id in get_devices(device_info):
            if device_id not in unique_devices:
                unique_devices.add(device_id)
                yield device_id
                # RestoreDefaultEnvironment.restore()
