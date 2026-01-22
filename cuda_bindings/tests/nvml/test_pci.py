# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE


import contextlib

from cuda.bindings import _nvml as nvml

from .conftest import unsupported_before


def test_discover_gpus(all_devices):
    for device in all_devices:
        pci_info = nvml.device_get_pci_info_v3(device)
        # Docs say this should be supported on PASCAL and later
        with unsupported_before(device, None), contextlib.suppress(nvml.OperatingSystemError):
            nvml.device_discover_gpus(pci_info.ptr)


def test_bridge_chip_hierarchy_t():
    hierarchy = nvml.BridgeChipHierarchy()
    assert len(hierarchy.bridge_chip_info) == 0
    assert not hasattr(hierarchy, "bridge_count")
    assert isinstance(hierarchy.bridge_chip_info, nvml.BridgeChipInfo)


def test_bridge_chip_info(all_devices):
    for device in all_devices:
        with unsupported_before(device, None):
            info = nvml.device_get_bridge_chip_info(device)
        assert isinstance(info, nvml.BridgeChipHierarchy)
        for entry in info.bridge_chip_info:
            assert isinstance(entry, nvml.BridgeChipInfo)
            assert isinstance(entry.type, int)
            assert isinstance(entry.fw_version, int)
