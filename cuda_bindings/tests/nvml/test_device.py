# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE


from functools import cache

import pytest
from cuda.bindings import _nvml as nvml

from .conftest import unsupported_before


@cache
def get_cuda_version():
    nvml.init_v2()
    try:
        version = nvml.system_get_cuda_driver_version()
    finally:
        nvml.shutdown()
    return version


def cuda_version_less_than(target):
    return get_cuda_version() < target


def test_device_capabilities(all_devices):
    for device in all_devices:
        capabilities = nvml.device_get_capabilities(device)
        assert isinstance(capabilities, int)


def test_clk_mon_status_t():
    obj = nvml.ClkMonStatus()
    assert len(obj.clk_mon_list) == 0
    assert not hasattr(obj, "clk_mon_list_size")


def test_current_clock_freqs(all_devices):
    for device in all_devices:
        clk_freqs = nvml.device_get_current_clock_freqs(device)
        assert isinstance(clk_freqs, str)


def test_grid_licensable_features(all_devices):
    for device in all_devices:
        features = nvml.device_get_grid_licensable_features_v4(device)
        assert isinstance(features, nvml.GridLicensableFeatures)
        # #define NVML_GRID_LICENSE_FEATURE_MAX_COUNT 3
        assert len(features.grid_licensable_features) <= 3
        assert not hasattr(features, "licensable_features_count")

        for feature in features.grid_licensable_features:
            nvml.GridLicenseFeatureCode(feature.feature_code)
            assert isinstance(feature.feature_state, int)
            assert isinstance(feature.license_info, str)
            assert isinstance(feature.product_name, str)
            assert isinstance(feature.feature_enabled, int)
            nvml.GridLicenseExpiry(feature.license_expiry)


def test_get_handle_by_uuidv(all_devices):
    for device in all_devices:
        uuid = nvml.device_get_uuid(device)
        new_handle = nvml.device_get_handle_by_uuidv(nvml.UUIDType.ASCII, uuid.encode("ascii"))
        assert new_handle == device


def test_get_nv_link_supported_bw_modes(all_devices):
    for device in all_devices:
        with unsupported_before(device, None):
            modes = nvml.device_get_nvlink_supported_bw_modes(device)
        assert isinstance(modes, nvml.NvLinkSupportedBWModes)
        # #define NVML_NVLINK_TOTAL_SUPPORTED_BW_MODES 23
        assert len(modes.bw_modes) <= 23
        assert not hasattr(modes, "total_bw_modes")

        for mode in modes.bw_modes:
            assert isinstance(mode, int)


def test_device_get_pdi(all_devices):
    for device in all_devices:
        pdi = nvml.device_get_pdi(device)
        assert isinstance(pdi, int)


def test_device_get_performance_modes(all_devices):
    for device in all_devices:
        modes = nvml.device_get_performance_modes(device)
        assert isinstance(modes, str)


@pytest.mark.skipif(cuda_version_less_than(13010), reason="Introduced in 13.1")
def test_device_get_unrepairable_memory_flag(all_devices):
    for device in all_devices:
        status = nvml.device_get_unrepairable_memory_flag_v1(device)
        assert isinstance(status, int)


def test_device_vgpu_get_heterogeneous_mode(all_devices):
    for device in all_devices:
        with unsupported_before(device, None):
            mode = nvml.device_get_vgpu_heterogeneous_mode(device)
        assert isinstance(mode, int)


@pytest.mark.skipif(cuda_version_less_than(13010), reason="Introduced in 13.1")
def test_read_prm_counters(all_devices):
    for device in all_devices:
        counters = nvml.PRMCounter_v1(5)
        with unsupported_before(device, None):
            read_counters = nvml.device_read_prm_counters_v1(device, counters)
        assert counters is read_counters
        assert len(read_counters) == 5


def test_read_write_prm(all_devices):
    for device in all_devices:
        # Docs say supported in BLACKWELL or later
        with unsupported_before(device, None):
            try:
                result = nvml.device_read_write_prm_v1(device, b"012345678")
            except nvml.NoPermissionError:
                pytest.skip("No permission to read/write PRM")
        assert isinstance(result, tuple)
        assert isinstance(result[0], int)
        assert isinstance(result[1], bytes)


def test_nvlink_low_power_threshold(all_devices):
    for device in all_devices:
        # Docs say supported on HOPPER or newer
        with unsupported_before(device, None):
            nvml.device_set_nvlink_device_low_power_threshold(device, 0)


def test_get_power_management_limit(all_devices):
    for device in all_devices:
        # Docs say supported on KEPLER or later
        with unsupported_before(device, None):
            nvml.device_get_power_management_limit(device)


def test_set_power_management_limit(all_devices):
    for device in all_devices:
        with unsupported_before(device, nvml.DeviceArch.KEPLER):
            try:
                nvml.device_set_power_management_limit_v2(device, nvml.PowerScope.GPU, 10000)
            except nvml.NoPermissionError:
                pytest.skip("No permission to set power management limit")
            except nvml.InvalidArgumentError:
                pytest.skip("Invalid argument when setting power management limit -- probably unsupported")


def test_set_temperature_threshold(all_devices):
    for device in all_devices:
        # Docs say supported on MAXWELL or newer
        with unsupported_before(device, None):
            temp = nvml.device_get_temperature_threshold(
                device, nvml.TemperatureThresholds.TEMPERATURE_THRESHOLD_ACOUSTIC_CURR
            )
        try:
            nvml.device_set_temperature_threshold(
                device, nvml.TemperatureThresholds.TEMPERATURE_THRESHOLD_ACOUSTIC_CURR, temp
            )
        except nvml.NoPermissionError:
            pytest.skip("No permission to set temperature threshold")
        except nvml.InvalidArgumentError:
            pytest.skip("Invalid argument when setting temperature threshold -- this is probably the temp type")
