# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from functools import cache

import numpy as np
import pytest

from cuda.bindings import nvml

from . import util
from .conftest import unsupported_before

# A timestamp far in the future, so that "samples newer than this timestamp"
# is reliably empty without requiring any particular hardware state.
_FUTURE_TIMESTAMP = 2**63 - 1


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
        with unsupported_before(device, None):
            capabilities = nvml.device_get_capabilities(device)
            assert isinstance(capabilities, int)


def test_clk_mon_status_t():
    obj = nvml.ClkMonStatus()
    assert len(obj.clk_mon_list) == 0
    assert not hasattr(obj, "clk_mon_list_size")


def test_current_clock_freqs(all_devices, subtests):
    for device in all_devices:
        with subtests.test(device_index=nvml.device_get_index(device)):
            with unsupported_before(device, None):
                clk_freqs = nvml.device_get_current_clock_freqs(device)
            assert isinstance(clk_freqs, str)


def test_grid_licensable_features(all_devices):
    for device in all_devices:
        with unsupported_before(device, None):
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


def test_get_handle_by_uuidv(all_devices, subtests):
    for device in all_devices:
        with subtests.test(device_index=nvml.device_get_index(device)):
            uuid = nvml.device_get_uuid(device)
            if "Orin" in nvml.device_get_name(device) and len(uuid) == 36:
                pytest.skip("UUID lookup is unsupported on Orin, which reports a UUID without a GPU- prefix")
            with unsupported_before(device, None):
                new_handle = nvml.device_get_handle_by_uuidv(nvml.UUIDType.ASCII, uuid.encode("ascii"))
            assert new_handle == device


def test_get_nv_link_supported_bw_modes(all_devices, subtests):
    for device in all_devices:
        with subtests.test(device_index=nvml.device_get_index(device)):
            with unsupported_before(device, None):
                modes = nvml.device_get_nvlink_supported_bw_modes(device)
            assert isinstance(modes, nvml.NvlinkSupportedBwModes_v1)
            # #define NVML_NVLINK_TOTAL_SUPPORTED_BW_MODES 23
            assert len(modes.bw_modes) <= 23
            assert not hasattr(modes, "total_bw_modes")

            for mode in modes.bw_modes:
                assert isinstance(mode, np.uint8)


def test_device_get_pdi(all_devices):
    for device in all_devices:
        with unsupported_before(device, None):
            pdi = nvml.device_get_pdi(device)
            assert isinstance(pdi, int)


def test_device_get_performance_modes(all_devices, subtests):
    for device in all_devices:
        with subtests.test(device_index=nvml.device_get_index(device)):
            with unsupported_before(device, None):
                modes = nvml.device_get_performance_modes(device)
            assert isinstance(modes, str)


@pytest.mark.skipif(cuda_version_less_than(13010), reason="Introduced in 13.1")
def test_device_get_unrepairable_memory_flag(all_devices, subtests):
    for device in all_devices:
        with subtests.test(device_index=nvml.device_get_index(device)):
            with unsupported_before(device, None):
                status = nvml.device_get_unrepairable_memory_flag_v1(device)
            assert isinstance(status, int)


def test_device_vgpu_get_heterogeneous_mode(all_devices, subtests):
    for device in all_devices:
        with subtests.test(device_index=nvml.device_get_index(device)):
            with unsupported_before(device, None):
                mode = nvml.device_get_vgpu_heterogeneous_mode(device)
            assert isinstance(mode, int)


@pytest.mark.skipif(cuda_version_less_than(13010), reason="Introduced in 13.1")
def test_read_prm_counters(all_devices, subtests):
    for device in all_devices:
        with subtests.test(device_index=nvml.device_get_index(device)):
            counters = nvml.PRMCounter_v1(5)
            with unsupported_before(device, None):
                read_counters = nvml.device_read_prm_counters_v1(device, counters)
            assert counters is read_counters
            assert len(read_counters) == 5


@pytest.mark.thread_unsafe(reason="API appears to be thread-unsafe (2026-06)")
def test_read_write_prm(all_devices, subtests):
    for device in all_devices:
        with subtests.test(device_index=nvml.device_get_index(device)):
            # Docs say supported in BLACKWELL or later
            with unsupported_before(device, None):
                try:
                    result = nvml.device_read_write_prm_v1(device, b"012345678")
                except nvml.NoPermissionError:
                    pytest.skip("No permission to read/write PRM")
            assert isinstance(result, tuple)
            assert isinstance(result[0], int)
            assert isinstance(result[1], bytes)


def test_get_power_management_limit(all_devices, subtests):
    for device in all_devices:
        # Docs say supported on KEPLER or later
        with subtests.test(device_index=nvml.device_get_index(device)), unsupported_before(device, None):
            nvml.device_get_power_management_limit(device)


def test_set_power_management_limit(all_devices, subtests):
    for device in all_devices:
        with (
            subtests.test(device_index=nvml.device_get_index(device)),
            unsupported_before(device, None),
        ):
            try:
                nvml.device_set_power_management_limit_v2(device, nvml.PowerScope.GPU, 10000)
            except nvml.NoPermissionError:
                pytest.skip("No permission to set power management limit")
            except nvml.InvalidArgumentError:
                pytest.skip("Invalid argument when setting power management limit -- probably unsupported")


def test_set_temperature_threshold(all_devices, subtests):
    for device in all_devices:
        with subtests.test(device_index=nvml.device_get_index(device)):
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


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_device_get_samples_zero_result_returns_tuple(all_devices, subtests):
    """device_get_samples must always return a (sample_val_type, samples) 2-tuple,
    even when there are zero samples to report, instead of a bare Sample instance.
    """
    for device in all_devices:
        with subtests.test(device_index=nvml.device_get_index(device)):
            with unsupported_before(device, None):
                # Query once to learn the timestamp of the most recent sample, then
                # re-query for samples strictly newer than that: on most drivers this
                # is a legitimate zero-sample SUCCESS; some drivers instead report
                # NotFoundError when there is nothing newer, which is also an
                # acceptable (non-crashing) outcome.
                _, latest_samples = nvml.device_get_samples(device, nvml.SamplingType.GPU_UTILIZATION_SAMPLES, 0)
                last_seen_time_stamp = max((s.time_stamp for s in latest_samples), default=_FUTURE_TIMESTAMP)
                try:
                    result = nvml.device_get_samples(
                        device, nvml.SamplingType.GPU_UTILIZATION_SAMPLES, last_seen_time_stamp
                    )
                except nvml.NotFoundError:
                    continue

            assert isinstance(result, tuple)
            assert len(result) == 2
            sample_val_type, samples = result
            assert isinstance(sample_val_type, int)
            assert len(samples) == 0


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_device_get_supported_vgpus_no_namerror(all_devices, subtests):
    """device_get_supported_vgpus must not raise a NameError, whether or not
    the device has any supported vGPU types.
    """
    for device in all_devices:
        with subtests.test(device_index=nvml.device_get_index(device)):
            with unsupported_before(device, None):
                vgpu_type_ids = nvml.device_get_supported_vgpus(device)
            assert hasattr(vgpu_type_ids, "__len__")


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_device_get_creatable_vgpus_no_namerror(all_devices, subtests):
    """device_get_creatable_vgpus must not raise a NameError, whether or not
    the device has any creatable vGPU types.
    """
    for device in all_devices:
        with subtests.test(device_index=nvml.device_get_index(device)):
            with unsupported_before(device, None):
                vgpu_type_ids = nvml.device_get_creatable_vgpus(device)
            assert hasattr(vgpu_type_ids, "__len__")


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_device_get_active_vgpus_no_namerror(all_devices, subtests):
    """device_get_active_vgpus must not raise a NameError, whether or not
    the device currently has any active vGPU instances.
    """
    for device in all_devices:
        with subtests.test(device_index=nvml.device_get_index(device)):
            with unsupported_before(device, None):
                active_vgpus = nvml.device_get_active_vgpus(device)
            assert hasattr(active_vgpus, "__len__")


def _iter_gpu_instance_profile_ids(device):
    for profile in nvml.GpuInstanceProfile:
        if profile == nvml.GpuInstanceProfile.PROFILE_COUNT:
            continue
        try:
            info = nvml.device_get_gpu_instance_profile_info_v(device, int(profile))
        except (nvml.NotSupportedError, nvml.InvalidArgumentError):
            continue
        yield info.id


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_device_get_gpu_instances_empty_result(all_devices, subtests):
    """device_get_gpu_instances must not raise ValueError: Invalid shape in
    axis 0: 0 when a profile has zero existing GPU instances.
    """
    for device in all_devices:
        with subtests.test(device_index=nvml.device_get_index(device)):
            if util.is_vgpu(device):
                pytest.skip(f"Not supported on vGPU device {device}")

            profile_ids = list(_iter_gpu_instance_profile_ids(device))
            if not profile_ids:
                pytest.skip("No GPU instance profiles supported on this device")

            for profile_id in profile_ids:
                gpu_instances = nvml.device_get_gpu_instances(device, profile_id)
                assert hasattr(gpu_instances, "__len__")


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_gpu_instance_get_compute_instances_empty_result(all_devices, subtests):
    """gpu_instance_get_compute_instances must not raise ValueError: Invalid
    shape in axis 0: 0 when a profile has zero existing compute instances.
    """
    for device in all_devices:
        with subtests.test(device_index=nvml.device_get_index(device)):
            if util.is_vgpu(device):
                pytest.skip(f"Not supported on vGPU device {device}")

            gpu_instance = None
            for profile_id in _iter_gpu_instance_profile_ids(device):
                gpu_instances = nvml.device_get_gpu_instances(device, profile_id)
                if len(gpu_instances) > 0:
                    gpu_instance = gpu_instances[0]
                    break

            if gpu_instance is None:
                pytest.skip("No existing GPU instances on this device to query compute instances for")

            for compute_profile in nvml.ComputeInstanceProfile:
                if compute_profile == nvml.ComputeInstanceProfile.PROFILE_COUNT:
                    continue
                try:
                    compute_info = nvml.gpu_instance_get_compute_instance_profile_info_v(
                        gpu_instance, int(compute_profile), 0
                    )
                except (nvml.NotSupportedError, nvml.InvalidArgumentError):
                    continue
                compute_instances = nvml.gpu_instance_get_compute_instances(gpu_instance, compute_info.id)
                assert hasattr(compute_instances, "__len__")


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_device_get_vgpu_utilization_sized_array(all_devices, subtests):
    """device_get_vgpu_utilization must return an array sized to the real
    sample count for every sample, not just fill in the first element and
    leave the rest as uninitialized garbage.
    """
    for device in all_devices:
        with subtests.test(device_index=nvml.device_get_index(device)):
            try:
                sample_val_type, samples = nvml.device_get_vgpu_utilization(device, _FUTURE_TIMESTAMP)
            except nvml.NotSupportedError:
                pytest.skip(f"vGPU utilization not supported on device {device}")

            assert isinstance(sample_val_type, int)
            # A future timestamp means no samples should be newer than it.
            assert len(samples) == 0


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_device_get_vgpu_process_utilization_returns_array(all_devices, subtests):
    """device_get_vgpu_process_utilization must return a correctly-sized array
    directly (not a (sample_count, samples) tuple with a stale one-element
    samples array).
    """
    for device in all_devices:
        with subtests.test(device_index=nvml.device_get_index(device)):
            try:
                samples = nvml.device_get_vgpu_process_utilization(device, _FUTURE_TIMESTAMP)
            except nvml.NotSupportedError:
                pytest.skip(f"vGPU process utilization not supported on device {device}")

            assert not isinstance(samples, tuple)
            # A future timestamp means no samples should be newer than it.
            assert len(samples) == 0
