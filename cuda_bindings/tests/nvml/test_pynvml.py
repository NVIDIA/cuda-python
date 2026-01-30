# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# A set of tests ported from https://github.com/gpuopenanalytics/pynvml/blob/11.5.3/pynvml/tests/test_nvml.py

import os
import time

import pytest
from cuda.bindings import nvml

from . import util
from .conftest import unsupported_before

XFAIL_LEGACY_NVLINK_MSG = "Legacy NVLink test expected to fail."


def test_system_get_nvml_version(nvml_init):
    vsn = nvml.system_get_nvml_version()
    assert isinstance(vsn, str)
    assert tuple(int(x) for x in vsn.split(".")[:2]) > (0, 0)


def test_system_get_cuda_driver_version(nvml_init):
    vsn = nvml.system_get_cuda_driver_version()
    assert vsn != 0.0


def test_nvml_system_get_process_name(nvml_init):
    try:
        procname = nvml.system_get_process_name(os.getpid())
    except nvml.NotFoundError:
        pytest.skip("Process not found")
        return
    assert procname is not None


def test_system_get_driver_version(nvml_init):
    vsn = nvml.system_get_driver_version()
    assert isinstance(vsn, str)
    assert tuple(int(x) for x in vsn.split(".")[:2]) > (0, 0)


def test_device_get_attributes(mig_handles):
    # nvmlDeviceGetAttributes requires MIG device handle

    if mig_handles:
        for handle in mig_handles:
            att = nvml.device_get_attributes(handle)
            assert att is not None
    else:
        pytest.skip("No MIG devices found")


def test_device_get_handle_by_uuid(ngpus, uuids):
    handles = [nvml.device_get_handle_by_uuid(uuids[i]) for i in range(ngpus)]
    assert len(handles) == ngpus


def test_device_get_handle_by_pci_bus_id(ngpus, pci_info):
    handles = [nvml.device_get_handle_by_pci_bus_id_v2(pci_info[i].bus_id) for i in range(ngpus)]
    assert len(handles) == ngpus


@pytest.mark.parametrize("scope", [nvml.AffinityScope.NODE, nvml.AffinityScope.SOCKET])
@pytest.mark.skipif(util.is_wsl() or util.is_windows(), reason="Not supported on WSL or Windows")
def test_device_get_memory_affinity(handles, scope):
    size = 1024
    for handle in handles:
        with unsupported_before(handle, nvml.DeviceArch.KEPLER):
            node_set = nvml.device_get_memory_affinity(handle, size, scope)
        assert node_set is not None
        assert len(node_set) == size


@pytest.mark.parametrize("scope", [nvml.AffinityScope.NODE, nvml.AffinityScope.SOCKET])
@pytest.mark.skipif(util.is_wsl() or util.is_windows(), reason="Not supported on WSL or Windows")
def test_device_get_cpu_affinity_within_scope(handles, scope):
    size = 1024
    for handle in handles:
        with unsupported_before(handle, nvml.DeviceArch.KEPLER):
            cpu_set = nvml.device_get_cpu_affinity_within_scope(handle, size, scope)
        assert cpu_set is not None
        assert len(cpu_set) == size


@pytest.mark.parametrize(
    "index",
    [
        nvml.GpuP2PCapsIndex.P2P_CAPS_INDEX_WRITE,
        nvml.GpuP2PCapsIndex.P2P_CAPS_INDEX_NVLINK,
        nvml.GpuP2PCapsIndex.P2P_CAPS_INDEX_WRITE,
        nvml.GpuP2PCapsIndex.P2P_CAPS_INDEX_NVLINK,
        nvml.GpuP2PCapsIndex.P2P_CAPS_INDEX_ATOMICS,
        nvml.GpuP2PCapsIndex.P2P_CAPS_INDEX_PROP,
        nvml.GpuP2PCapsIndex.P2P_CAPS_INDEX_UNKNOWN,
    ],
)
def test_device_get_p2p_status(handles, index):
    for h1 in handles:
        for h2 in handles:
            if h1 is not h2:
                status = nvml.device_get_p2p_status(h1, h2, index)
                assert nvml.GpuP2PStatus.P2P_STATUS_OK <= status <= nvml.GpuP2PStatus.P2P_STATUS_UNKNOWN


# [Skipping] pynvml.nvmlDeviceGetName
# [Skipping] pynvml.nvmlDeviceGetBoardId
# [Skipping] pynvml.nvmlDeviceGetMultiGpuBoard
# [Skipping] pynvml.nvmlDeviceGetBrand
# [Skipping] pynvml.nvmlDeviceGetCpuAffinity
# [Skipping] pynvml.nvmlDeviceSetCpuAffinity
# [Skipping] pynvml.nvmlDeviceClearCpuAffinity
# [Skipping] pynvml.nvmlDeviceGetMinorNumber
# [Skipping] pynvml.nvmlDeviceGetUUID
# [Skipping] pynvml.nvmlDeviceGetInforomVersion
# [Skipping] pynvml.nvmlDeviceGetInforomImageVersion
# [Skipping] pynvml.nvmlDeviceGetInforomConfigurationChecksum
# [Skipping] pynvml.nvmlDeviceValidateInforom
# [Skipping] pynvml.nvmlDeviceGetDisplayMode
# [Skipping] pynvml.nvmlDeviceGetPersistenceMode
# [Skipping] pynvml.nvmlDeviceGetClockInfo
# [Skipping] pynvml.nvmlDeviceGetMaxClockInfo
# [Skipping] pynvml.nvmlDeviceGetApplicationsCloc
# [Skipping] pynvml.nvmlDeviceGetDefaultApplicationsClock
# [Skipping] pynvml.nvmlDeviceGetSupportedMemoryClocks
# [Skipping] pynvml.nvmlDeviceGetSupportedGraphicsClocks
# [Skipping] pynvml.nvmlDeviceGetFanSpeed
# [Skipping] pynvml.nvmlDeviceGetTemperature
# [Skipping] pynvml.nvmlDeviceGetTemperatureThreshold
# [Skipping] pynvml.nvmlDeviceGetPowerState
# [Skipping] pynvml.nvmlDeviceGetPerformanceState
# [Skipping] pynvml.nvmlDeviceGetPowerManagementMode
# [Skipping] pynvml.nvmlDeviceGetPowerManagementLimit
# [Skipping] pynvml.nvmlDeviceGetPowerManagementLimitConstraints
# [Skipping] pynvml.nvmlDeviceGetPowerManagementDefaultLimit
# [Skipping] pynvml.nvmlDeviceGetEnforcedPowerLimit


def test_device_get_power_usage(ngpus, handles):
    for i in range(ngpus):
        # Note: documentation says this is supported on Fermi or newer,
        # but in practice it fails on some later architectures.
        with unsupported_before(handles[i], None):
            power_mwatts = nvml.device_get_power_usage(handles[i])
        assert power_mwatts >= 0.0


def test_device_get_total_energy_consumption(ngpus, handles):
    for i in range(ngpus):
        with unsupported_before(handles[i], nvml.DeviceArch.VOLTA):
            energy_mjoules1 = nvml.device_get_total_energy_consumption(handles[i])

        for j in range(10):  # idle for 150 ms
            time.sleep(0.015)  # and check for increase every 15 ms
            with unsupported_before(handles[i], nvml.DeviceArch.VOLTA):
                energy_mjoules2 = nvml.device_get_total_energy_consumption(handles[i])
            assert energy_mjoules2 >= energy_mjoules1
            if energy_mjoules2 > energy_mjoules1:
                break
        else:
            raise AssertionError("energy did not increase across 150 ms interval")


# [Skipping] pynvml.nvmlDeviceGetGpuOperationMode
# [Skipping] pynvml.nvmlDeviceGetCurrentGpuOperationMode
# [Skipping] pynvml.nvmlDeviceGetPendingGpuOperationMode


def test_device_get_memory_info(ngpus, handles):
    for i in range(ngpus):
        meminfo = nvml.device_get_memory_info_v2(handles[i])
        assert (meminfo.used <= meminfo.total) and (meminfo.free <= meminfo.total)


# [Skipping] pynvml.nvmlDeviceGetBAR1MemoryInfo
# [Skipping] pynvml.nvmlDeviceGetComputeMode
# [Skipping] pynvml.nvmlDeviceGetEccMode
# [Skipping] pynvml.nvmlDeviceGetCurrentEccMode (Python API Addition)
# [Skipping] pynvml.nvmlDeviceGetPendingEccMode (Python API Addition)
# [Skipping] pynvml.nvmlDeviceGetTotalEccErrors
# [Skipping] pynvml.nvmlDeviceGetDetailedEccErrors
# [Skipping] pynvml.nvmlDeviceGetMemoryErrorCounter


def test_device_get_utilization_rates(ngpus, handles):
    for i in range(ngpus):
        with unsupported_before(handles[i], "FERMI"):
            urate = nvml.device_get_utilization_rates(handles[i])
        assert urate.gpu >= 0
        assert urate.memory >= 0


# [Skipping] pynvml.nvmlDeviceGetEncoderUtilization
# [Skipping] pynvml.nvmlDeviceGetDecoderUtilization
# [Skipping] pynvml.nvmlDeviceGetPcieReplayCounter
# [Skipping] pynvml.nvmlDeviceGetDriverModel
# [Skipping] pynvml.nvmlDeviceGetCurrentDriverModel
# [Skipping] pynvml.nvmlDeviceGetPendingDriverModel
# [Skipping] pynvml.nvmlDeviceGetVbiosVersion
# [Skipping] pynvml.nvmlDeviceGetComputeRunningProcesses
# [Skipping] pynvml.nvmlDeviceGetGraphicsRunningProcesses
# [Skipping] pynvml.nvmlDeviceGetAutoBoostedClocksEnabled
# [Skipping] nvmlUnitSetLedState
# [Skipping] pynvml.nvmlDeviceSetPersistenceMode
# [Skipping] pynvml.nvmlDeviceSetComputeMode
# [Skipping] pynvml.nvmlDeviceSetEccMode
# [Skipping] pynvml.nvmlDeviceClearEccErrorCounts
# [Skipping] pynvml.nvmlDeviceSetDriverModel
# [Skipping] pynvml.nvmlDeviceSetAutoBoostedClocksEnabled
# [Skipping] pynvml.nvmlDeviceSetDefaultAutoBoostedClocksEnabled
# [Skipping] pynvml.nvmlDeviceSetApplicationsClocks
# [Skipping] pynvml.nvmlDeviceResetApplicationsClocks
# [Skipping] pynvml.nvmlDeviceSetPowerManagementLimit
# [Skipping] pynvml.nvmlDeviceSetGpuOperationMode
# [Skipping] nvmlEventSetCreate
# [Skipping] pynvml.nvmlDeviceRegisterEvents
# [Skipping] pynvml.nvmlDeviceGetSupportedEventTypes
# [Skipping] nvmlEventSetWait
# [Skipping] nvmlEventSetFree
# [Skipping] pynvml.nvmlDeviceOnSameBoard
# [Skipping] pynvml.nvmlDeviceGetCurrPcieLinkGeneration
# [Skipping] pynvml.nvmlDeviceGetMaxPcieLinkGeneration
# [Skipping] pynvml.nvmlDeviceGetCurrPcieLinkWidth
# [Skipping] pynvml.nvmlDeviceGetMaxPcieLinkWidth
# [Skipping] pynvml.nvmlDeviceGetSupportedClocksThrottleReasons
# [Skipping] pynvml.nvmlDeviceGetCurrentClocksThrottleReasons
# [Skipping] pynvml.nvmlDeviceGetIndex
# [Skipping] pynvml.nvmlDeviceGetAccountingMode
# [Skipping] pynvml.nvmlDeviceSetAccountingMode
# [Skipping] pynvml.nvmlDeviceClearAccountingPids
# [Skipping] pynvml.nvmlDeviceGetAccountingStats
# [Skipping] pynvml.nvmlDeviceGetAccountingPids
# [Skipping] pynvml.nvmlDeviceGetAccountingBufferSize
# [Skipping] pynvml.nvmlDeviceGetRetiredPages
# [Skipping] pynvml.nvmlDeviceGetRetiredPagesPendingStatus
# [Skipping] pynvml.nvmlDeviceGetAPIRestriction
# [Skipping] pynvml.nvmlDeviceSetAPIRestriction
# [Skipping] pynvml.nvmlDeviceGetBridgeChipInfo
# [Skipping] pynvml.nvmlDeviceGetSamples
# [Skipping] pynvml.nvmlDeviceGetViolationStatus


def test_device_get_pcie_throughput(ngpus, handles):
    for i in range(ngpus):
        with unsupported_before(handles[i], nvml.DeviceArch.MAXWELL):
            tx_bytes_tp = nvml.device_get_pcie_throughput(handles[i], nvml.PcieUtilCounter.PCIE_UTIL_TX_BYTES)
        assert tx_bytes_tp >= 0
        rx_bytes_tp = nvml.device_get_pcie_throughput(handles[i], nvml.PcieUtilCounter.PCIE_UTIL_RX_BYTES)
        assert rx_bytes_tp >= 0

        # with pytest.raises(nvml.InvalidArgumentError):
        #     nvml.device_get_pcie_throughput(handles[i], nvml.PcieUtilCounter.PCIE_UTIL_COUNT)


# [Skipping] pynvml.nvmlSystemGetTopologyGpuSet
# [Skipping] pynvml.nvmlDeviceGetTopologyNearestGpus
# [Skipping] pynvml.nvmlDeviceGetTopologyCommonAncestor

# Test pynvml.nvmlDeviceGetNvLinkVersion
# Test pynvml.nvmlDeviceGetNvLinkState
# Test pynvml.nvmlDeviceGetNvLinkRemotePciInfo


@pytest.mark.parametrize(
    "cap_type",
    [
        nvml.NvLinkCapability.NVLINK_CAP_P2P_SUPPORTED,  # P2P over NVLink is supported
        nvml.NvLinkCapability.NVLINK_CAP_SYSMEM_ACCESS,  # Access to system memory is supported
        nvml.NvLinkCapability.NVLINK_CAP_P2P_ATOMICS,  # P2P atomics are supported
        nvml.NvLinkCapability.NVLINK_CAP_SYSMEM_ATOMICS,  # System memory atomics are supported
        nvml.NvLinkCapability.NVLINK_CAP_SLI_BRIDGE,  # SLI is supported over this link
        nvml.NvLinkCapability.NVLINK_CAP_VALID,
    ],
)  # Link is supported on this device
def test_device_get_nvlink_capability(ngpus, handles, cap_type):
    for i in range(ngpus):
        for j in range(nvml.NVLINK_MAX_LINKS):
            # By the documentation, this should be supported on PASCAL or newer,
            # but this also seems to fail on newer.
            with unsupported_before(handles[i], None):
                cap = nvml.device_get_nvlink_capability(handles[i], j, cap_type)
            assert cap >= 0


# Test pynvml.nvmlDeviceResetNvLinkUtilizationCounter
# Test pynvml.nvmlDeviceSetNvLinkUtilizationControl
# Test pynvml.nvmlDeviceGetNvLinkUtilizationCounter
# Test pynvml.nvmlDeviceGetNvLinkUtilizationControl
# Test pynvml.nvmlDeviceFreezeNvLinkUtilizationCounter

# Test pynvml.nvmlDeviceResetNvLinkErrorCounters
# Test pynvml.nvmlDeviceGetNvLinkErrorCounter
