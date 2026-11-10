# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.9.1 to 13.5.0. Do not modify it directly.
# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=5f17ea23ec5cd82621614d664bfddb8c7cb48a59dedab4f94d090f5ed88fc43b


# <<<< PREAMBLE CONTENT >>>>

from libc.stdint cimport intptr_t

cdef extern from * nogil:
    """
    #if defined(_MSC_VER) && !defined(__clang__)
        #include <intrin.h>
        static __forceinline int atomic_int_load(int *p) {
            int v = *(int volatile *)p; _ReadBarrier(); return v;
        }
        static __forceinline void atomic_int_store(int *p, int v) {
            _WriteBarrier(); *(int volatile *)p = v;
        }
    #elif defined(__cplusplus)
        /* GCC/Clang __atomic builtins work in any C++ standard without headers */
        static inline int atomic_int_load(int *p) {
            return __atomic_load_n(p, __ATOMIC_ACQUIRE);
        }
        static inline void atomic_int_store(int *p, int v) {
            __atomic_store_n(p, v, __ATOMIC_RELEASE);
        }
    #else
        #include <stdatomic.h>
        static inline int atomic_int_load(int *p) {
            return (int)atomic_load_explicit((atomic_int *)p, memory_order_acquire);
        }
        static inline void atomic_int_store(int *p, int v) {
            atomic_store_explicit((atomic_int *)p, v, memory_order_release);
        }
    #endif

    """
    cdef int _cyb_atomic_int_load "atomic_int_load"(int *p) nogil
    cdef void _cyb_atomic_int_store "atomic_int_store"(int *p, int v) nogil

cdef extern from "<dlfcn.h>":
    void* _cyb_dlsym "dlsym"(void*, const char*) nogil
    const void * _cyb_RTLD_DEFAULT "RTLD_DEFAULT"

import threading as _cyb_threading

cdef int _cyb___py_nvml_init = 0
cdef dict _cyb_func_ptrs = None
cdef object _cyb_symbol_lock = _cyb_threading.Lock()

# <<<< END OF PREAMBLE CONTENT >>>>

from libc.stdint cimport uintptr_t

from .utils import FunctionNotFoundError, NotSupportedError
from cuda.pathfinder import load_nvidia_dynamic_lib


###############################################################################
# Wrapper init
###############################################################################

cdef void* __nvmlInit_v2 = NULL
cdef void* __nvmlInitWithFlags = NULL
cdef void* __nvmlShutdown = NULL
cdef void* __nvmlErrorString = NULL
cdef void* __nvmlSystemGetDriverVersion = NULL
cdef void* __nvmlSystemGetNVMLVersion = NULL
cdef void* __nvmlSystemGetCudaDriverVersion = NULL
cdef void* __nvmlSystemGetCudaDriverVersion_v2 = NULL
cdef void* __nvmlSystemGetProcessName = NULL
cdef void* __nvmlSystemGetHicVersion = NULL
cdef void* __nvmlSystemGetTopologyGpuSet = NULL
cdef void* __nvmlSystemGetDriverBranch = NULL
cdef void* __nvmlUnitGetCount = NULL
cdef void* __nvmlUnitGetHandleByIndex = NULL
cdef void* __nvmlUnitGetUnitInfo = NULL
cdef void* __nvmlUnitGetLedState = NULL
cdef void* __nvmlUnitGetPsuInfo = NULL
cdef void* __nvmlUnitGetTemperature = NULL
cdef void* __nvmlUnitGetFanSpeedInfo = NULL
cdef void* __nvmlUnitGetDevices = NULL
cdef void* __nvmlDeviceGetCount_v2 = NULL
cdef void* __nvmlDeviceGetAttributes_v2 = NULL
cdef void* __nvmlDeviceGetHandleByIndex_v2 = NULL
cdef void* __nvmlDeviceGetHandleBySerial = NULL
cdef void* __nvmlDeviceGetHandleByUUID = NULL
cdef void* __nvmlDeviceGetHandleByUUIDV = NULL
cdef void* __nvmlDeviceGetHandleByPciBusId_v2 = NULL
cdef void* __nvmlDeviceGetName = NULL
cdef void* __nvmlDeviceGetBrand = NULL
cdef void* __nvmlDeviceGetIndex = NULL
cdef void* __nvmlDeviceGetSerial = NULL
cdef void* __nvmlDeviceGetModuleId = NULL
cdef void* __nvmlDeviceGetC2cModeInfoV = NULL
cdef void* __nvmlDeviceGetMemoryAffinity = NULL
cdef void* __nvmlDeviceGetCpuAffinityWithinScope = NULL
cdef void* __nvmlDeviceGetCpuAffinity = NULL
cdef void* __nvmlDeviceSetCpuAffinity = NULL
cdef void* __nvmlDeviceClearCpuAffinity = NULL
cdef void* __nvmlDeviceGetNumaNodeId = NULL
cdef void* __nvmlDeviceGetTopologyCommonAncestor = NULL
cdef void* __nvmlDeviceGetTopologyNearestGpus = NULL
cdef void* __nvmlDeviceGetP2PStatus = NULL
cdef void* __nvmlDeviceGetUUID = NULL
cdef void* __nvmlDeviceGetMinorNumber = NULL
cdef void* __nvmlDeviceGetBoardPartNumber = NULL
cdef void* __nvmlDeviceGetInforomVersion = NULL
cdef void* __nvmlDeviceGetInforomImageVersion = NULL
cdef void* __nvmlDeviceGetInforomConfigurationChecksum = NULL
cdef void* __nvmlDeviceValidateInforom = NULL
cdef void* __nvmlDeviceGetLastBBXFlushTime = NULL
cdef void* __nvmlDeviceGetDisplayMode = NULL
cdef void* __nvmlDeviceGetDisplayActive = NULL
cdef void* __nvmlDeviceGetPersistenceMode = NULL
cdef void* __nvmlDeviceGetPciInfoExt = NULL
cdef void* __nvmlDeviceGetPciInfo_v3 = NULL
cdef void* __nvmlDeviceGetMaxPcieLinkGeneration = NULL
cdef void* __nvmlDeviceGetGpuMaxPcieLinkGeneration = NULL
cdef void* __nvmlDeviceGetMaxPcieLinkWidth = NULL
cdef void* __nvmlDeviceGetCurrPcieLinkGeneration = NULL
cdef void* __nvmlDeviceGetCurrPcieLinkWidth = NULL
cdef void* __nvmlDeviceGetPcieThroughput = NULL
cdef void* __nvmlDeviceGetPcieReplayCounter = NULL
cdef void* __nvmlDeviceGetClockInfo = NULL
cdef void* __nvmlDeviceGetMaxClockInfo = NULL
cdef void* __nvmlDeviceGetGpcClkVfOffset = NULL
cdef void* __nvmlDeviceGetClock = NULL
cdef void* __nvmlDeviceGetMaxCustomerBoostClock = NULL
cdef void* __nvmlDeviceGetSupportedMemoryClocks = NULL
cdef void* __nvmlDeviceGetSupportedGraphicsClocks = NULL
cdef void* __nvmlDeviceGetAutoBoostedClocksEnabled = NULL
cdef void* __nvmlDeviceGetFanSpeed = NULL
cdef void* __nvmlDeviceGetFanSpeed_v2 = NULL
cdef void* __nvmlDeviceGetFanSpeedRPM = NULL
cdef void* __nvmlDeviceGetTargetFanSpeed = NULL
cdef void* __nvmlDeviceGetMinMaxFanSpeed = NULL
cdef void* __nvmlDeviceGetFanControlPolicy_v2 = NULL
cdef void* __nvmlDeviceGetNumFans = NULL
cdef void* __nvmlDeviceGetCoolerInfo = NULL
cdef void* __nvmlDeviceGetTemperatureV = NULL
cdef void* __nvmlDeviceGetTemperatureThreshold = NULL
cdef void* __nvmlDeviceGetMarginTemperature = NULL
cdef void* __nvmlDeviceGetThermalSettings = NULL
cdef void* __nvmlDeviceGetPerformanceState = NULL
cdef void* __nvmlDeviceGetCurrentClocksEventReasons = NULL
cdef void* __nvmlDeviceGetSupportedClocksEventReasons = NULL
cdef void* __nvmlDeviceGetPowerState = NULL
cdef void* __nvmlDeviceGetDynamicPstatesInfo = NULL
cdef void* __nvmlDeviceGetMemClkVfOffset = NULL
cdef void* __nvmlDeviceGetMinMaxClockOfPState = NULL
cdef void* __nvmlDeviceGetSupportedPerformanceStates = NULL
cdef void* __nvmlDeviceGetGpcClkMinMaxVfOffset = NULL
cdef void* __nvmlDeviceGetMemClkMinMaxVfOffset = NULL
cdef void* __nvmlDeviceGetClockOffsets = NULL
cdef void* __nvmlDeviceSetClockOffsets = NULL
cdef void* __nvmlDeviceGetPerformanceModes = NULL
cdef void* __nvmlDeviceGetCurrentClockFreqs = NULL
cdef void* __nvmlDeviceGetPowerManagementLimit = NULL
cdef void* __nvmlDeviceGetPowerManagementLimitConstraints = NULL
cdef void* __nvmlDeviceGetPowerManagementDefaultLimit = NULL
cdef void* __nvmlDeviceGetPowerUsage = NULL
cdef void* __nvmlDeviceGetTotalEnergyConsumption = NULL
cdef void* __nvmlDeviceGetEnforcedPowerLimit = NULL
cdef void* __nvmlDeviceGetGpuOperationMode = NULL
cdef void* __nvmlDeviceGetMemoryInfo_v2 = NULL
cdef void* __nvmlDeviceGetComputeMode = NULL
cdef void* __nvmlDeviceGetCudaComputeCapability = NULL
cdef void* __nvmlDeviceGetDramEncryptionMode = NULL
cdef void* __nvmlDeviceSetDramEncryptionMode = NULL
cdef void* __nvmlDeviceGetEccMode = NULL
cdef void* __nvmlDeviceGetDefaultEccMode = NULL
cdef void* __nvmlDeviceGetBoardId = NULL
cdef void* __nvmlDeviceGetMultiGpuBoard = NULL
cdef void* __nvmlDeviceGetTotalEccErrors = NULL
cdef void* __nvmlDeviceGetMemoryErrorCounter = NULL
cdef void* __nvmlDeviceGetUtilizationRates = NULL
cdef void* __nvmlDeviceGetEncoderUtilization = NULL
cdef void* __nvmlDeviceGetEncoderCapacity = NULL
cdef void* __nvmlDeviceGetEncoderStats = NULL
cdef void* __nvmlDeviceGetEncoderSessions = NULL
cdef void* __nvmlDeviceGetDecoderUtilization = NULL
cdef void* __nvmlDeviceGetJpgUtilization = NULL
cdef void* __nvmlDeviceGetOfaUtilization = NULL
cdef void* __nvmlDeviceGetFBCStats = NULL
cdef void* __nvmlDeviceGetFBCSessions = NULL
cdef void* __nvmlDeviceGetDriverModel_v2 = NULL
cdef void* __nvmlDeviceGetVbiosVersion = NULL
cdef void* __nvmlDeviceGetBridgeChipInfo = NULL
cdef void* __nvmlDeviceGetComputeRunningProcesses_v3 = NULL
cdef void* __nvmlDeviceGetGraphicsRunningProcesses_v3 = NULL
cdef void* __nvmlDeviceGetMPSComputeRunningProcesses_v3 = NULL
cdef void* __nvmlDeviceGetRunningProcessDetailList = NULL
cdef void* __nvmlDeviceOnSameBoard = NULL
cdef void* __nvmlDeviceGetAPIRestriction = NULL
cdef void* __nvmlDeviceGetSamples = NULL
cdef void* __nvmlDeviceGetBAR1MemoryInfo = NULL
cdef void* __nvmlDeviceGetIrqNum = NULL
cdef void* __nvmlDeviceGetNumGpuCores = NULL
cdef void* __nvmlDeviceGetPowerSource = NULL
cdef void* __nvmlDeviceGetMemoryBusWidth = NULL
cdef void* __nvmlDeviceGetPcieLinkMaxSpeed = NULL
cdef void* __nvmlDeviceGetPcieSpeed = NULL
cdef void* __nvmlDeviceGetAdaptiveClockInfoStatus = NULL
cdef void* __nvmlDeviceGetBusType = NULL
cdef void* __nvmlDeviceGetGpuFabricInfoV = NULL
cdef void* __nvmlSystemGetConfComputeCapabilities = NULL
cdef void* __nvmlSystemGetConfComputeState = NULL
cdef void* __nvmlDeviceGetConfComputeMemSizeInfo = NULL
cdef void* __nvmlSystemGetConfComputeGpusReadyState = NULL
cdef void* __nvmlDeviceGetConfComputeProtectedMemoryUsage = NULL
cdef void* __nvmlDeviceGetConfComputeGpuCertificate = NULL
cdef void* __nvmlDeviceGetConfComputeGpuAttestationReport = NULL
cdef void* __nvmlSystemGetConfComputeKeyRotationThresholdInfo = NULL
cdef void* __nvmlDeviceSetConfComputeUnprotectedMemSize = NULL
cdef void* __nvmlSystemSetConfComputeGpusReadyState = NULL
cdef void* __nvmlSystemSetConfComputeKeyRotationThresholdInfo = NULL
cdef void* __nvmlSystemGetConfComputeSettings = NULL
cdef void* __nvmlDeviceGetGspFirmwareVersion = NULL
cdef void* __nvmlDeviceGetGspFirmwareMode = NULL
cdef void* __nvmlDeviceGetSramEccErrorStatus = NULL
cdef void* __nvmlDeviceGetAccountingMode = NULL
cdef void* __nvmlDeviceGetAccountingStats = NULL
cdef void* __nvmlDeviceGetAccountingPids = NULL
cdef void* __nvmlDeviceGetAccountingBufferSize = NULL
cdef void* __nvmlDeviceGetRetiredPages = NULL
cdef void* __nvmlDeviceGetRetiredPages_v2 = NULL
cdef void* __nvmlDeviceGetRetiredPagesPendingStatus = NULL
cdef void* __nvmlDeviceGetRemappedRows = NULL
cdef void* __nvmlDeviceGetRowRemapperHistogram = NULL
cdef void* __nvmlDeviceGetArchitecture = NULL
cdef void* __nvmlDeviceGetClkMonStatus = NULL
cdef void* __nvmlDeviceGetProcessUtilization = NULL
cdef void* __nvmlDeviceGetProcessesUtilizationInfo = NULL
cdef void* __nvmlDeviceGetPlatformInfo = NULL
cdef void* __nvmlUnitSetLedState = NULL
cdef void* __nvmlDeviceSetPersistenceMode = NULL
cdef void* __nvmlDeviceSetComputeMode = NULL
cdef void* __nvmlDeviceSetEccMode = NULL
cdef void* __nvmlDeviceClearEccErrorCounts = NULL
cdef void* __nvmlDeviceSetDriverModel = NULL
cdef void* __nvmlDeviceSetGpuLockedClocks = NULL
cdef void* __nvmlDeviceResetGpuLockedClocks = NULL
cdef void* __nvmlDeviceSetMemoryLockedClocks = NULL
cdef void* __nvmlDeviceResetMemoryLockedClocks = NULL
cdef void* __nvmlDeviceSetAutoBoostedClocksEnabled = NULL
cdef void* __nvmlDeviceSetDefaultAutoBoostedClocksEnabled = NULL
cdef void* __nvmlDeviceSetDefaultFanSpeed_v2 = NULL
cdef void* __nvmlDeviceSetFanControlPolicy = NULL
cdef void* __nvmlDeviceSetTemperatureThreshold = NULL
cdef void* __nvmlDeviceSetGpuOperationMode = NULL
cdef void* __nvmlDeviceSetAPIRestriction = NULL
cdef void* __nvmlDeviceSetFanSpeed_v2 = NULL
cdef void* __nvmlDeviceSetAccountingMode = NULL
cdef void* __nvmlDeviceClearAccountingPids = NULL
cdef void* __nvmlDeviceSetPowerManagementLimit_v2 = NULL
cdef void* __nvmlDeviceGetNvLinkState = NULL
cdef void* __nvmlDeviceGetNvLinkVersion = NULL
cdef void* __nvmlDeviceGetNvLinkCapability = NULL
cdef void* __nvmlDeviceGetNvLinkRemotePciInfo_v2 = NULL
cdef void* __nvmlDeviceGetNvLinkErrorCounter = NULL
cdef void* __nvmlDeviceResetNvLinkErrorCounters = NULL
cdef void* __nvmlDeviceGetNvLinkRemoteDeviceType = NULL
cdef void* __nvmlDeviceSetNvLinkDeviceLowPowerThreshold = NULL
cdef void* __nvmlSystemSetNvlinkBwMode = NULL
cdef void* __nvmlSystemGetNvlinkBwMode = NULL
cdef void* __nvmlDeviceGetNvlinkSupportedBwModes = NULL
cdef void* __nvmlDeviceGetNvlinkBwMode = NULL
cdef void* __nvmlDeviceSetNvlinkBwMode = NULL
cdef void* __nvmlEventSetCreate = NULL
cdef void* __nvmlDeviceRegisterEvents = NULL
cdef void* __nvmlDeviceGetSupportedEventTypes = NULL
cdef void* __nvmlEventSetWait_v2 = NULL
cdef void* __nvmlEventSetFree = NULL
cdef void* __nvmlSystemEventSetCreate = NULL
cdef void* __nvmlSystemEventSetFree = NULL
cdef void* __nvmlSystemRegisterEvents = NULL
cdef void* __nvmlSystemEventSetWait = NULL
cdef void* __nvmlDeviceModifyDrainState = NULL
cdef void* __nvmlDeviceQueryDrainState = NULL
cdef void* __nvmlDeviceRemoveGpu_v2 = NULL
cdef void* __nvmlDeviceDiscoverGpus = NULL
cdef void* __nvmlDeviceGetFieldValues = NULL
cdef void* __nvmlDeviceClearFieldValues = NULL
cdef void* __nvmlDeviceGetVirtualizationMode = NULL
cdef void* __nvmlDeviceGetHostVgpuMode = NULL
cdef void* __nvmlDeviceSetVirtualizationMode = NULL
cdef void* __nvmlDeviceGetVgpuHeterogeneousMode = NULL
cdef void* __nvmlDeviceSetVgpuHeterogeneousMode = NULL
cdef void* __nvmlVgpuInstanceGetPlacementId = NULL
cdef void* __nvmlDeviceGetVgpuTypeSupportedPlacements = NULL
cdef void* __nvmlDeviceGetVgpuTypeCreatablePlacements = NULL
cdef void* __nvmlVgpuTypeGetGspHeapSize = NULL
cdef void* __nvmlVgpuTypeGetFbReservation = NULL
cdef void* __nvmlVgpuInstanceGetRuntimeStateSize = NULL
cdef void* __nvmlDeviceSetVgpuCapabilities = NULL
cdef void* __nvmlDeviceGetGridLicensableFeatures_v4 = NULL
cdef void* __nvmlGetVgpuDriverCapabilities = NULL
cdef void* __nvmlDeviceGetVgpuCapabilities = NULL
cdef void* __nvmlDeviceGetSupportedVgpus = NULL
cdef void* __nvmlDeviceGetCreatableVgpus = NULL
cdef void* __nvmlVgpuTypeGetClass = NULL
cdef void* __nvmlVgpuTypeGetName = NULL
cdef void* __nvmlVgpuTypeGetGpuInstanceProfileId = NULL
cdef void* __nvmlVgpuTypeGetDeviceID = NULL
cdef void* __nvmlVgpuTypeGetFramebufferSize = NULL
cdef void* __nvmlVgpuTypeGetNumDisplayHeads = NULL
cdef void* __nvmlVgpuTypeGetResolution = NULL
cdef void* __nvmlVgpuTypeGetLicense = NULL
cdef void* __nvmlVgpuTypeGetFrameRateLimit = NULL
cdef void* __nvmlVgpuTypeGetMaxInstances = NULL
cdef void* __nvmlVgpuTypeGetMaxInstancesPerVm = NULL
cdef void* __nvmlVgpuTypeGetBAR1Info = NULL
cdef void* __nvmlDeviceGetActiveVgpus = NULL
cdef void* __nvmlVgpuInstanceGetVmID = NULL
cdef void* __nvmlVgpuInstanceGetUUID = NULL
cdef void* __nvmlVgpuInstanceGetVmDriverVersion = NULL
cdef void* __nvmlVgpuInstanceGetFbUsage = NULL
cdef void* __nvmlVgpuInstanceGetLicenseStatus = NULL
cdef void* __nvmlVgpuInstanceGetType = NULL
cdef void* __nvmlVgpuInstanceGetFrameRateLimit = NULL
cdef void* __nvmlVgpuInstanceGetEccMode = NULL
cdef void* __nvmlVgpuInstanceGetEncoderCapacity = NULL
cdef void* __nvmlVgpuInstanceSetEncoderCapacity = NULL
cdef void* __nvmlVgpuInstanceGetEncoderStats = NULL
cdef void* __nvmlVgpuInstanceGetEncoderSessions = NULL
cdef void* __nvmlVgpuInstanceGetFBCStats = NULL
cdef void* __nvmlVgpuInstanceGetFBCSessions = NULL
cdef void* __nvmlVgpuInstanceGetGpuInstanceId = NULL
cdef void* __nvmlVgpuInstanceGetGpuPciId = NULL
cdef void* __nvmlVgpuTypeGetCapabilities = NULL
cdef void* __nvmlVgpuInstanceGetMdevUUID = NULL
cdef void* __nvmlGpuInstanceGetCreatableVgpus = NULL
cdef void* __nvmlVgpuTypeGetMaxInstancesPerGpuInstance = NULL
cdef void* __nvmlGpuInstanceGetActiveVgpus = NULL
cdef void* __nvmlGpuInstanceSetVgpuSchedulerState = NULL
cdef void* __nvmlGpuInstanceGetVgpuSchedulerState = NULL
cdef void* __nvmlGpuInstanceGetVgpuSchedulerLog = NULL
cdef void* __nvmlGpuInstanceGetVgpuTypeCreatablePlacements = NULL
cdef void* __nvmlGpuInstanceGetVgpuHeterogeneousMode = NULL
cdef void* __nvmlGpuInstanceSetVgpuHeterogeneousMode = NULL
cdef void* __nvmlVgpuInstanceGetMetadata = NULL
cdef void* __nvmlDeviceGetVgpuMetadata = NULL
cdef void* __nvmlGetVgpuCompatibility = NULL
cdef void* __nvmlDeviceGetPgpuMetadataString = NULL
cdef void* __nvmlDeviceGetVgpuSchedulerLog = NULL
cdef void* __nvmlDeviceGetVgpuSchedulerState = NULL
cdef void* __nvmlDeviceGetVgpuSchedulerCapabilities = NULL
cdef void* __nvmlDeviceSetVgpuSchedulerState = NULL
cdef void* __nvmlGetVgpuVersion = NULL
cdef void* __nvmlSetVgpuVersion = NULL
cdef void* __nvmlDeviceGetVgpuUtilization = NULL
cdef void* __nvmlDeviceGetVgpuInstancesUtilizationInfo = NULL
cdef void* __nvmlDeviceGetVgpuProcessUtilization = NULL
cdef void* __nvmlDeviceGetVgpuProcessesUtilizationInfo = NULL
cdef void* __nvmlVgpuInstanceGetAccountingMode = NULL
cdef void* __nvmlVgpuInstanceGetAccountingPids = NULL
cdef void* __nvmlVgpuInstanceGetAccountingStats = NULL
cdef void* __nvmlVgpuInstanceClearAccountingPids = NULL
cdef void* __nvmlVgpuInstanceGetLicenseInfo_v2 = NULL
cdef void* __nvmlGetExcludedDeviceCount = NULL
cdef void* __nvmlGetExcludedDeviceInfoByIndex = NULL
cdef void* __nvmlDeviceSetMigMode = NULL
cdef void* __nvmlDeviceGetMigMode = NULL
cdef void* __nvmlDeviceGetGpuInstanceProfileInfoV = NULL
cdef void* __nvmlDeviceGetGpuInstancePossiblePlacements_v2 = NULL
cdef void* __nvmlDeviceGetGpuInstanceRemainingCapacity = NULL
cdef void* __nvmlDeviceCreateGpuInstance = NULL
cdef void* __nvmlDeviceCreateGpuInstanceWithPlacement = NULL
cdef void* __nvmlGpuInstanceDestroy = NULL
cdef void* __nvmlDeviceGetGpuInstances = NULL
cdef void* __nvmlDeviceGetGpuInstanceById = NULL
cdef void* __nvmlGpuInstanceGetInfo = NULL
cdef void* __nvmlGpuInstanceGetComputeInstanceProfileInfoV = NULL
cdef void* __nvmlGpuInstanceGetComputeInstanceRemainingCapacity = NULL
cdef void* __nvmlGpuInstanceGetComputeInstancePossiblePlacements = NULL
cdef void* __nvmlGpuInstanceCreateComputeInstance = NULL
cdef void* __nvmlGpuInstanceCreateComputeInstanceWithPlacement = NULL
cdef void* __nvmlComputeInstanceDestroy = NULL
cdef void* __nvmlGpuInstanceGetComputeInstances = NULL
cdef void* __nvmlGpuInstanceGetComputeInstanceById = NULL
cdef void* __nvmlComputeInstanceGetInfo_v2 = NULL
cdef void* __nvmlDeviceIsMigDeviceHandle = NULL
cdef void* __nvmlDeviceGetGpuInstanceId = NULL
cdef void* __nvmlDeviceGetComputeInstanceId = NULL
cdef void* __nvmlDeviceGetMaxMigDeviceCount = NULL
cdef void* __nvmlDeviceGetMigDeviceHandleByIndex = NULL
cdef void* __nvmlDeviceGetDeviceHandleFromMigDeviceHandle = NULL
cdef void* __nvmlDeviceGetCapabilities = NULL
cdef void* __nvmlDevicePowerSmoothingActivatePresetProfile = NULL
cdef void* __nvmlDevicePowerSmoothingUpdatePresetProfileParam = NULL
cdef void* __nvmlDevicePowerSmoothingSetState = NULL
cdef void* __nvmlDeviceGetAddressingMode = NULL
cdef void* __nvmlDeviceGetRepairStatus = NULL
cdef void* __nvmlDeviceGetPowerMizerMode_v1 = NULL
cdef void* __nvmlDeviceSetPowerMizerMode_v1 = NULL
cdef void* __nvmlDeviceGetPdi = NULL
cdef void* __nvmlDeviceSetHostname_v1 = NULL
cdef void* __nvmlDeviceGetHostname_v1 = NULL
cdef void* __nvmlDeviceGetNvLinkInfo = NULL
cdef void* __nvmlDeviceReadWritePRM_v1 = NULL
cdef void* __nvmlDeviceGetGpuInstanceProfileInfoByIdV = NULL
cdef void* __nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts = NULL
cdef void* __nvmlDeviceGetUnrepairableMemoryFlag_v1 = NULL
cdef void* __nvmlDeviceReadPRMCounters_v1 = NULL
cdef void* __nvmlDeviceSetRusdSettings_v1 = NULL
cdef void* __nvmlDeviceVgpuForceGspUnload = NULL
cdef void* __nvmlDeviceGetVgpuSchedulerState_v2 = NULL
cdef void* __nvmlGpuInstanceGetVgpuSchedulerState_v2 = NULL
cdef void* __nvmlDeviceGetVgpuSchedulerLog_v2 = NULL
cdef void* __nvmlGpuInstanceGetVgpuSchedulerLog_v2 = NULL
cdef void* __nvmlDeviceSetVgpuSchedulerState_v2 = NULL
cdef void* __nvmlGpuInstanceSetVgpuSchedulerState_v2 = NULL
cdef void* __nvmlSystemGetCPER_v1 = NULL
cdef void* __nvmlDeviceGetBBXTimeData_v1 = NULL
cdef void* __nvmlDeviceGetAccountingStats_v2 = NULL
cdef void* __nvmlDeviceGetRemappedRows_v2 = NULL
cdef void* __nvmlDeviceSetAdaptiveTgpMode_v1 = NULL
cdef void* __nvmlDeviceGetAdaptiveTgpModeInfo_v1 = NULL
cdef void* __nvmlDeviceSetMemoryLimits_v1 = NULL
cdef void* __nvmlDeviceGetMemoryLimits_v1 = NULL
cdef void* __nvmlDeviceGetGpuFabricInfo_v4 = NULL
cdef void* __nvmlDevicePerfMetricsGetSamples_v1 = NULL
cdef void* __nvmlDeviceSetNvlinkBwModeAsync_v1 = NULL
cdef void* __nvmlDeviceGetNvLinkTelemetrySamples_v1 = NULL
cdef void* __nvmlEventSetRegisterGpuOperationalEvents_v1 = NULL
cdef void* __nvmlEventSetWait_v3 = NULL
cdef void* __nvmlEventSetGetContextCount_v1 = NULL
cdef void* __nvmlEventSetGetContextInfo_v1 = NULL
cdef void* __nvmlEventSetGetContextData_v1 = NULL
cdef void* __nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1 = NULL
cdef void* __nvmlDeviceGetBankRemapperStatus_v1 = NULL
cdef void* __nvmlDeviceGetGpuUptime_v1 = NULL
cdef void* __nvmlSystemGetConfComputeSettings_v2 = NULL
cdef void* __nvmlDeviceGetNvLinkInfo_v3 = NULL
cdef void* __nvmlDeviceGetNvlinkRemapTableInfo_v1 = NULL
cdef void* __nvmlDeviceSetNvlinkNvleReady_v1 = NULL
cdef void* __nvmlDeviceLockNvlinkRemapTable_v1 = NULL
cdef void* __nvmlDeviceSetupNvlinkNvleEncryptionKey_v1 = NULL
cdef void* __nvmlDeviceGetNvlinkNvlePrivateInfo_v1 = NULL

cdef enum: _cyb_nvml_n_functions = 378
cdef void** _cyb_nvml_slots[_cyb_nvml_n_functions]
cdef const char* _cyb_nvml_names[_cyb_nvml_n_functions]
_cyb_nvml_slots[0] = &__nvmlInit_v2; _cyb_nvml_names[0] = b'nvmlInit_v2'
_cyb_nvml_slots[1] = &__nvmlInitWithFlags; _cyb_nvml_names[1] = b'nvmlInitWithFlags'
_cyb_nvml_slots[2] = &__nvmlShutdown; _cyb_nvml_names[2] = b'nvmlShutdown'
_cyb_nvml_slots[3] = &__nvmlErrorString; _cyb_nvml_names[3] = b'nvmlErrorString'
_cyb_nvml_slots[4] = &__nvmlSystemGetDriverVersion; _cyb_nvml_names[4] = b'nvmlSystemGetDriverVersion'
_cyb_nvml_slots[5] = &__nvmlSystemGetNVMLVersion; _cyb_nvml_names[5] = b'nvmlSystemGetNVMLVersion'
_cyb_nvml_slots[6] = &__nvmlSystemGetCudaDriverVersion; _cyb_nvml_names[6] = b'nvmlSystemGetCudaDriverVersion'
_cyb_nvml_slots[7] = &__nvmlSystemGetCudaDriverVersion_v2; _cyb_nvml_names[7] = b'nvmlSystemGetCudaDriverVersion_v2'
_cyb_nvml_slots[8] = &__nvmlSystemGetProcessName; _cyb_nvml_names[8] = b'nvmlSystemGetProcessName'
_cyb_nvml_slots[9] = &__nvmlSystemGetHicVersion; _cyb_nvml_names[9] = b'nvmlSystemGetHicVersion'
_cyb_nvml_slots[10] = &__nvmlSystemGetTopologyGpuSet; _cyb_nvml_names[10] = b'nvmlSystemGetTopologyGpuSet'
_cyb_nvml_slots[11] = &__nvmlSystemGetDriverBranch; _cyb_nvml_names[11] = b'nvmlSystemGetDriverBranch'
_cyb_nvml_slots[12] = &__nvmlUnitGetCount; _cyb_nvml_names[12] = b'nvmlUnitGetCount'
_cyb_nvml_slots[13] = &__nvmlUnitGetHandleByIndex; _cyb_nvml_names[13] = b'nvmlUnitGetHandleByIndex'
_cyb_nvml_slots[14] = &__nvmlUnitGetUnitInfo; _cyb_nvml_names[14] = b'nvmlUnitGetUnitInfo'
_cyb_nvml_slots[15] = &__nvmlUnitGetLedState; _cyb_nvml_names[15] = b'nvmlUnitGetLedState'
_cyb_nvml_slots[16] = &__nvmlUnitGetPsuInfo; _cyb_nvml_names[16] = b'nvmlUnitGetPsuInfo'
_cyb_nvml_slots[17] = &__nvmlUnitGetTemperature; _cyb_nvml_names[17] = b'nvmlUnitGetTemperature'
_cyb_nvml_slots[18] = &__nvmlUnitGetFanSpeedInfo; _cyb_nvml_names[18] = b'nvmlUnitGetFanSpeedInfo'
_cyb_nvml_slots[19] = &__nvmlUnitGetDevices; _cyb_nvml_names[19] = b'nvmlUnitGetDevices'
_cyb_nvml_slots[20] = &__nvmlDeviceGetCount_v2; _cyb_nvml_names[20] = b'nvmlDeviceGetCount_v2'
_cyb_nvml_slots[21] = &__nvmlDeviceGetAttributes_v2; _cyb_nvml_names[21] = b'nvmlDeviceGetAttributes_v2'
_cyb_nvml_slots[22] = &__nvmlDeviceGetHandleByIndex_v2; _cyb_nvml_names[22] = b'nvmlDeviceGetHandleByIndex_v2'
_cyb_nvml_slots[23] = &__nvmlDeviceGetHandleBySerial; _cyb_nvml_names[23] = b'nvmlDeviceGetHandleBySerial'
_cyb_nvml_slots[24] = &__nvmlDeviceGetHandleByUUID; _cyb_nvml_names[24] = b'nvmlDeviceGetHandleByUUID'
_cyb_nvml_slots[25] = &__nvmlDeviceGetHandleByUUIDV; _cyb_nvml_names[25] = b'nvmlDeviceGetHandleByUUIDV'
_cyb_nvml_slots[26] = &__nvmlDeviceGetHandleByPciBusId_v2; _cyb_nvml_names[26] = b'nvmlDeviceGetHandleByPciBusId_v2'
_cyb_nvml_slots[27] = &__nvmlDeviceGetName; _cyb_nvml_names[27] = b'nvmlDeviceGetName'
_cyb_nvml_slots[28] = &__nvmlDeviceGetBrand; _cyb_nvml_names[28] = b'nvmlDeviceGetBrand'
_cyb_nvml_slots[29] = &__nvmlDeviceGetIndex; _cyb_nvml_names[29] = b'nvmlDeviceGetIndex'
_cyb_nvml_slots[30] = &__nvmlDeviceGetSerial; _cyb_nvml_names[30] = b'nvmlDeviceGetSerial'
_cyb_nvml_slots[31] = &__nvmlDeviceGetModuleId; _cyb_nvml_names[31] = b'nvmlDeviceGetModuleId'
_cyb_nvml_slots[32] = &__nvmlDeviceGetC2cModeInfoV; _cyb_nvml_names[32] = b'nvmlDeviceGetC2cModeInfoV'
_cyb_nvml_slots[33] = &__nvmlDeviceGetMemoryAffinity; _cyb_nvml_names[33] = b'nvmlDeviceGetMemoryAffinity'
_cyb_nvml_slots[34] = &__nvmlDeviceGetCpuAffinityWithinScope; _cyb_nvml_names[34] = b'nvmlDeviceGetCpuAffinityWithinScope'
_cyb_nvml_slots[35] = &__nvmlDeviceGetCpuAffinity; _cyb_nvml_names[35] = b'nvmlDeviceGetCpuAffinity'
_cyb_nvml_slots[36] = &__nvmlDeviceSetCpuAffinity; _cyb_nvml_names[36] = b'nvmlDeviceSetCpuAffinity'
_cyb_nvml_slots[37] = &__nvmlDeviceClearCpuAffinity; _cyb_nvml_names[37] = b'nvmlDeviceClearCpuAffinity'
_cyb_nvml_slots[38] = &__nvmlDeviceGetNumaNodeId; _cyb_nvml_names[38] = b'nvmlDeviceGetNumaNodeId'
_cyb_nvml_slots[39] = &__nvmlDeviceGetTopologyCommonAncestor; _cyb_nvml_names[39] = b'nvmlDeviceGetTopologyCommonAncestor'
_cyb_nvml_slots[40] = &__nvmlDeviceGetTopologyNearestGpus; _cyb_nvml_names[40] = b'nvmlDeviceGetTopologyNearestGpus'
_cyb_nvml_slots[41] = &__nvmlDeviceGetP2PStatus; _cyb_nvml_names[41] = b'nvmlDeviceGetP2PStatus'
_cyb_nvml_slots[42] = &__nvmlDeviceGetUUID; _cyb_nvml_names[42] = b'nvmlDeviceGetUUID'
_cyb_nvml_slots[43] = &__nvmlDeviceGetMinorNumber; _cyb_nvml_names[43] = b'nvmlDeviceGetMinorNumber'
_cyb_nvml_slots[44] = &__nvmlDeviceGetBoardPartNumber; _cyb_nvml_names[44] = b'nvmlDeviceGetBoardPartNumber'
_cyb_nvml_slots[45] = &__nvmlDeviceGetInforomVersion; _cyb_nvml_names[45] = b'nvmlDeviceGetInforomVersion'
_cyb_nvml_slots[46] = &__nvmlDeviceGetInforomImageVersion; _cyb_nvml_names[46] = b'nvmlDeviceGetInforomImageVersion'
_cyb_nvml_slots[47] = &__nvmlDeviceGetInforomConfigurationChecksum; _cyb_nvml_names[47] = b'nvmlDeviceGetInforomConfigurationChecksum'
_cyb_nvml_slots[48] = &__nvmlDeviceValidateInforom; _cyb_nvml_names[48] = b'nvmlDeviceValidateInforom'
_cyb_nvml_slots[49] = &__nvmlDeviceGetLastBBXFlushTime; _cyb_nvml_names[49] = b'nvmlDeviceGetLastBBXFlushTime'
_cyb_nvml_slots[50] = &__nvmlDeviceGetDisplayMode; _cyb_nvml_names[50] = b'nvmlDeviceGetDisplayMode'
_cyb_nvml_slots[51] = &__nvmlDeviceGetDisplayActive; _cyb_nvml_names[51] = b'nvmlDeviceGetDisplayActive'
_cyb_nvml_slots[52] = &__nvmlDeviceGetPersistenceMode; _cyb_nvml_names[52] = b'nvmlDeviceGetPersistenceMode'
_cyb_nvml_slots[53] = &__nvmlDeviceGetPciInfoExt; _cyb_nvml_names[53] = b'nvmlDeviceGetPciInfoExt'
_cyb_nvml_slots[54] = &__nvmlDeviceGetPciInfo_v3; _cyb_nvml_names[54] = b'nvmlDeviceGetPciInfo_v3'
_cyb_nvml_slots[55] = &__nvmlDeviceGetMaxPcieLinkGeneration; _cyb_nvml_names[55] = b'nvmlDeviceGetMaxPcieLinkGeneration'
_cyb_nvml_slots[56] = &__nvmlDeviceGetGpuMaxPcieLinkGeneration; _cyb_nvml_names[56] = b'nvmlDeviceGetGpuMaxPcieLinkGeneration'
_cyb_nvml_slots[57] = &__nvmlDeviceGetMaxPcieLinkWidth; _cyb_nvml_names[57] = b'nvmlDeviceGetMaxPcieLinkWidth'
_cyb_nvml_slots[58] = &__nvmlDeviceGetCurrPcieLinkGeneration; _cyb_nvml_names[58] = b'nvmlDeviceGetCurrPcieLinkGeneration'
_cyb_nvml_slots[59] = &__nvmlDeviceGetCurrPcieLinkWidth; _cyb_nvml_names[59] = b'nvmlDeviceGetCurrPcieLinkWidth'
_cyb_nvml_slots[60] = &__nvmlDeviceGetPcieThroughput; _cyb_nvml_names[60] = b'nvmlDeviceGetPcieThroughput'
_cyb_nvml_slots[61] = &__nvmlDeviceGetPcieReplayCounter; _cyb_nvml_names[61] = b'nvmlDeviceGetPcieReplayCounter'
_cyb_nvml_slots[62] = &__nvmlDeviceGetClockInfo; _cyb_nvml_names[62] = b'nvmlDeviceGetClockInfo'
_cyb_nvml_slots[63] = &__nvmlDeviceGetMaxClockInfo; _cyb_nvml_names[63] = b'nvmlDeviceGetMaxClockInfo'
_cyb_nvml_slots[64] = &__nvmlDeviceGetGpcClkVfOffset; _cyb_nvml_names[64] = b'nvmlDeviceGetGpcClkVfOffset'
_cyb_nvml_slots[65] = &__nvmlDeviceGetClock; _cyb_nvml_names[65] = b'nvmlDeviceGetClock'
_cyb_nvml_slots[66] = &__nvmlDeviceGetMaxCustomerBoostClock; _cyb_nvml_names[66] = b'nvmlDeviceGetMaxCustomerBoostClock'
_cyb_nvml_slots[67] = &__nvmlDeviceGetSupportedMemoryClocks; _cyb_nvml_names[67] = b'nvmlDeviceGetSupportedMemoryClocks'
_cyb_nvml_slots[68] = &__nvmlDeviceGetSupportedGraphicsClocks; _cyb_nvml_names[68] = b'nvmlDeviceGetSupportedGraphicsClocks'
_cyb_nvml_slots[69] = &__nvmlDeviceGetAutoBoostedClocksEnabled; _cyb_nvml_names[69] = b'nvmlDeviceGetAutoBoostedClocksEnabled'
_cyb_nvml_slots[70] = &__nvmlDeviceGetFanSpeed; _cyb_nvml_names[70] = b'nvmlDeviceGetFanSpeed'
_cyb_nvml_slots[71] = &__nvmlDeviceGetFanSpeed_v2; _cyb_nvml_names[71] = b'nvmlDeviceGetFanSpeed_v2'
_cyb_nvml_slots[72] = &__nvmlDeviceGetFanSpeedRPM; _cyb_nvml_names[72] = b'nvmlDeviceGetFanSpeedRPM'
_cyb_nvml_slots[73] = &__nvmlDeviceGetTargetFanSpeed; _cyb_nvml_names[73] = b'nvmlDeviceGetTargetFanSpeed'
_cyb_nvml_slots[74] = &__nvmlDeviceGetMinMaxFanSpeed; _cyb_nvml_names[74] = b'nvmlDeviceGetMinMaxFanSpeed'
_cyb_nvml_slots[75] = &__nvmlDeviceGetFanControlPolicy_v2; _cyb_nvml_names[75] = b'nvmlDeviceGetFanControlPolicy_v2'
_cyb_nvml_slots[76] = &__nvmlDeviceGetNumFans; _cyb_nvml_names[76] = b'nvmlDeviceGetNumFans'
_cyb_nvml_slots[77] = &__nvmlDeviceGetCoolerInfo; _cyb_nvml_names[77] = b'nvmlDeviceGetCoolerInfo'
_cyb_nvml_slots[78] = &__nvmlDeviceGetTemperatureV; _cyb_nvml_names[78] = b'nvmlDeviceGetTemperatureV'
_cyb_nvml_slots[79] = &__nvmlDeviceGetTemperatureThreshold; _cyb_nvml_names[79] = b'nvmlDeviceGetTemperatureThreshold'
_cyb_nvml_slots[80] = &__nvmlDeviceGetMarginTemperature; _cyb_nvml_names[80] = b'nvmlDeviceGetMarginTemperature'
_cyb_nvml_slots[81] = &__nvmlDeviceGetThermalSettings; _cyb_nvml_names[81] = b'nvmlDeviceGetThermalSettings'
_cyb_nvml_slots[82] = &__nvmlDeviceGetPerformanceState; _cyb_nvml_names[82] = b'nvmlDeviceGetPerformanceState'
_cyb_nvml_slots[83] = &__nvmlDeviceGetCurrentClocksEventReasons; _cyb_nvml_names[83] = b'nvmlDeviceGetCurrentClocksEventReasons'
_cyb_nvml_slots[84] = &__nvmlDeviceGetSupportedClocksEventReasons; _cyb_nvml_names[84] = b'nvmlDeviceGetSupportedClocksEventReasons'
_cyb_nvml_slots[85] = &__nvmlDeviceGetPowerState; _cyb_nvml_names[85] = b'nvmlDeviceGetPowerState'
_cyb_nvml_slots[86] = &__nvmlDeviceGetDynamicPstatesInfo; _cyb_nvml_names[86] = b'nvmlDeviceGetDynamicPstatesInfo'
_cyb_nvml_slots[87] = &__nvmlDeviceGetMemClkVfOffset; _cyb_nvml_names[87] = b'nvmlDeviceGetMemClkVfOffset'
_cyb_nvml_slots[88] = &__nvmlDeviceGetMinMaxClockOfPState; _cyb_nvml_names[88] = b'nvmlDeviceGetMinMaxClockOfPState'
_cyb_nvml_slots[89] = &__nvmlDeviceGetSupportedPerformanceStates; _cyb_nvml_names[89] = b'nvmlDeviceGetSupportedPerformanceStates'
_cyb_nvml_slots[90] = &__nvmlDeviceGetGpcClkMinMaxVfOffset; _cyb_nvml_names[90] = b'nvmlDeviceGetGpcClkMinMaxVfOffset'
_cyb_nvml_slots[91] = &__nvmlDeviceGetMemClkMinMaxVfOffset; _cyb_nvml_names[91] = b'nvmlDeviceGetMemClkMinMaxVfOffset'
_cyb_nvml_slots[92] = &__nvmlDeviceGetClockOffsets; _cyb_nvml_names[92] = b'nvmlDeviceGetClockOffsets'
_cyb_nvml_slots[93] = &__nvmlDeviceSetClockOffsets; _cyb_nvml_names[93] = b'nvmlDeviceSetClockOffsets'
_cyb_nvml_slots[94] = &__nvmlDeviceGetPerformanceModes; _cyb_nvml_names[94] = b'nvmlDeviceGetPerformanceModes'
_cyb_nvml_slots[95] = &__nvmlDeviceGetCurrentClockFreqs; _cyb_nvml_names[95] = b'nvmlDeviceGetCurrentClockFreqs'
_cyb_nvml_slots[96] = &__nvmlDeviceGetPowerManagementLimit; _cyb_nvml_names[96] = b'nvmlDeviceGetPowerManagementLimit'
_cyb_nvml_slots[97] = &__nvmlDeviceGetPowerManagementLimitConstraints; _cyb_nvml_names[97] = b'nvmlDeviceGetPowerManagementLimitConstraints'
_cyb_nvml_slots[98] = &__nvmlDeviceGetPowerManagementDefaultLimit; _cyb_nvml_names[98] = b'nvmlDeviceGetPowerManagementDefaultLimit'
_cyb_nvml_slots[99] = &__nvmlDeviceGetPowerUsage; _cyb_nvml_names[99] = b'nvmlDeviceGetPowerUsage'
_cyb_nvml_slots[100] = &__nvmlDeviceGetTotalEnergyConsumption; _cyb_nvml_names[100] = b'nvmlDeviceGetTotalEnergyConsumption'
_cyb_nvml_slots[101] = &__nvmlDeviceGetEnforcedPowerLimit; _cyb_nvml_names[101] = b'nvmlDeviceGetEnforcedPowerLimit'
_cyb_nvml_slots[102] = &__nvmlDeviceGetGpuOperationMode; _cyb_nvml_names[102] = b'nvmlDeviceGetGpuOperationMode'
_cyb_nvml_slots[103] = &__nvmlDeviceGetMemoryInfo_v2; _cyb_nvml_names[103] = b'nvmlDeviceGetMemoryInfo_v2'
_cyb_nvml_slots[104] = &__nvmlDeviceGetComputeMode; _cyb_nvml_names[104] = b'nvmlDeviceGetComputeMode'
_cyb_nvml_slots[105] = &__nvmlDeviceGetCudaComputeCapability; _cyb_nvml_names[105] = b'nvmlDeviceGetCudaComputeCapability'
_cyb_nvml_slots[106] = &__nvmlDeviceGetDramEncryptionMode; _cyb_nvml_names[106] = b'nvmlDeviceGetDramEncryptionMode'
_cyb_nvml_slots[107] = &__nvmlDeviceSetDramEncryptionMode; _cyb_nvml_names[107] = b'nvmlDeviceSetDramEncryptionMode'
_cyb_nvml_slots[108] = &__nvmlDeviceGetEccMode; _cyb_nvml_names[108] = b'nvmlDeviceGetEccMode'
_cyb_nvml_slots[109] = &__nvmlDeviceGetDefaultEccMode; _cyb_nvml_names[109] = b'nvmlDeviceGetDefaultEccMode'
_cyb_nvml_slots[110] = &__nvmlDeviceGetBoardId; _cyb_nvml_names[110] = b'nvmlDeviceGetBoardId'
_cyb_nvml_slots[111] = &__nvmlDeviceGetMultiGpuBoard; _cyb_nvml_names[111] = b'nvmlDeviceGetMultiGpuBoard'
_cyb_nvml_slots[112] = &__nvmlDeviceGetTotalEccErrors; _cyb_nvml_names[112] = b'nvmlDeviceGetTotalEccErrors'
_cyb_nvml_slots[113] = &__nvmlDeviceGetMemoryErrorCounter; _cyb_nvml_names[113] = b'nvmlDeviceGetMemoryErrorCounter'
_cyb_nvml_slots[114] = &__nvmlDeviceGetUtilizationRates; _cyb_nvml_names[114] = b'nvmlDeviceGetUtilizationRates'
_cyb_nvml_slots[115] = &__nvmlDeviceGetEncoderUtilization; _cyb_nvml_names[115] = b'nvmlDeviceGetEncoderUtilization'
_cyb_nvml_slots[116] = &__nvmlDeviceGetEncoderCapacity; _cyb_nvml_names[116] = b'nvmlDeviceGetEncoderCapacity'
_cyb_nvml_slots[117] = &__nvmlDeviceGetEncoderStats; _cyb_nvml_names[117] = b'nvmlDeviceGetEncoderStats'
_cyb_nvml_slots[118] = &__nvmlDeviceGetEncoderSessions; _cyb_nvml_names[118] = b'nvmlDeviceGetEncoderSessions'
_cyb_nvml_slots[119] = &__nvmlDeviceGetDecoderUtilization; _cyb_nvml_names[119] = b'nvmlDeviceGetDecoderUtilization'
_cyb_nvml_slots[120] = &__nvmlDeviceGetJpgUtilization; _cyb_nvml_names[120] = b'nvmlDeviceGetJpgUtilization'
_cyb_nvml_slots[121] = &__nvmlDeviceGetOfaUtilization; _cyb_nvml_names[121] = b'nvmlDeviceGetOfaUtilization'
_cyb_nvml_slots[122] = &__nvmlDeviceGetFBCStats; _cyb_nvml_names[122] = b'nvmlDeviceGetFBCStats'
_cyb_nvml_slots[123] = &__nvmlDeviceGetFBCSessions; _cyb_nvml_names[123] = b'nvmlDeviceGetFBCSessions'
_cyb_nvml_slots[124] = &__nvmlDeviceGetDriverModel_v2; _cyb_nvml_names[124] = b'nvmlDeviceGetDriverModel_v2'
_cyb_nvml_slots[125] = &__nvmlDeviceGetVbiosVersion; _cyb_nvml_names[125] = b'nvmlDeviceGetVbiosVersion'
_cyb_nvml_slots[126] = &__nvmlDeviceGetBridgeChipInfo; _cyb_nvml_names[126] = b'nvmlDeviceGetBridgeChipInfo'
_cyb_nvml_slots[127] = &__nvmlDeviceGetComputeRunningProcesses_v3; _cyb_nvml_names[127] = b'nvmlDeviceGetComputeRunningProcesses_v3'
_cyb_nvml_slots[128] = &__nvmlDeviceGetGraphicsRunningProcesses_v3; _cyb_nvml_names[128] = b'nvmlDeviceGetGraphicsRunningProcesses_v3'
_cyb_nvml_slots[129] = &__nvmlDeviceGetMPSComputeRunningProcesses_v3; _cyb_nvml_names[129] = b'nvmlDeviceGetMPSComputeRunningProcesses_v3'
_cyb_nvml_slots[130] = &__nvmlDeviceGetRunningProcessDetailList; _cyb_nvml_names[130] = b'nvmlDeviceGetRunningProcessDetailList'
_cyb_nvml_slots[131] = &__nvmlDeviceOnSameBoard; _cyb_nvml_names[131] = b'nvmlDeviceOnSameBoard'
_cyb_nvml_slots[132] = &__nvmlDeviceGetAPIRestriction; _cyb_nvml_names[132] = b'nvmlDeviceGetAPIRestriction'
_cyb_nvml_slots[133] = &__nvmlDeviceGetSamples; _cyb_nvml_names[133] = b'nvmlDeviceGetSamples'
_cyb_nvml_slots[134] = &__nvmlDeviceGetBAR1MemoryInfo; _cyb_nvml_names[134] = b'nvmlDeviceGetBAR1MemoryInfo'
_cyb_nvml_slots[135] = &__nvmlDeviceGetIrqNum; _cyb_nvml_names[135] = b'nvmlDeviceGetIrqNum'
_cyb_nvml_slots[136] = &__nvmlDeviceGetNumGpuCores; _cyb_nvml_names[136] = b'nvmlDeviceGetNumGpuCores'
_cyb_nvml_slots[137] = &__nvmlDeviceGetPowerSource; _cyb_nvml_names[137] = b'nvmlDeviceGetPowerSource'
_cyb_nvml_slots[138] = &__nvmlDeviceGetMemoryBusWidth; _cyb_nvml_names[138] = b'nvmlDeviceGetMemoryBusWidth'
_cyb_nvml_slots[139] = &__nvmlDeviceGetPcieLinkMaxSpeed; _cyb_nvml_names[139] = b'nvmlDeviceGetPcieLinkMaxSpeed'
_cyb_nvml_slots[140] = &__nvmlDeviceGetPcieSpeed; _cyb_nvml_names[140] = b'nvmlDeviceGetPcieSpeed'
_cyb_nvml_slots[141] = &__nvmlDeviceGetAdaptiveClockInfoStatus; _cyb_nvml_names[141] = b'nvmlDeviceGetAdaptiveClockInfoStatus'
_cyb_nvml_slots[142] = &__nvmlDeviceGetBusType; _cyb_nvml_names[142] = b'nvmlDeviceGetBusType'
_cyb_nvml_slots[143] = &__nvmlDeviceGetGpuFabricInfoV; _cyb_nvml_names[143] = b'nvmlDeviceGetGpuFabricInfoV'
_cyb_nvml_slots[144] = &__nvmlSystemGetConfComputeCapabilities; _cyb_nvml_names[144] = b'nvmlSystemGetConfComputeCapabilities'
_cyb_nvml_slots[145] = &__nvmlSystemGetConfComputeState; _cyb_nvml_names[145] = b'nvmlSystemGetConfComputeState'
_cyb_nvml_slots[146] = &__nvmlDeviceGetConfComputeMemSizeInfo; _cyb_nvml_names[146] = b'nvmlDeviceGetConfComputeMemSizeInfo'
_cyb_nvml_slots[147] = &__nvmlSystemGetConfComputeGpusReadyState; _cyb_nvml_names[147] = b'nvmlSystemGetConfComputeGpusReadyState'
_cyb_nvml_slots[148] = &__nvmlDeviceGetConfComputeProtectedMemoryUsage; _cyb_nvml_names[148] = b'nvmlDeviceGetConfComputeProtectedMemoryUsage'
_cyb_nvml_slots[149] = &__nvmlDeviceGetConfComputeGpuCertificate; _cyb_nvml_names[149] = b'nvmlDeviceGetConfComputeGpuCertificate'
_cyb_nvml_slots[150] = &__nvmlDeviceGetConfComputeGpuAttestationReport; _cyb_nvml_names[150] = b'nvmlDeviceGetConfComputeGpuAttestationReport'
_cyb_nvml_slots[151] = &__nvmlSystemGetConfComputeKeyRotationThresholdInfo; _cyb_nvml_names[151] = b'nvmlSystemGetConfComputeKeyRotationThresholdInfo'
_cyb_nvml_slots[152] = &__nvmlDeviceSetConfComputeUnprotectedMemSize; _cyb_nvml_names[152] = b'nvmlDeviceSetConfComputeUnprotectedMemSize'
_cyb_nvml_slots[153] = &__nvmlSystemSetConfComputeGpusReadyState; _cyb_nvml_names[153] = b'nvmlSystemSetConfComputeGpusReadyState'
_cyb_nvml_slots[154] = &__nvmlSystemSetConfComputeKeyRotationThresholdInfo; _cyb_nvml_names[154] = b'nvmlSystemSetConfComputeKeyRotationThresholdInfo'
_cyb_nvml_slots[155] = &__nvmlSystemGetConfComputeSettings; _cyb_nvml_names[155] = b'nvmlSystemGetConfComputeSettings'
_cyb_nvml_slots[156] = &__nvmlDeviceGetGspFirmwareVersion; _cyb_nvml_names[156] = b'nvmlDeviceGetGspFirmwareVersion'
_cyb_nvml_slots[157] = &__nvmlDeviceGetGspFirmwareMode; _cyb_nvml_names[157] = b'nvmlDeviceGetGspFirmwareMode'
_cyb_nvml_slots[158] = &__nvmlDeviceGetSramEccErrorStatus; _cyb_nvml_names[158] = b'nvmlDeviceGetSramEccErrorStatus'
_cyb_nvml_slots[159] = &__nvmlDeviceGetAccountingMode; _cyb_nvml_names[159] = b'nvmlDeviceGetAccountingMode'
_cyb_nvml_slots[160] = &__nvmlDeviceGetAccountingStats; _cyb_nvml_names[160] = b'nvmlDeviceGetAccountingStats'
_cyb_nvml_slots[161] = &__nvmlDeviceGetAccountingPids; _cyb_nvml_names[161] = b'nvmlDeviceGetAccountingPids'
_cyb_nvml_slots[162] = &__nvmlDeviceGetAccountingBufferSize; _cyb_nvml_names[162] = b'nvmlDeviceGetAccountingBufferSize'
_cyb_nvml_slots[163] = &__nvmlDeviceGetRetiredPages; _cyb_nvml_names[163] = b'nvmlDeviceGetRetiredPages'
_cyb_nvml_slots[164] = &__nvmlDeviceGetRetiredPages_v2; _cyb_nvml_names[164] = b'nvmlDeviceGetRetiredPages_v2'
_cyb_nvml_slots[165] = &__nvmlDeviceGetRetiredPagesPendingStatus; _cyb_nvml_names[165] = b'nvmlDeviceGetRetiredPagesPendingStatus'
_cyb_nvml_slots[166] = &__nvmlDeviceGetRemappedRows; _cyb_nvml_names[166] = b'nvmlDeviceGetRemappedRows'
_cyb_nvml_slots[167] = &__nvmlDeviceGetRowRemapperHistogram; _cyb_nvml_names[167] = b'nvmlDeviceGetRowRemapperHistogram'
_cyb_nvml_slots[168] = &__nvmlDeviceGetArchitecture; _cyb_nvml_names[168] = b'nvmlDeviceGetArchitecture'
_cyb_nvml_slots[169] = &__nvmlDeviceGetClkMonStatus; _cyb_nvml_names[169] = b'nvmlDeviceGetClkMonStatus'
_cyb_nvml_slots[170] = &__nvmlDeviceGetProcessUtilization; _cyb_nvml_names[170] = b'nvmlDeviceGetProcessUtilization'
_cyb_nvml_slots[171] = &__nvmlDeviceGetProcessesUtilizationInfo; _cyb_nvml_names[171] = b'nvmlDeviceGetProcessesUtilizationInfo'
_cyb_nvml_slots[172] = &__nvmlDeviceGetPlatformInfo; _cyb_nvml_names[172] = b'nvmlDeviceGetPlatformInfo'
_cyb_nvml_slots[173] = &__nvmlUnitSetLedState; _cyb_nvml_names[173] = b'nvmlUnitSetLedState'
_cyb_nvml_slots[174] = &__nvmlDeviceSetPersistenceMode; _cyb_nvml_names[174] = b'nvmlDeviceSetPersistenceMode'
_cyb_nvml_slots[175] = &__nvmlDeviceSetComputeMode; _cyb_nvml_names[175] = b'nvmlDeviceSetComputeMode'
_cyb_nvml_slots[176] = &__nvmlDeviceSetEccMode; _cyb_nvml_names[176] = b'nvmlDeviceSetEccMode'
_cyb_nvml_slots[177] = &__nvmlDeviceClearEccErrorCounts; _cyb_nvml_names[177] = b'nvmlDeviceClearEccErrorCounts'
_cyb_nvml_slots[178] = &__nvmlDeviceSetDriverModel; _cyb_nvml_names[178] = b'nvmlDeviceSetDriverModel'
_cyb_nvml_slots[179] = &__nvmlDeviceSetGpuLockedClocks; _cyb_nvml_names[179] = b'nvmlDeviceSetGpuLockedClocks'
_cyb_nvml_slots[180] = &__nvmlDeviceResetGpuLockedClocks; _cyb_nvml_names[180] = b'nvmlDeviceResetGpuLockedClocks'
_cyb_nvml_slots[181] = &__nvmlDeviceSetMemoryLockedClocks; _cyb_nvml_names[181] = b'nvmlDeviceSetMemoryLockedClocks'
_cyb_nvml_slots[182] = &__nvmlDeviceResetMemoryLockedClocks; _cyb_nvml_names[182] = b'nvmlDeviceResetMemoryLockedClocks'
_cyb_nvml_slots[183] = &__nvmlDeviceSetAutoBoostedClocksEnabled; _cyb_nvml_names[183] = b'nvmlDeviceSetAutoBoostedClocksEnabled'
_cyb_nvml_slots[184] = &__nvmlDeviceSetDefaultAutoBoostedClocksEnabled; _cyb_nvml_names[184] = b'nvmlDeviceSetDefaultAutoBoostedClocksEnabled'
_cyb_nvml_slots[185] = &__nvmlDeviceSetDefaultFanSpeed_v2; _cyb_nvml_names[185] = b'nvmlDeviceSetDefaultFanSpeed_v2'
_cyb_nvml_slots[186] = &__nvmlDeviceSetFanControlPolicy; _cyb_nvml_names[186] = b'nvmlDeviceSetFanControlPolicy'
_cyb_nvml_slots[187] = &__nvmlDeviceSetTemperatureThreshold; _cyb_nvml_names[187] = b'nvmlDeviceSetTemperatureThreshold'
_cyb_nvml_slots[188] = &__nvmlDeviceSetGpuOperationMode; _cyb_nvml_names[188] = b'nvmlDeviceSetGpuOperationMode'
_cyb_nvml_slots[189] = &__nvmlDeviceSetAPIRestriction; _cyb_nvml_names[189] = b'nvmlDeviceSetAPIRestriction'
_cyb_nvml_slots[190] = &__nvmlDeviceSetFanSpeed_v2; _cyb_nvml_names[190] = b'nvmlDeviceSetFanSpeed_v2'
_cyb_nvml_slots[191] = &__nvmlDeviceSetAccountingMode; _cyb_nvml_names[191] = b'nvmlDeviceSetAccountingMode'
_cyb_nvml_slots[192] = &__nvmlDeviceClearAccountingPids; _cyb_nvml_names[192] = b'nvmlDeviceClearAccountingPids'
_cyb_nvml_slots[193] = &__nvmlDeviceSetPowerManagementLimit_v2; _cyb_nvml_names[193] = b'nvmlDeviceSetPowerManagementLimit_v2'
_cyb_nvml_slots[194] = &__nvmlDeviceGetNvLinkState; _cyb_nvml_names[194] = b'nvmlDeviceGetNvLinkState'
_cyb_nvml_slots[195] = &__nvmlDeviceGetNvLinkVersion; _cyb_nvml_names[195] = b'nvmlDeviceGetNvLinkVersion'
_cyb_nvml_slots[196] = &__nvmlDeviceGetNvLinkCapability; _cyb_nvml_names[196] = b'nvmlDeviceGetNvLinkCapability'
_cyb_nvml_slots[197] = &__nvmlDeviceGetNvLinkRemotePciInfo_v2; _cyb_nvml_names[197] = b'nvmlDeviceGetNvLinkRemotePciInfo_v2'
_cyb_nvml_slots[198] = &__nvmlDeviceGetNvLinkErrorCounter; _cyb_nvml_names[198] = b'nvmlDeviceGetNvLinkErrorCounter'
_cyb_nvml_slots[199] = &__nvmlDeviceResetNvLinkErrorCounters; _cyb_nvml_names[199] = b'nvmlDeviceResetNvLinkErrorCounters'
_cyb_nvml_slots[200] = &__nvmlDeviceGetNvLinkRemoteDeviceType; _cyb_nvml_names[200] = b'nvmlDeviceGetNvLinkRemoteDeviceType'
_cyb_nvml_slots[201] = &__nvmlDeviceSetNvLinkDeviceLowPowerThreshold; _cyb_nvml_names[201] = b'nvmlDeviceSetNvLinkDeviceLowPowerThreshold'
_cyb_nvml_slots[202] = &__nvmlSystemSetNvlinkBwMode; _cyb_nvml_names[202] = b'nvmlSystemSetNvlinkBwMode'
_cyb_nvml_slots[203] = &__nvmlSystemGetNvlinkBwMode; _cyb_nvml_names[203] = b'nvmlSystemGetNvlinkBwMode'
_cyb_nvml_slots[204] = &__nvmlDeviceGetNvlinkSupportedBwModes; _cyb_nvml_names[204] = b'nvmlDeviceGetNvlinkSupportedBwModes'
_cyb_nvml_slots[205] = &__nvmlDeviceGetNvlinkBwMode; _cyb_nvml_names[205] = b'nvmlDeviceGetNvlinkBwMode'
_cyb_nvml_slots[206] = &__nvmlDeviceSetNvlinkBwMode; _cyb_nvml_names[206] = b'nvmlDeviceSetNvlinkBwMode'
_cyb_nvml_slots[207] = &__nvmlEventSetCreate; _cyb_nvml_names[207] = b'nvmlEventSetCreate'
_cyb_nvml_slots[208] = &__nvmlDeviceRegisterEvents; _cyb_nvml_names[208] = b'nvmlDeviceRegisterEvents'
_cyb_nvml_slots[209] = &__nvmlDeviceGetSupportedEventTypes; _cyb_nvml_names[209] = b'nvmlDeviceGetSupportedEventTypes'
_cyb_nvml_slots[210] = &__nvmlEventSetWait_v2; _cyb_nvml_names[210] = b'nvmlEventSetWait_v2'
_cyb_nvml_slots[211] = &__nvmlEventSetFree; _cyb_nvml_names[211] = b'nvmlEventSetFree'
_cyb_nvml_slots[212] = &__nvmlSystemEventSetCreate; _cyb_nvml_names[212] = b'nvmlSystemEventSetCreate'
_cyb_nvml_slots[213] = &__nvmlSystemEventSetFree; _cyb_nvml_names[213] = b'nvmlSystemEventSetFree'
_cyb_nvml_slots[214] = &__nvmlSystemRegisterEvents; _cyb_nvml_names[214] = b'nvmlSystemRegisterEvents'
_cyb_nvml_slots[215] = &__nvmlSystemEventSetWait; _cyb_nvml_names[215] = b'nvmlSystemEventSetWait'
_cyb_nvml_slots[216] = &__nvmlDeviceModifyDrainState; _cyb_nvml_names[216] = b'nvmlDeviceModifyDrainState'
_cyb_nvml_slots[217] = &__nvmlDeviceQueryDrainState; _cyb_nvml_names[217] = b'nvmlDeviceQueryDrainState'
_cyb_nvml_slots[218] = &__nvmlDeviceRemoveGpu_v2; _cyb_nvml_names[218] = b'nvmlDeviceRemoveGpu_v2'
_cyb_nvml_slots[219] = &__nvmlDeviceDiscoverGpus; _cyb_nvml_names[219] = b'nvmlDeviceDiscoverGpus'
_cyb_nvml_slots[220] = &__nvmlDeviceGetFieldValues; _cyb_nvml_names[220] = b'nvmlDeviceGetFieldValues'
_cyb_nvml_slots[221] = &__nvmlDeviceClearFieldValues; _cyb_nvml_names[221] = b'nvmlDeviceClearFieldValues'
_cyb_nvml_slots[222] = &__nvmlDeviceGetVirtualizationMode; _cyb_nvml_names[222] = b'nvmlDeviceGetVirtualizationMode'
_cyb_nvml_slots[223] = &__nvmlDeviceGetHostVgpuMode; _cyb_nvml_names[223] = b'nvmlDeviceGetHostVgpuMode'
_cyb_nvml_slots[224] = &__nvmlDeviceSetVirtualizationMode; _cyb_nvml_names[224] = b'nvmlDeviceSetVirtualizationMode'
_cyb_nvml_slots[225] = &__nvmlDeviceGetVgpuHeterogeneousMode; _cyb_nvml_names[225] = b'nvmlDeviceGetVgpuHeterogeneousMode'
_cyb_nvml_slots[226] = &__nvmlDeviceSetVgpuHeterogeneousMode; _cyb_nvml_names[226] = b'nvmlDeviceSetVgpuHeterogeneousMode'
_cyb_nvml_slots[227] = &__nvmlVgpuInstanceGetPlacementId; _cyb_nvml_names[227] = b'nvmlVgpuInstanceGetPlacementId'
_cyb_nvml_slots[228] = &__nvmlDeviceGetVgpuTypeSupportedPlacements; _cyb_nvml_names[228] = b'nvmlDeviceGetVgpuTypeSupportedPlacements'
_cyb_nvml_slots[229] = &__nvmlDeviceGetVgpuTypeCreatablePlacements; _cyb_nvml_names[229] = b'nvmlDeviceGetVgpuTypeCreatablePlacements'
_cyb_nvml_slots[230] = &__nvmlVgpuTypeGetGspHeapSize; _cyb_nvml_names[230] = b'nvmlVgpuTypeGetGspHeapSize'
_cyb_nvml_slots[231] = &__nvmlVgpuTypeGetFbReservation; _cyb_nvml_names[231] = b'nvmlVgpuTypeGetFbReservation'
_cyb_nvml_slots[232] = &__nvmlVgpuInstanceGetRuntimeStateSize; _cyb_nvml_names[232] = b'nvmlVgpuInstanceGetRuntimeStateSize'
_cyb_nvml_slots[233] = &__nvmlDeviceSetVgpuCapabilities; _cyb_nvml_names[233] = b'nvmlDeviceSetVgpuCapabilities'
_cyb_nvml_slots[234] = &__nvmlDeviceGetGridLicensableFeatures_v4; _cyb_nvml_names[234] = b'nvmlDeviceGetGridLicensableFeatures_v4'
_cyb_nvml_slots[235] = &__nvmlGetVgpuDriverCapabilities; _cyb_nvml_names[235] = b'nvmlGetVgpuDriverCapabilities'
_cyb_nvml_slots[236] = &__nvmlDeviceGetVgpuCapabilities; _cyb_nvml_names[236] = b'nvmlDeviceGetVgpuCapabilities'
_cyb_nvml_slots[237] = &__nvmlDeviceGetSupportedVgpus; _cyb_nvml_names[237] = b'nvmlDeviceGetSupportedVgpus'
_cyb_nvml_slots[238] = &__nvmlDeviceGetCreatableVgpus; _cyb_nvml_names[238] = b'nvmlDeviceGetCreatableVgpus'
_cyb_nvml_slots[239] = &__nvmlVgpuTypeGetClass; _cyb_nvml_names[239] = b'nvmlVgpuTypeGetClass'
_cyb_nvml_slots[240] = &__nvmlVgpuTypeGetName; _cyb_nvml_names[240] = b'nvmlVgpuTypeGetName'
_cyb_nvml_slots[241] = &__nvmlVgpuTypeGetGpuInstanceProfileId; _cyb_nvml_names[241] = b'nvmlVgpuTypeGetGpuInstanceProfileId'
_cyb_nvml_slots[242] = &__nvmlVgpuTypeGetDeviceID; _cyb_nvml_names[242] = b'nvmlVgpuTypeGetDeviceID'
_cyb_nvml_slots[243] = &__nvmlVgpuTypeGetFramebufferSize; _cyb_nvml_names[243] = b'nvmlVgpuTypeGetFramebufferSize'
_cyb_nvml_slots[244] = &__nvmlVgpuTypeGetNumDisplayHeads; _cyb_nvml_names[244] = b'nvmlVgpuTypeGetNumDisplayHeads'
_cyb_nvml_slots[245] = &__nvmlVgpuTypeGetResolution; _cyb_nvml_names[245] = b'nvmlVgpuTypeGetResolution'
_cyb_nvml_slots[246] = &__nvmlVgpuTypeGetLicense; _cyb_nvml_names[246] = b'nvmlVgpuTypeGetLicense'
_cyb_nvml_slots[247] = &__nvmlVgpuTypeGetFrameRateLimit; _cyb_nvml_names[247] = b'nvmlVgpuTypeGetFrameRateLimit'
_cyb_nvml_slots[248] = &__nvmlVgpuTypeGetMaxInstances; _cyb_nvml_names[248] = b'nvmlVgpuTypeGetMaxInstances'
_cyb_nvml_slots[249] = &__nvmlVgpuTypeGetMaxInstancesPerVm; _cyb_nvml_names[249] = b'nvmlVgpuTypeGetMaxInstancesPerVm'
_cyb_nvml_slots[250] = &__nvmlVgpuTypeGetBAR1Info; _cyb_nvml_names[250] = b'nvmlVgpuTypeGetBAR1Info'
_cyb_nvml_slots[251] = &__nvmlDeviceGetActiveVgpus; _cyb_nvml_names[251] = b'nvmlDeviceGetActiveVgpus'
_cyb_nvml_slots[252] = &__nvmlVgpuInstanceGetVmID; _cyb_nvml_names[252] = b'nvmlVgpuInstanceGetVmID'
_cyb_nvml_slots[253] = &__nvmlVgpuInstanceGetUUID; _cyb_nvml_names[253] = b'nvmlVgpuInstanceGetUUID'
_cyb_nvml_slots[254] = &__nvmlVgpuInstanceGetVmDriverVersion; _cyb_nvml_names[254] = b'nvmlVgpuInstanceGetVmDriverVersion'
_cyb_nvml_slots[255] = &__nvmlVgpuInstanceGetFbUsage; _cyb_nvml_names[255] = b'nvmlVgpuInstanceGetFbUsage'
_cyb_nvml_slots[256] = &__nvmlVgpuInstanceGetLicenseStatus; _cyb_nvml_names[256] = b'nvmlVgpuInstanceGetLicenseStatus'
_cyb_nvml_slots[257] = &__nvmlVgpuInstanceGetType; _cyb_nvml_names[257] = b'nvmlVgpuInstanceGetType'
_cyb_nvml_slots[258] = &__nvmlVgpuInstanceGetFrameRateLimit; _cyb_nvml_names[258] = b'nvmlVgpuInstanceGetFrameRateLimit'
_cyb_nvml_slots[259] = &__nvmlVgpuInstanceGetEccMode; _cyb_nvml_names[259] = b'nvmlVgpuInstanceGetEccMode'
_cyb_nvml_slots[260] = &__nvmlVgpuInstanceGetEncoderCapacity; _cyb_nvml_names[260] = b'nvmlVgpuInstanceGetEncoderCapacity'
_cyb_nvml_slots[261] = &__nvmlVgpuInstanceSetEncoderCapacity; _cyb_nvml_names[261] = b'nvmlVgpuInstanceSetEncoderCapacity'
_cyb_nvml_slots[262] = &__nvmlVgpuInstanceGetEncoderStats; _cyb_nvml_names[262] = b'nvmlVgpuInstanceGetEncoderStats'
_cyb_nvml_slots[263] = &__nvmlVgpuInstanceGetEncoderSessions; _cyb_nvml_names[263] = b'nvmlVgpuInstanceGetEncoderSessions'
_cyb_nvml_slots[264] = &__nvmlVgpuInstanceGetFBCStats; _cyb_nvml_names[264] = b'nvmlVgpuInstanceGetFBCStats'
_cyb_nvml_slots[265] = &__nvmlVgpuInstanceGetFBCSessions; _cyb_nvml_names[265] = b'nvmlVgpuInstanceGetFBCSessions'
_cyb_nvml_slots[266] = &__nvmlVgpuInstanceGetGpuInstanceId; _cyb_nvml_names[266] = b'nvmlVgpuInstanceGetGpuInstanceId'
_cyb_nvml_slots[267] = &__nvmlVgpuInstanceGetGpuPciId; _cyb_nvml_names[267] = b'nvmlVgpuInstanceGetGpuPciId'
_cyb_nvml_slots[268] = &__nvmlVgpuTypeGetCapabilities; _cyb_nvml_names[268] = b'nvmlVgpuTypeGetCapabilities'
_cyb_nvml_slots[269] = &__nvmlVgpuInstanceGetMdevUUID; _cyb_nvml_names[269] = b'nvmlVgpuInstanceGetMdevUUID'
_cyb_nvml_slots[270] = &__nvmlGpuInstanceGetCreatableVgpus; _cyb_nvml_names[270] = b'nvmlGpuInstanceGetCreatableVgpus'
_cyb_nvml_slots[271] = &__nvmlVgpuTypeGetMaxInstancesPerGpuInstance; _cyb_nvml_names[271] = b'nvmlVgpuTypeGetMaxInstancesPerGpuInstance'
_cyb_nvml_slots[272] = &__nvmlGpuInstanceGetActiveVgpus; _cyb_nvml_names[272] = b'nvmlGpuInstanceGetActiveVgpus'
_cyb_nvml_slots[273] = &__nvmlGpuInstanceSetVgpuSchedulerState; _cyb_nvml_names[273] = b'nvmlGpuInstanceSetVgpuSchedulerState'
_cyb_nvml_slots[274] = &__nvmlGpuInstanceGetVgpuSchedulerState; _cyb_nvml_names[274] = b'nvmlGpuInstanceGetVgpuSchedulerState'
_cyb_nvml_slots[275] = &__nvmlGpuInstanceGetVgpuSchedulerLog; _cyb_nvml_names[275] = b'nvmlGpuInstanceGetVgpuSchedulerLog'
_cyb_nvml_slots[276] = &__nvmlGpuInstanceGetVgpuTypeCreatablePlacements; _cyb_nvml_names[276] = b'nvmlGpuInstanceGetVgpuTypeCreatablePlacements'
_cyb_nvml_slots[277] = &__nvmlGpuInstanceGetVgpuHeterogeneousMode; _cyb_nvml_names[277] = b'nvmlGpuInstanceGetVgpuHeterogeneousMode'
_cyb_nvml_slots[278] = &__nvmlGpuInstanceSetVgpuHeterogeneousMode; _cyb_nvml_names[278] = b'nvmlGpuInstanceSetVgpuHeterogeneousMode'
_cyb_nvml_slots[279] = &__nvmlVgpuInstanceGetMetadata; _cyb_nvml_names[279] = b'nvmlVgpuInstanceGetMetadata'
_cyb_nvml_slots[280] = &__nvmlDeviceGetVgpuMetadata; _cyb_nvml_names[280] = b'nvmlDeviceGetVgpuMetadata'
_cyb_nvml_slots[281] = &__nvmlGetVgpuCompatibility; _cyb_nvml_names[281] = b'nvmlGetVgpuCompatibility'
_cyb_nvml_slots[282] = &__nvmlDeviceGetPgpuMetadataString; _cyb_nvml_names[282] = b'nvmlDeviceGetPgpuMetadataString'
_cyb_nvml_slots[283] = &__nvmlDeviceGetVgpuSchedulerLog; _cyb_nvml_names[283] = b'nvmlDeviceGetVgpuSchedulerLog'
_cyb_nvml_slots[284] = &__nvmlDeviceGetVgpuSchedulerState; _cyb_nvml_names[284] = b'nvmlDeviceGetVgpuSchedulerState'
_cyb_nvml_slots[285] = &__nvmlDeviceGetVgpuSchedulerCapabilities; _cyb_nvml_names[285] = b'nvmlDeviceGetVgpuSchedulerCapabilities'
_cyb_nvml_slots[286] = &__nvmlDeviceSetVgpuSchedulerState; _cyb_nvml_names[286] = b'nvmlDeviceSetVgpuSchedulerState'
_cyb_nvml_slots[287] = &__nvmlGetVgpuVersion; _cyb_nvml_names[287] = b'nvmlGetVgpuVersion'
_cyb_nvml_slots[288] = &__nvmlSetVgpuVersion; _cyb_nvml_names[288] = b'nvmlSetVgpuVersion'
_cyb_nvml_slots[289] = &__nvmlDeviceGetVgpuUtilization; _cyb_nvml_names[289] = b'nvmlDeviceGetVgpuUtilization'
_cyb_nvml_slots[290] = &__nvmlDeviceGetVgpuInstancesUtilizationInfo; _cyb_nvml_names[290] = b'nvmlDeviceGetVgpuInstancesUtilizationInfo'
_cyb_nvml_slots[291] = &__nvmlDeviceGetVgpuProcessUtilization; _cyb_nvml_names[291] = b'nvmlDeviceGetVgpuProcessUtilization'
_cyb_nvml_slots[292] = &__nvmlDeviceGetVgpuProcessesUtilizationInfo; _cyb_nvml_names[292] = b'nvmlDeviceGetVgpuProcessesUtilizationInfo'
_cyb_nvml_slots[293] = &__nvmlVgpuInstanceGetAccountingMode; _cyb_nvml_names[293] = b'nvmlVgpuInstanceGetAccountingMode'
_cyb_nvml_slots[294] = &__nvmlVgpuInstanceGetAccountingPids; _cyb_nvml_names[294] = b'nvmlVgpuInstanceGetAccountingPids'
_cyb_nvml_slots[295] = &__nvmlVgpuInstanceGetAccountingStats; _cyb_nvml_names[295] = b'nvmlVgpuInstanceGetAccountingStats'
_cyb_nvml_slots[296] = &__nvmlVgpuInstanceClearAccountingPids; _cyb_nvml_names[296] = b'nvmlVgpuInstanceClearAccountingPids'
_cyb_nvml_slots[297] = &__nvmlVgpuInstanceGetLicenseInfo_v2; _cyb_nvml_names[297] = b'nvmlVgpuInstanceGetLicenseInfo_v2'
_cyb_nvml_slots[298] = &__nvmlGetExcludedDeviceCount; _cyb_nvml_names[298] = b'nvmlGetExcludedDeviceCount'
_cyb_nvml_slots[299] = &__nvmlGetExcludedDeviceInfoByIndex; _cyb_nvml_names[299] = b'nvmlGetExcludedDeviceInfoByIndex'
_cyb_nvml_slots[300] = &__nvmlDeviceSetMigMode; _cyb_nvml_names[300] = b'nvmlDeviceSetMigMode'
_cyb_nvml_slots[301] = &__nvmlDeviceGetMigMode; _cyb_nvml_names[301] = b'nvmlDeviceGetMigMode'
_cyb_nvml_slots[302] = &__nvmlDeviceGetGpuInstanceProfileInfoV; _cyb_nvml_names[302] = b'nvmlDeviceGetGpuInstanceProfileInfoV'
_cyb_nvml_slots[303] = &__nvmlDeviceGetGpuInstancePossiblePlacements_v2; _cyb_nvml_names[303] = b'nvmlDeviceGetGpuInstancePossiblePlacements_v2'
_cyb_nvml_slots[304] = &__nvmlDeviceGetGpuInstanceRemainingCapacity; _cyb_nvml_names[304] = b'nvmlDeviceGetGpuInstanceRemainingCapacity'
_cyb_nvml_slots[305] = &__nvmlDeviceCreateGpuInstance; _cyb_nvml_names[305] = b'nvmlDeviceCreateGpuInstance'
_cyb_nvml_slots[306] = &__nvmlDeviceCreateGpuInstanceWithPlacement; _cyb_nvml_names[306] = b'nvmlDeviceCreateGpuInstanceWithPlacement'
_cyb_nvml_slots[307] = &__nvmlGpuInstanceDestroy; _cyb_nvml_names[307] = b'nvmlGpuInstanceDestroy'
_cyb_nvml_slots[308] = &__nvmlDeviceGetGpuInstances; _cyb_nvml_names[308] = b'nvmlDeviceGetGpuInstances'
_cyb_nvml_slots[309] = &__nvmlDeviceGetGpuInstanceById; _cyb_nvml_names[309] = b'nvmlDeviceGetGpuInstanceById'
_cyb_nvml_slots[310] = &__nvmlGpuInstanceGetInfo; _cyb_nvml_names[310] = b'nvmlGpuInstanceGetInfo'
_cyb_nvml_slots[311] = &__nvmlGpuInstanceGetComputeInstanceProfileInfoV; _cyb_nvml_names[311] = b'nvmlGpuInstanceGetComputeInstanceProfileInfoV'
_cyb_nvml_slots[312] = &__nvmlGpuInstanceGetComputeInstanceRemainingCapacity; _cyb_nvml_names[312] = b'nvmlGpuInstanceGetComputeInstanceRemainingCapacity'
_cyb_nvml_slots[313] = &__nvmlGpuInstanceGetComputeInstancePossiblePlacements; _cyb_nvml_names[313] = b'nvmlGpuInstanceGetComputeInstancePossiblePlacements'
_cyb_nvml_slots[314] = &__nvmlGpuInstanceCreateComputeInstance; _cyb_nvml_names[314] = b'nvmlGpuInstanceCreateComputeInstance'
_cyb_nvml_slots[315] = &__nvmlGpuInstanceCreateComputeInstanceWithPlacement; _cyb_nvml_names[315] = b'nvmlGpuInstanceCreateComputeInstanceWithPlacement'
_cyb_nvml_slots[316] = &__nvmlComputeInstanceDestroy; _cyb_nvml_names[316] = b'nvmlComputeInstanceDestroy'
_cyb_nvml_slots[317] = &__nvmlGpuInstanceGetComputeInstances; _cyb_nvml_names[317] = b'nvmlGpuInstanceGetComputeInstances'
_cyb_nvml_slots[318] = &__nvmlGpuInstanceGetComputeInstanceById; _cyb_nvml_names[318] = b'nvmlGpuInstanceGetComputeInstanceById'
_cyb_nvml_slots[319] = &__nvmlComputeInstanceGetInfo_v2; _cyb_nvml_names[319] = b'nvmlComputeInstanceGetInfo_v2'
_cyb_nvml_slots[320] = &__nvmlDeviceIsMigDeviceHandle; _cyb_nvml_names[320] = b'nvmlDeviceIsMigDeviceHandle'
_cyb_nvml_slots[321] = &__nvmlDeviceGetGpuInstanceId; _cyb_nvml_names[321] = b'nvmlDeviceGetGpuInstanceId'
_cyb_nvml_slots[322] = &__nvmlDeviceGetComputeInstanceId; _cyb_nvml_names[322] = b'nvmlDeviceGetComputeInstanceId'
_cyb_nvml_slots[323] = &__nvmlDeviceGetMaxMigDeviceCount; _cyb_nvml_names[323] = b'nvmlDeviceGetMaxMigDeviceCount'
_cyb_nvml_slots[324] = &__nvmlDeviceGetMigDeviceHandleByIndex; _cyb_nvml_names[324] = b'nvmlDeviceGetMigDeviceHandleByIndex'
_cyb_nvml_slots[325] = &__nvmlDeviceGetDeviceHandleFromMigDeviceHandle; _cyb_nvml_names[325] = b'nvmlDeviceGetDeviceHandleFromMigDeviceHandle'
_cyb_nvml_slots[326] = &__nvmlDeviceGetCapabilities; _cyb_nvml_names[326] = b'nvmlDeviceGetCapabilities'
_cyb_nvml_slots[327] = &__nvmlDevicePowerSmoothingActivatePresetProfile; _cyb_nvml_names[327] = b'nvmlDevicePowerSmoothingActivatePresetProfile'
_cyb_nvml_slots[328] = &__nvmlDevicePowerSmoothingUpdatePresetProfileParam; _cyb_nvml_names[328] = b'nvmlDevicePowerSmoothingUpdatePresetProfileParam'
_cyb_nvml_slots[329] = &__nvmlDevicePowerSmoothingSetState; _cyb_nvml_names[329] = b'nvmlDevicePowerSmoothingSetState'
_cyb_nvml_slots[330] = &__nvmlDeviceGetAddressingMode; _cyb_nvml_names[330] = b'nvmlDeviceGetAddressingMode'
_cyb_nvml_slots[331] = &__nvmlDeviceGetRepairStatus; _cyb_nvml_names[331] = b'nvmlDeviceGetRepairStatus'
_cyb_nvml_slots[332] = &__nvmlDeviceGetPowerMizerMode_v1; _cyb_nvml_names[332] = b'nvmlDeviceGetPowerMizerMode_v1'
_cyb_nvml_slots[333] = &__nvmlDeviceSetPowerMizerMode_v1; _cyb_nvml_names[333] = b'nvmlDeviceSetPowerMizerMode_v1'
_cyb_nvml_slots[334] = &__nvmlDeviceGetPdi; _cyb_nvml_names[334] = b'nvmlDeviceGetPdi'
_cyb_nvml_slots[335] = &__nvmlDeviceSetHostname_v1; _cyb_nvml_names[335] = b'nvmlDeviceSetHostname_v1'
_cyb_nvml_slots[336] = &__nvmlDeviceGetHostname_v1; _cyb_nvml_names[336] = b'nvmlDeviceGetHostname_v1'
_cyb_nvml_slots[337] = &__nvmlDeviceGetNvLinkInfo; _cyb_nvml_names[337] = b'nvmlDeviceGetNvLinkInfo'
_cyb_nvml_slots[338] = &__nvmlDeviceReadWritePRM_v1; _cyb_nvml_names[338] = b'nvmlDeviceReadWritePRM_v1'
_cyb_nvml_slots[339] = &__nvmlDeviceGetGpuInstanceProfileInfoByIdV; _cyb_nvml_names[339] = b'nvmlDeviceGetGpuInstanceProfileInfoByIdV'
_cyb_nvml_slots[340] = &__nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts; _cyb_nvml_names[340] = b'nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts'
_cyb_nvml_slots[341] = &__nvmlDeviceGetUnrepairableMemoryFlag_v1; _cyb_nvml_names[341] = b'nvmlDeviceGetUnrepairableMemoryFlag_v1'
_cyb_nvml_slots[342] = &__nvmlDeviceReadPRMCounters_v1; _cyb_nvml_names[342] = b'nvmlDeviceReadPRMCounters_v1'
_cyb_nvml_slots[343] = &__nvmlDeviceSetRusdSettings_v1; _cyb_nvml_names[343] = b'nvmlDeviceSetRusdSettings_v1'
_cyb_nvml_slots[344] = &__nvmlDeviceVgpuForceGspUnload; _cyb_nvml_names[344] = b'nvmlDeviceVgpuForceGspUnload'
_cyb_nvml_slots[345] = &__nvmlDeviceGetVgpuSchedulerState_v2; _cyb_nvml_names[345] = b'nvmlDeviceGetVgpuSchedulerState_v2'
_cyb_nvml_slots[346] = &__nvmlGpuInstanceGetVgpuSchedulerState_v2; _cyb_nvml_names[346] = b'nvmlGpuInstanceGetVgpuSchedulerState_v2'
_cyb_nvml_slots[347] = &__nvmlDeviceGetVgpuSchedulerLog_v2; _cyb_nvml_names[347] = b'nvmlDeviceGetVgpuSchedulerLog_v2'
_cyb_nvml_slots[348] = &__nvmlGpuInstanceGetVgpuSchedulerLog_v2; _cyb_nvml_names[348] = b'nvmlGpuInstanceGetVgpuSchedulerLog_v2'
_cyb_nvml_slots[349] = &__nvmlDeviceSetVgpuSchedulerState_v2; _cyb_nvml_names[349] = b'nvmlDeviceSetVgpuSchedulerState_v2'
_cyb_nvml_slots[350] = &__nvmlGpuInstanceSetVgpuSchedulerState_v2; _cyb_nvml_names[350] = b'nvmlGpuInstanceSetVgpuSchedulerState_v2'
_cyb_nvml_slots[351] = &__nvmlSystemGetCPER_v1; _cyb_nvml_names[351] = b'nvmlSystemGetCPER_v1'
_cyb_nvml_slots[352] = &__nvmlDeviceGetBBXTimeData_v1; _cyb_nvml_names[352] = b'nvmlDeviceGetBBXTimeData_v1'
_cyb_nvml_slots[353] = &__nvmlDeviceGetAccountingStats_v2; _cyb_nvml_names[353] = b'nvmlDeviceGetAccountingStats_v2'
_cyb_nvml_slots[354] = &__nvmlDeviceGetRemappedRows_v2; _cyb_nvml_names[354] = b'nvmlDeviceGetRemappedRows_v2'
_cyb_nvml_slots[355] = &__nvmlDeviceSetAdaptiveTgpMode_v1; _cyb_nvml_names[355] = b'nvmlDeviceSetAdaptiveTgpMode_v1'
_cyb_nvml_slots[356] = &__nvmlDeviceGetAdaptiveTgpModeInfo_v1; _cyb_nvml_names[356] = b'nvmlDeviceGetAdaptiveTgpModeInfo_v1'
_cyb_nvml_slots[357] = &__nvmlDeviceSetMemoryLimits_v1; _cyb_nvml_names[357] = b'nvmlDeviceSetMemoryLimits_v1'
_cyb_nvml_slots[358] = &__nvmlDeviceGetMemoryLimits_v1; _cyb_nvml_names[358] = b'nvmlDeviceGetMemoryLimits_v1'
_cyb_nvml_slots[359] = &__nvmlDeviceGetGpuFabricInfo_v4; _cyb_nvml_names[359] = b'nvmlDeviceGetGpuFabricInfo_v4'
_cyb_nvml_slots[360] = &__nvmlDevicePerfMetricsGetSamples_v1; _cyb_nvml_names[360] = b'nvmlDevicePerfMetricsGetSamples_v1'
_cyb_nvml_slots[361] = &__nvmlDeviceSetNvlinkBwModeAsync_v1; _cyb_nvml_names[361] = b'nvmlDeviceSetNvlinkBwModeAsync_v1'
_cyb_nvml_slots[362] = &__nvmlDeviceGetNvLinkTelemetrySamples_v1; _cyb_nvml_names[362] = b'nvmlDeviceGetNvLinkTelemetrySamples_v1'
_cyb_nvml_slots[363] = &__nvmlEventSetRegisterGpuOperationalEvents_v1; _cyb_nvml_names[363] = b'nvmlEventSetRegisterGpuOperationalEvents_v1'
_cyb_nvml_slots[364] = &__nvmlEventSetWait_v3; _cyb_nvml_names[364] = b'nvmlEventSetWait_v3'
_cyb_nvml_slots[365] = &__nvmlEventSetGetContextCount_v1; _cyb_nvml_names[365] = b'nvmlEventSetGetContextCount_v1'
_cyb_nvml_slots[366] = &__nvmlEventSetGetContextInfo_v1; _cyb_nvml_names[366] = b'nvmlEventSetGetContextInfo_v1'
_cyb_nvml_slots[367] = &__nvmlEventSetGetContextData_v1; _cyb_nvml_names[367] = b'nvmlEventSetGetContextData_v1'
_cyb_nvml_slots[368] = &__nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1; _cyb_nvml_names[368] = b'nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1'
_cyb_nvml_slots[369] = &__nvmlDeviceGetBankRemapperStatus_v1; _cyb_nvml_names[369] = b'nvmlDeviceGetBankRemapperStatus_v1'
_cyb_nvml_slots[370] = &__nvmlDeviceGetGpuUptime_v1; _cyb_nvml_names[370] = b'nvmlDeviceGetGpuUptime_v1'
_cyb_nvml_slots[371] = &__nvmlSystemGetConfComputeSettings_v2; _cyb_nvml_names[371] = b'nvmlSystemGetConfComputeSettings_v2'
_cyb_nvml_slots[372] = &__nvmlDeviceGetNvLinkInfo_v3; _cyb_nvml_names[372] = b'nvmlDeviceGetNvLinkInfo_v3'
_cyb_nvml_slots[373] = &__nvmlDeviceGetNvlinkRemapTableInfo_v1; _cyb_nvml_names[373] = b'nvmlDeviceGetNvlinkRemapTableInfo_v1'
_cyb_nvml_slots[374] = &__nvmlDeviceSetNvlinkNvleReady_v1; _cyb_nvml_names[374] = b'nvmlDeviceSetNvlinkNvleReady_v1'
_cyb_nvml_slots[375] = &__nvmlDeviceLockNvlinkRemapTable_v1; _cyb_nvml_names[375] = b'nvmlDeviceLockNvlinkRemapTable_v1'
_cyb_nvml_slots[376] = &__nvmlDeviceSetupNvlinkNvleEncryptionKey_v1; _cyb_nvml_names[376] = b'nvmlDeviceSetupNvlinkNvleEncryptionKey_v1'
_cyb_nvml_slots[377] = &__nvmlDeviceGetNvlinkNvlePrivateInfo_v1; _cyb_nvml_names[377] = b'nvmlDeviceGetNvlinkNvlePrivateInfo_v1'

cdef int _init_nvml() except -1 nogil:
    global _cyb___py_nvml_init
    cdef void* handle = NULL
    cdef Py_ssize_t i
    with gil, _cyb_symbol_lock:
        if _cyb___py_nvml_init: return 0

        for i in range(_cyb_nvml_n_functions):
            _cyb_nvml_slots[i][0] = _cyb_dlsym(_cyb_RTLD_DEFAULT, _cyb_nvml_names[i])
            if _cyb_nvml_slots[i][0] == NULL:
                if handle == NULL:
                    handle = load_library()
                _cyb_nvml_slots[i][0] = _cyb_dlsym(handle, _cyb_nvml_names[i])

        _cyb_atomic_int_store(<int *>&_cyb___py_nvml_init, 1)
        return 0

cdef inline int _check_or_init_nvml() except -1 nogil:
    if _cyb_atomic_int_load(<int *>&_cyb___py_nvml_init):
        return 0

    return _init_nvml()


cpdef dict _inspect_function_pointers():
    global _cyb_func_ptrs
    if _cyb_func_ptrs is not None:
        return _cyb_func_ptrs

    _check_or_init_nvml()
    cdef dict data = {}
    cdef Py_ssize_t i
    for i in range(_cyb_nvml_n_functions):
        data['__' + _cyb_nvml_names[i].decode()] = <intptr_t>_cyb_nvml_slots[i][0]
    _cyb_func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global _cyb_func_ptrs
    if _cyb_func_ptrs is None:
        _cyb_func_ptrs = _inspect_function_pointers()
    return _cyb_func_ptrs[name]




cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("nvml")._handle_uint
    return <void*>handle


###############################################################################
# Wrapper functions
###############################################################################

cdef nvmlReturn_t _nvmlInit_v2() except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlInit_v2
    _check_or_init_nvml()
    if __nvmlInit_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlInit_v2 is not found")
    return (<nvmlReturn_t (*)() noexcept nogil>__nvmlInit_v2)(
        )


cdef nvmlReturn_t _nvmlInitWithFlags(unsigned int flags) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlInitWithFlags
    _check_or_init_nvml()
    if __nvmlInitWithFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlInitWithFlags is not found")
    return (<nvmlReturn_t (*)(unsigned int) noexcept nogil>__nvmlInitWithFlags)(
        flags)


cdef nvmlReturn_t _nvmlShutdown() except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlShutdown
    _check_or_init_nvml()
    if __nvmlShutdown == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlShutdown is not found")
    return (<nvmlReturn_t (*)() noexcept nogil>__nvmlShutdown)(
        )


cdef const char* _nvmlErrorString(nvmlReturn_t result) except?NULL nogil:
    global __nvmlErrorString
    _check_or_init_nvml()
    if __nvmlErrorString == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlErrorString is not found")
    return (<const char* (*)(nvmlReturn_t) noexcept nogil>__nvmlErrorString)(
        result)


cdef nvmlReturn_t _nvmlSystemGetDriverVersion(char* version, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemGetDriverVersion
    _check_or_init_nvml()
    if __nvmlSystemGetDriverVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemGetDriverVersion is not found")
    return (<nvmlReturn_t (*)(char*, unsigned int) noexcept nogil>__nvmlSystemGetDriverVersion)(
        version, length)


cdef nvmlReturn_t _nvmlSystemGetNVMLVersion(char* version, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemGetNVMLVersion
    _check_or_init_nvml()
    if __nvmlSystemGetNVMLVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemGetNVMLVersion is not found")
    return (<nvmlReturn_t (*)(char*, unsigned int) noexcept nogil>__nvmlSystemGetNVMLVersion)(
        version, length)


cdef nvmlReturn_t _nvmlSystemGetCudaDriverVersion(int* cudaDriverVersion) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemGetCudaDriverVersion
    _check_or_init_nvml()
    if __nvmlSystemGetCudaDriverVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemGetCudaDriverVersion is not found")
    return (<nvmlReturn_t (*)(int*) noexcept nogil>__nvmlSystemGetCudaDriverVersion)(
        cudaDriverVersion)


cdef nvmlReturn_t _nvmlSystemGetCudaDriverVersion_v2(int* cudaDriverVersion) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemGetCudaDriverVersion_v2
    _check_or_init_nvml()
    if __nvmlSystemGetCudaDriverVersion_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemGetCudaDriverVersion_v2 is not found")
    return (<nvmlReturn_t (*)(int*) noexcept nogil>__nvmlSystemGetCudaDriverVersion_v2)(
        cudaDriverVersion)


cdef nvmlReturn_t _nvmlSystemGetProcessName(unsigned int pid, char* name, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemGetProcessName
    _check_or_init_nvml()
    if __nvmlSystemGetProcessName == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemGetProcessName is not found")
    return (<nvmlReturn_t (*)(unsigned int, char*, unsigned int) noexcept nogil>__nvmlSystemGetProcessName)(
        pid, name, length)


cdef nvmlReturn_t _nvmlSystemGetHicVersion(unsigned int* hwbcCount, nvmlHwbcEntry_t* hwbcEntries) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemGetHicVersion
    _check_or_init_nvml()
    if __nvmlSystemGetHicVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemGetHicVersion is not found")
    return (<nvmlReturn_t (*)(unsigned int*, nvmlHwbcEntry_t*) noexcept nogil>__nvmlSystemGetHicVersion)(
        hwbcCount, hwbcEntries)


cdef nvmlReturn_t _nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber, unsigned int* count, nvmlDevice_t* deviceArray) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemGetTopologyGpuSet
    _check_or_init_nvml()
    if __nvmlSystemGetTopologyGpuSet == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemGetTopologyGpuSet is not found")
    return (<nvmlReturn_t (*)(unsigned int, unsigned int*, nvmlDevice_t*) noexcept nogil>__nvmlSystemGetTopologyGpuSet)(
        cpuNumber, count, deviceArray)


cdef nvmlReturn_t _nvmlSystemGetDriverBranch(nvmlSystemDriverBranchInfo_t* branchInfo, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemGetDriverBranch
    _check_or_init_nvml()
    if __nvmlSystemGetDriverBranch == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemGetDriverBranch is not found")
    return (<nvmlReturn_t (*)(nvmlSystemDriverBranchInfo_t*, unsigned int) noexcept nogil>__nvmlSystemGetDriverBranch)(
        branchInfo, length)


cdef nvmlReturn_t _nvmlUnitGetCount(unsigned int* unitCount) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlUnitGetCount
    _check_or_init_nvml()
    if __nvmlUnitGetCount == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlUnitGetCount is not found")
    return (<nvmlReturn_t (*)(unsigned int*) noexcept nogil>__nvmlUnitGetCount)(
        unitCount)


cdef nvmlReturn_t _nvmlUnitGetHandleByIndex(unsigned int index, nvmlUnit_t* unit) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlUnitGetHandleByIndex
    _check_or_init_nvml()
    if __nvmlUnitGetHandleByIndex == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlUnitGetHandleByIndex is not found")
    return (<nvmlReturn_t (*)(unsigned int, nvmlUnit_t*) noexcept nogil>__nvmlUnitGetHandleByIndex)(
        index, unit)


cdef nvmlReturn_t _nvmlUnitGetUnitInfo(nvmlUnit_t unit, nvmlUnitInfo_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlUnitGetUnitInfo
    _check_or_init_nvml()
    if __nvmlUnitGetUnitInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlUnitGetUnitInfo is not found")
    return (<nvmlReturn_t (*)(nvmlUnit_t, nvmlUnitInfo_t*) noexcept nogil>__nvmlUnitGetUnitInfo)(
        unit, info)


cdef nvmlReturn_t _nvmlUnitGetLedState(nvmlUnit_t unit, nvmlLedState_t* state) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlUnitGetLedState
    _check_or_init_nvml()
    if __nvmlUnitGetLedState == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlUnitGetLedState is not found")
    return (<nvmlReturn_t (*)(nvmlUnit_t, nvmlLedState_t*) noexcept nogil>__nvmlUnitGetLedState)(
        unit, state)


cdef nvmlReturn_t _nvmlUnitGetPsuInfo(nvmlUnit_t unit, nvmlPSUInfo_t* psu) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlUnitGetPsuInfo
    _check_or_init_nvml()
    if __nvmlUnitGetPsuInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlUnitGetPsuInfo is not found")
    return (<nvmlReturn_t (*)(nvmlUnit_t, nvmlPSUInfo_t*) noexcept nogil>__nvmlUnitGetPsuInfo)(
        unit, psu)


cdef nvmlReturn_t _nvmlUnitGetTemperature(nvmlUnit_t unit, unsigned int type, unsigned int* temp) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlUnitGetTemperature
    _check_or_init_nvml()
    if __nvmlUnitGetTemperature == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlUnitGetTemperature is not found")
    return (<nvmlReturn_t (*)(nvmlUnit_t, unsigned int, unsigned int*) noexcept nogil>__nvmlUnitGetTemperature)(
        unit, type, temp)


cdef nvmlReturn_t _nvmlUnitGetFanSpeedInfo(nvmlUnit_t unit, nvmlUnitFanSpeeds_t* fanSpeeds) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlUnitGetFanSpeedInfo
    _check_or_init_nvml()
    if __nvmlUnitGetFanSpeedInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlUnitGetFanSpeedInfo is not found")
    return (<nvmlReturn_t (*)(nvmlUnit_t, nvmlUnitFanSpeeds_t*) noexcept nogil>__nvmlUnitGetFanSpeedInfo)(
        unit, fanSpeeds)


cdef nvmlReturn_t _nvmlUnitGetDevices(nvmlUnit_t unit, unsigned int* deviceCount, nvmlDevice_t* devices) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlUnitGetDevices
    _check_or_init_nvml()
    if __nvmlUnitGetDevices == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlUnitGetDevices is not found")
    return (<nvmlReturn_t (*)(nvmlUnit_t, unsigned int*, nvmlDevice_t*) noexcept nogil>__nvmlUnitGetDevices)(
        unit, deviceCount, devices)


cdef nvmlReturn_t _nvmlDeviceGetCount_v2(unsigned int* deviceCount) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetCount_v2
    _check_or_init_nvml()
    if __nvmlDeviceGetCount_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetCount_v2 is not found")
    return (<nvmlReturn_t (*)(unsigned int*) noexcept nogil>__nvmlDeviceGetCount_v2)(
        deviceCount)


cdef nvmlReturn_t _nvmlDeviceGetAttributes_v2(nvmlDevice_t device, nvmlDeviceAttributes_t* attributes) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetAttributes_v2
    _check_or_init_nvml()
    if __nvmlDeviceGetAttributes_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetAttributes_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlDeviceAttributes_t*) noexcept nogil>__nvmlDeviceGetAttributes_v2)(
        device, attributes)


cdef nvmlReturn_t _nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t* device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetHandleByIndex_v2
    _check_or_init_nvml()
    if __nvmlDeviceGetHandleByIndex_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetHandleByIndex_v2 is not found")
    return (<nvmlReturn_t (*)(unsigned int, nvmlDevice_t*) noexcept nogil>__nvmlDeviceGetHandleByIndex_v2)(
        index, device)


cdef nvmlReturn_t _nvmlDeviceGetHandleBySerial(const char* serial, nvmlDevice_t* device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetHandleBySerial
    _check_or_init_nvml()
    if __nvmlDeviceGetHandleBySerial == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetHandleBySerial is not found")
    return (<nvmlReturn_t (*)(const char*, nvmlDevice_t*) noexcept nogil>__nvmlDeviceGetHandleBySerial)(
        serial, device)


cdef nvmlReturn_t _nvmlDeviceGetHandleByUUID(const char* uuid, nvmlDevice_t* device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetHandleByUUID
    _check_or_init_nvml()
    if __nvmlDeviceGetHandleByUUID == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetHandleByUUID is not found")
    return (<nvmlReturn_t (*)(const char*, nvmlDevice_t*) noexcept nogil>__nvmlDeviceGetHandleByUUID)(
        uuid, device)


cdef nvmlReturn_t _nvmlDeviceGetHandleByUUIDV(const nvmlUUID_t* uuid, nvmlDevice_t* device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetHandleByUUIDV
    _check_or_init_nvml()
    if __nvmlDeviceGetHandleByUUIDV == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetHandleByUUIDV is not found")
    return (<nvmlReturn_t (*)(const nvmlUUID_t*, nvmlDevice_t*) noexcept nogil>__nvmlDeviceGetHandleByUUIDV)(
        uuid, device)


cdef nvmlReturn_t _nvmlDeviceGetHandleByPciBusId_v2(const char* pciBusId, nvmlDevice_t* device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetHandleByPciBusId_v2
    _check_or_init_nvml()
    if __nvmlDeviceGetHandleByPciBusId_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetHandleByPciBusId_v2 is not found")
    return (<nvmlReturn_t (*)(const char*, nvmlDevice_t*) noexcept nogil>__nvmlDeviceGetHandleByPciBusId_v2)(
        pciBusId, device)


cdef nvmlReturn_t _nvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetName
    _check_or_init_nvml()
    if __nvmlDeviceGetName == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetName is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, char*, unsigned int) noexcept nogil>__nvmlDeviceGetName)(
        device, name, length)


cdef nvmlReturn_t _nvmlDeviceGetBrand(nvmlDevice_t device, nvmlBrandType_t* type) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetBrand
    _check_or_init_nvml()
    if __nvmlDeviceGetBrand == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetBrand is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlBrandType_t*) noexcept nogil>__nvmlDeviceGetBrand)(
        device, type)


cdef nvmlReturn_t _nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetIndex
    _check_or_init_nvml()
    if __nvmlDeviceGetIndex == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetIndex is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetIndex)(
        device, index)


cdef nvmlReturn_t _nvmlDeviceGetSerial(nvmlDevice_t device, char* serial, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetSerial
    _check_or_init_nvml()
    if __nvmlDeviceGetSerial == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetSerial is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, char*, unsigned int) noexcept nogil>__nvmlDeviceGetSerial)(
        device, serial, length)


cdef nvmlReturn_t _nvmlDeviceGetModuleId(nvmlDevice_t device, unsigned int* moduleId) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetModuleId
    _check_or_init_nvml()
    if __nvmlDeviceGetModuleId == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetModuleId is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetModuleId)(
        device, moduleId)


cdef nvmlReturn_t _nvmlDeviceGetC2cModeInfoV(nvmlDevice_t device, nvmlC2cModeInfo_v1_t* c2cModeInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetC2cModeInfoV
    _check_or_init_nvml()
    if __nvmlDeviceGetC2cModeInfoV == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetC2cModeInfoV is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlC2cModeInfo_v1_t*) noexcept nogil>__nvmlDeviceGetC2cModeInfoV)(
        device, c2cModeInfo)


cdef nvmlReturn_t _nvmlDeviceGetMemoryAffinity(nvmlDevice_t device, unsigned int nodeSetSize, unsigned long* nodeSet, nvmlAffinityScope_t scope) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetMemoryAffinity
    _check_or_init_nvml()
    if __nvmlDeviceGetMemoryAffinity == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetMemoryAffinity is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned long*, nvmlAffinityScope_t) noexcept nogil>__nvmlDeviceGetMemoryAffinity)(
        device, nodeSetSize, nodeSet, scope)


cdef nvmlReturn_t _nvmlDeviceGetCpuAffinityWithinScope(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long* cpuSet, nvmlAffinityScope_t scope) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetCpuAffinityWithinScope
    _check_or_init_nvml()
    if __nvmlDeviceGetCpuAffinityWithinScope == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetCpuAffinityWithinScope is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned long*, nvmlAffinityScope_t) noexcept nogil>__nvmlDeviceGetCpuAffinityWithinScope)(
        device, cpuSetSize, cpuSet, scope)


cdef nvmlReturn_t _nvmlDeviceGetCpuAffinity(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long* cpuSet) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetCpuAffinity
    _check_or_init_nvml()
    if __nvmlDeviceGetCpuAffinity == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetCpuAffinity is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned long*) noexcept nogil>__nvmlDeviceGetCpuAffinity)(
        device, cpuSetSize, cpuSet)


cdef nvmlReturn_t _nvmlDeviceSetCpuAffinity(nvmlDevice_t device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetCpuAffinity
    _check_or_init_nvml()
    if __nvmlDeviceSetCpuAffinity == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetCpuAffinity is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t) noexcept nogil>__nvmlDeviceSetCpuAffinity)(
        device)


cdef nvmlReturn_t _nvmlDeviceClearCpuAffinity(nvmlDevice_t device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceClearCpuAffinity
    _check_or_init_nvml()
    if __nvmlDeviceClearCpuAffinity == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceClearCpuAffinity is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t) noexcept nogil>__nvmlDeviceClearCpuAffinity)(
        device)


cdef nvmlReturn_t _nvmlDeviceGetNumaNodeId(nvmlDevice_t device, unsigned int* node) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetNumaNodeId
    _check_or_init_nvml()
    if __nvmlDeviceGetNumaNodeId == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetNumaNodeId is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetNumaNodeId)(
        device, node)


cdef nvmlReturn_t _nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuTopologyLevel_t* pathInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetTopologyCommonAncestor
    _check_or_init_nvml()
    if __nvmlDeviceGetTopologyCommonAncestor == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetTopologyCommonAncestor is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlDevice_t, nvmlGpuTopologyLevel_t*) noexcept nogil>__nvmlDeviceGetTopologyCommonAncestor)(
        device1, device2, pathInfo)


cdef nvmlReturn_t _nvmlDeviceGetTopologyNearestGpus(nvmlDevice_t device, nvmlGpuTopologyLevel_t level, unsigned int* count, nvmlDevice_t* deviceArray) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetTopologyNearestGpus
    _check_or_init_nvml()
    if __nvmlDeviceGetTopologyNearestGpus == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetTopologyNearestGpus is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlGpuTopologyLevel_t, unsigned int*, nvmlDevice_t*) noexcept nogil>__nvmlDeviceGetTopologyNearestGpus)(
        device, level, count, deviceArray)


cdef nvmlReturn_t _nvmlDeviceGetP2PStatus(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex, nvmlGpuP2PStatus_t* p2pStatus) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetP2PStatus
    _check_or_init_nvml()
    if __nvmlDeviceGetP2PStatus == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetP2PStatus is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlDevice_t, nvmlGpuP2PCapsIndex_t, nvmlGpuP2PStatus_t*) noexcept nogil>__nvmlDeviceGetP2PStatus)(
        device1, device2, p2pIndex, p2pStatus)


cdef nvmlReturn_t _nvmlDeviceGetUUID(nvmlDevice_t device, char* uuid, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetUUID
    _check_or_init_nvml()
    if __nvmlDeviceGetUUID == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetUUID is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, char*, unsigned int) noexcept nogil>__nvmlDeviceGetUUID)(
        device, uuid, length)


cdef nvmlReturn_t _nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int* minorNumber) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetMinorNumber
    _check_or_init_nvml()
    if __nvmlDeviceGetMinorNumber == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetMinorNumber is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetMinorNumber)(
        device, minorNumber)


cdef nvmlReturn_t _nvmlDeviceGetBoardPartNumber(nvmlDevice_t device, char* partNumber, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetBoardPartNumber
    _check_or_init_nvml()
    if __nvmlDeviceGetBoardPartNumber == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetBoardPartNumber is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, char*, unsigned int) noexcept nogil>__nvmlDeviceGetBoardPartNumber)(
        device, partNumber, length)


cdef nvmlReturn_t _nvmlDeviceGetInforomVersion(nvmlDevice_t device, nvmlInforomObject_t object, char* version, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetInforomVersion
    _check_or_init_nvml()
    if __nvmlDeviceGetInforomVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetInforomVersion is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlInforomObject_t, char*, unsigned int) noexcept nogil>__nvmlDeviceGetInforomVersion)(
        device, object, version, length)


cdef nvmlReturn_t _nvmlDeviceGetInforomImageVersion(nvmlDevice_t device, char* version, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetInforomImageVersion
    _check_or_init_nvml()
    if __nvmlDeviceGetInforomImageVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetInforomImageVersion is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, char*, unsigned int) noexcept nogil>__nvmlDeviceGetInforomImageVersion)(
        device, version, length)


cdef nvmlReturn_t _nvmlDeviceGetInforomConfigurationChecksum(nvmlDevice_t device, unsigned int* checksum) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetInforomConfigurationChecksum
    _check_or_init_nvml()
    if __nvmlDeviceGetInforomConfigurationChecksum == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetInforomConfigurationChecksum is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetInforomConfigurationChecksum)(
        device, checksum)


cdef nvmlReturn_t _nvmlDeviceValidateInforom(nvmlDevice_t device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceValidateInforom
    _check_or_init_nvml()
    if __nvmlDeviceValidateInforom == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceValidateInforom is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t) noexcept nogil>__nvmlDeviceValidateInforom)(
        device)


cdef nvmlReturn_t _nvmlDeviceGetLastBBXFlushTime(nvmlDevice_t device, unsigned long long* timestamp, unsigned long* durationUs) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetLastBBXFlushTime
    _check_or_init_nvml()
    if __nvmlDeviceGetLastBBXFlushTime == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetLastBBXFlushTime is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned long long*, unsigned long*) noexcept nogil>__nvmlDeviceGetLastBBXFlushTime)(
        device, timestamp, durationUs)


cdef nvmlReturn_t _nvmlDeviceGetDisplayMode(nvmlDevice_t device, nvmlEnableState_t* display) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetDisplayMode
    _check_or_init_nvml()
    if __nvmlDeviceGetDisplayMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetDisplayMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t*) noexcept nogil>__nvmlDeviceGetDisplayMode)(
        device, display)


cdef nvmlReturn_t _nvmlDeviceGetDisplayActive(nvmlDevice_t device, nvmlEnableState_t* isActive) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetDisplayActive
    _check_or_init_nvml()
    if __nvmlDeviceGetDisplayActive == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetDisplayActive is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t*) noexcept nogil>__nvmlDeviceGetDisplayActive)(
        device, isActive)


cdef nvmlReturn_t _nvmlDeviceGetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t* mode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetPersistenceMode
    _check_or_init_nvml()
    if __nvmlDeviceGetPersistenceMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetPersistenceMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t*) noexcept nogil>__nvmlDeviceGetPersistenceMode)(
        device, mode)


cdef nvmlReturn_t _nvmlDeviceGetPciInfoExt(nvmlDevice_t device, nvmlPciInfoExt_t* pci) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetPciInfoExt
    _check_or_init_nvml()
    if __nvmlDeviceGetPciInfoExt == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetPciInfoExt is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlPciInfoExt_t*) noexcept nogil>__nvmlDeviceGetPciInfoExt)(
        device, pci)


cdef nvmlReturn_t _nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t* pci) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetPciInfo_v3
    _check_or_init_nvml()
    if __nvmlDeviceGetPciInfo_v3 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetPciInfo_v3 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlPciInfo_t*) noexcept nogil>__nvmlDeviceGetPciInfo_v3)(
        device, pci)


cdef nvmlReturn_t _nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int* maxLinkGen) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetMaxPcieLinkGeneration
    _check_or_init_nvml()
    if __nvmlDeviceGetMaxPcieLinkGeneration == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetMaxPcieLinkGeneration is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetMaxPcieLinkGeneration)(
        device, maxLinkGen)


cdef nvmlReturn_t _nvmlDeviceGetGpuMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int* maxLinkGenDevice) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetGpuMaxPcieLinkGeneration
    _check_or_init_nvml()
    if __nvmlDeviceGetGpuMaxPcieLinkGeneration == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetGpuMaxPcieLinkGeneration is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetGpuMaxPcieLinkGeneration)(
        device, maxLinkGenDevice)


cdef nvmlReturn_t _nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device, unsigned int* maxLinkWidth) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetMaxPcieLinkWidth
    _check_or_init_nvml()
    if __nvmlDeviceGetMaxPcieLinkWidth == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetMaxPcieLinkWidth is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetMaxPcieLinkWidth)(
        device, maxLinkWidth)


cdef nvmlReturn_t _nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int* currLinkGen) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetCurrPcieLinkGeneration
    _check_or_init_nvml()
    if __nvmlDeviceGetCurrPcieLinkGeneration == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetCurrPcieLinkGeneration is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetCurrPcieLinkGeneration)(
        device, currLinkGen)


cdef nvmlReturn_t _nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int* currLinkWidth) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetCurrPcieLinkWidth
    _check_or_init_nvml()
    if __nvmlDeviceGetCurrPcieLinkWidth == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetCurrPcieLinkWidth is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetCurrPcieLinkWidth)(
        device, currLinkWidth)


cdef nvmlReturn_t _nvmlDeviceGetPcieThroughput(nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int* value) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetPcieThroughput
    _check_or_init_nvml()
    if __nvmlDeviceGetPcieThroughput == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetPcieThroughput is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlPcieUtilCounter_t, unsigned int*) noexcept nogil>__nvmlDeviceGetPcieThroughput)(
        device, counter, value)


cdef nvmlReturn_t _nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device, unsigned int* value) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetPcieReplayCounter
    _check_or_init_nvml()
    if __nvmlDeviceGetPcieReplayCounter == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetPcieReplayCounter is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetPcieReplayCounter)(
        device, value)


cdef nvmlReturn_t _nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetClockInfo
    _check_or_init_nvml()
    if __nvmlDeviceGetClockInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetClockInfo is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t, unsigned int*) noexcept nogil>__nvmlDeviceGetClockInfo)(
        device, type, clock)


cdef nvmlReturn_t _nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetMaxClockInfo
    _check_or_init_nvml()
    if __nvmlDeviceGetMaxClockInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetMaxClockInfo is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t, unsigned int*) noexcept nogil>__nvmlDeviceGetMaxClockInfo)(
        device, type, clock)


cdef nvmlReturn_t _nvmlDeviceGetGpcClkVfOffset(nvmlDevice_t device, int* offset) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetGpcClkVfOffset
    _check_or_init_nvml()
    if __nvmlDeviceGetGpcClkVfOffset == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetGpcClkVfOffset is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, int*) noexcept nogil>__nvmlDeviceGetGpcClkVfOffset)(
        device, offset)


cdef nvmlReturn_t _nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int* clockMHz) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetClock
    _check_or_init_nvml()
    if __nvmlDeviceGetClock == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetClock is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t, nvmlClockId_t, unsigned int*) noexcept nogil>__nvmlDeviceGetClock)(
        device, clockType, clockId, clockMHz)


cdef nvmlReturn_t _nvmlDeviceGetMaxCustomerBoostClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int* clockMHz) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetMaxCustomerBoostClock
    _check_or_init_nvml()
    if __nvmlDeviceGetMaxCustomerBoostClock == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetMaxCustomerBoostClock is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t, unsigned int*) noexcept nogil>__nvmlDeviceGetMaxCustomerBoostClock)(
        device, clockType, clockMHz)


cdef nvmlReturn_t _nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device, unsigned int* count, unsigned int* clocksMHz) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetSupportedMemoryClocks
    _check_or_init_nvml()
    if __nvmlDeviceGetSupportedMemoryClocks == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetSupportedMemoryClocks is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, unsigned int*) noexcept nogil>__nvmlDeviceGetSupportedMemoryClocks)(
        device, count, clocksMHz)


cdef nvmlReturn_t _nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t device, unsigned int memoryClockMHz, unsigned int* count, unsigned int* clocksMHz) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetSupportedGraphicsClocks
    _check_or_init_nvml()
    if __nvmlDeviceGetSupportedGraphicsClocks == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetSupportedGraphicsClocks is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int*, unsigned int*) noexcept nogil>__nvmlDeviceGetSupportedGraphicsClocks)(
        device, memoryClockMHz, count, clocksMHz)


cdef nvmlReturn_t _nvmlDeviceGetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t* isEnabled, nvmlEnableState_t* defaultIsEnabled) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetAutoBoostedClocksEnabled
    _check_or_init_nvml()
    if __nvmlDeviceGetAutoBoostedClocksEnabled == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetAutoBoostedClocksEnabled is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t*, nvmlEnableState_t*) noexcept nogil>__nvmlDeviceGetAutoBoostedClocksEnabled)(
        device, isEnabled, defaultIsEnabled)


cdef nvmlReturn_t _nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int* speed) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetFanSpeed
    _check_or_init_nvml()
    if __nvmlDeviceGetFanSpeed == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetFanSpeed is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetFanSpeed)(
        device, speed)


cdef nvmlReturn_t _nvmlDeviceGetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int* speed) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetFanSpeed_v2
    _check_or_init_nvml()
    if __nvmlDeviceGetFanSpeed_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetFanSpeed_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int*) noexcept nogil>__nvmlDeviceGetFanSpeed_v2)(
        device, fan, speed)


cdef nvmlReturn_t _nvmlDeviceGetFanSpeedRPM(nvmlDevice_t device, nvmlFanSpeedInfo_t* fanSpeed) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetFanSpeedRPM
    _check_or_init_nvml()
    if __nvmlDeviceGetFanSpeedRPM == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetFanSpeedRPM is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlFanSpeedInfo_t*) noexcept nogil>__nvmlDeviceGetFanSpeedRPM)(
        device, fanSpeed)


cdef nvmlReturn_t _nvmlDeviceGetTargetFanSpeed(nvmlDevice_t device, unsigned int fan, unsigned int* targetSpeed) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetTargetFanSpeed
    _check_or_init_nvml()
    if __nvmlDeviceGetTargetFanSpeed == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetTargetFanSpeed is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int*) noexcept nogil>__nvmlDeviceGetTargetFanSpeed)(
        device, fan, targetSpeed)


cdef nvmlReturn_t _nvmlDeviceGetMinMaxFanSpeed(nvmlDevice_t device, unsigned int* minSpeed, unsigned int* maxSpeed) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetMinMaxFanSpeed
    _check_or_init_nvml()
    if __nvmlDeviceGetMinMaxFanSpeed == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetMinMaxFanSpeed is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, unsigned int*) noexcept nogil>__nvmlDeviceGetMinMaxFanSpeed)(
        device, minSpeed, maxSpeed)


cdef nvmlReturn_t _nvmlDeviceGetFanControlPolicy_v2(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t* policy) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetFanControlPolicy_v2
    _check_or_init_nvml()
    if __nvmlDeviceGetFanControlPolicy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetFanControlPolicy_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlFanControlPolicy_t*) noexcept nogil>__nvmlDeviceGetFanControlPolicy_v2)(
        device, fan, policy)


cdef nvmlReturn_t _nvmlDeviceGetNumFans(nvmlDevice_t device, unsigned int* numFans) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetNumFans
    _check_or_init_nvml()
    if __nvmlDeviceGetNumFans == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetNumFans is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetNumFans)(
        device, numFans)


cdef nvmlReturn_t _nvmlDeviceGetCoolerInfo(nvmlDevice_t device, nvmlCoolerInfo_t* coolerInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetCoolerInfo
    _check_or_init_nvml()
    if __nvmlDeviceGetCoolerInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetCoolerInfo is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlCoolerInfo_t*) noexcept nogil>__nvmlDeviceGetCoolerInfo)(
        device, coolerInfo)


cdef nvmlReturn_t _nvmlDeviceGetTemperatureV(nvmlDevice_t device, nvmlTemperature_t* temperature) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetTemperatureV
    _check_or_init_nvml()
    if __nvmlDeviceGetTemperatureV == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetTemperatureV is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlTemperature_t*) noexcept nogil>__nvmlDeviceGetTemperatureV)(
        device, temperature)


cdef nvmlReturn_t _nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, unsigned int* temp) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetTemperatureThreshold
    _check_or_init_nvml()
    if __nvmlDeviceGetTemperatureThreshold == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetTemperatureThreshold is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlTemperatureThresholds_t, unsigned int*) noexcept nogil>__nvmlDeviceGetTemperatureThreshold)(
        device, thresholdType, temp)


cdef nvmlReturn_t _nvmlDeviceGetMarginTemperature(nvmlDevice_t device, nvmlMarginTemperature_t* marginTempInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetMarginTemperature
    _check_or_init_nvml()
    if __nvmlDeviceGetMarginTemperature == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetMarginTemperature is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlMarginTemperature_t*) noexcept nogil>__nvmlDeviceGetMarginTemperature)(
        device, marginTempInfo)


cdef nvmlReturn_t _nvmlDeviceGetThermalSettings(nvmlDevice_t device, unsigned int sensorIndex, nvmlGpuThermalSettings_t* pThermalSettings) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetThermalSettings
    _check_or_init_nvml()
    if __nvmlDeviceGetThermalSettings == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetThermalSettings is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlGpuThermalSettings_t*) noexcept nogil>__nvmlDeviceGetThermalSettings)(
        device, sensorIndex, pThermalSettings)


cdef nvmlReturn_t _nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t* pState) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetPerformanceState
    _check_or_init_nvml()
    if __nvmlDeviceGetPerformanceState == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetPerformanceState is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlPstates_t*) noexcept nogil>__nvmlDeviceGetPerformanceState)(
        device, pState)


cdef nvmlReturn_t _nvmlDeviceGetCurrentClocksEventReasons(nvmlDevice_t device, unsigned long long* clocksEventReasons) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetCurrentClocksEventReasons
    _check_or_init_nvml()
    if __nvmlDeviceGetCurrentClocksEventReasons == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetCurrentClocksEventReasons is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned long long*) noexcept nogil>__nvmlDeviceGetCurrentClocksEventReasons)(
        device, clocksEventReasons)


cdef nvmlReturn_t _nvmlDeviceGetSupportedClocksEventReasons(nvmlDevice_t device, unsigned long long* supportedClocksEventReasons) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetSupportedClocksEventReasons
    _check_or_init_nvml()
    if __nvmlDeviceGetSupportedClocksEventReasons == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetSupportedClocksEventReasons is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned long long*) noexcept nogil>__nvmlDeviceGetSupportedClocksEventReasons)(
        device, supportedClocksEventReasons)


cdef nvmlReturn_t _nvmlDeviceGetPowerState(nvmlDevice_t device, nvmlPstates_t* pState) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetPowerState
    _check_or_init_nvml()
    if __nvmlDeviceGetPowerState == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetPowerState is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlPstates_t*) noexcept nogil>__nvmlDeviceGetPowerState)(
        device, pState)


cdef nvmlReturn_t _nvmlDeviceGetDynamicPstatesInfo(nvmlDevice_t device, nvmlGpuDynamicPstatesInfo_t* pDynamicPstatesInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetDynamicPstatesInfo
    _check_or_init_nvml()
    if __nvmlDeviceGetDynamicPstatesInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetDynamicPstatesInfo is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlGpuDynamicPstatesInfo_t*) noexcept nogil>__nvmlDeviceGetDynamicPstatesInfo)(
        device, pDynamicPstatesInfo)


cdef nvmlReturn_t _nvmlDeviceGetMemClkVfOffset(nvmlDevice_t device, int* offset) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetMemClkVfOffset
    _check_or_init_nvml()
    if __nvmlDeviceGetMemClkVfOffset == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetMemClkVfOffset is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, int*) noexcept nogil>__nvmlDeviceGetMemClkVfOffset)(
        device, offset)


cdef nvmlReturn_t _nvmlDeviceGetMinMaxClockOfPState(nvmlDevice_t device, nvmlClockType_t type, nvmlPstates_t pstate, unsigned int* minClockMHz, unsigned int* maxClockMHz) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetMinMaxClockOfPState
    _check_or_init_nvml()
    if __nvmlDeviceGetMinMaxClockOfPState == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetMinMaxClockOfPState is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t, nvmlPstates_t, unsigned int*, unsigned int*) noexcept nogil>__nvmlDeviceGetMinMaxClockOfPState)(
        device, type, pstate, minClockMHz, maxClockMHz)


cdef nvmlReturn_t _nvmlDeviceGetSupportedPerformanceStates(nvmlDevice_t device, nvmlPstates_t* pstates, unsigned int size) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetSupportedPerformanceStates
    _check_or_init_nvml()
    if __nvmlDeviceGetSupportedPerformanceStates == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetSupportedPerformanceStates is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlPstates_t*, unsigned int) noexcept nogil>__nvmlDeviceGetSupportedPerformanceStates)(
        device, pstates, size)


cdef nvmlReturn_t _nvmlDeviceGetGpcClkMinMaxVfOffset(nvmlDevice_t device, int* minOffset, int* maxOffset) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetGpcClkMinMaxVfOffset
    _check_or_init_nvml()
    if __nvmlDeviceGetGpcClkMinMaxVfOffset == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetGpcClkMinMaxVfOffset is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, int*, int*) noexcept nogil>__nvmlDeviceGetGpcClkMinMaxVfOffset)(
        device, minOffset, maxOffset)


cdef nvmlReturn_t _nvmlDeviceGetMemClkMinMaxVfOffset(nvmlDevice_t device, int* minOffset, int* maxOffset) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetMemClkMinMaxVfOffset
    _check_or_init_nvml()
    if __nvmlDeviceGetMemClkMinMaxVfOffset == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetMemClkMinMaxVfOffset is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, int*, int*) noexcept nogil>__nvmlDeviceGetMemClkMinMaxVfOffset)(
        device, minOffset, maxOffset)


cdef nvmlReturn_t _nvmlDeviceGetClockOffsets(nvmlDevice_t device, nvmlClockOffset_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetClockOffsets
    _check_or_init_nvml()
    if __nvmlDeviceGetClockOffsets == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetClockOffsets is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlClockOffset_t*) noexcept nogil>__nvmlDeviceGetClockOffsets)(
        device, info)


cdef nvmlReturn_t _nvmlDeviceSetClockOffsets(nvmlDevice_t device, nvmlClockOffset_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetClockOffsets
    _check_or_init_nvml()
    if __nvmlDeviceSetClockOffsets == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetClockOffsets is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlClockOffset_t*) noexcept nogil>__nvmlDeviceSetClockOffsets)(
        device, info)


cdef nvmlReturn_t _nvmlDeviceGetPerformanceModes(nvmlDevice_t device, nvmlDevicePerfModes_t* perfModes) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetPerformanceModes
    _check_or_init_nvml()
    if __nvmlDeviceGetPerformanceModes == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetPerformanceModes is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlDevicePerfModes_t*) noexcept nogil>__nvmlDeviceGetPerformanceModes)(
        device, perfModes)


cdef nvmlReturn_t _nvmlDeviceGetCurrentClockFreqs(nvmlDevice_t device, nvmlDeviceCurrentClockFreqs_t* currentClockFreqs) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetCurrentClockFreqs
    _check_or_init_nvml()
    if __nvmlDeviceGetCurrentClockFreqs == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetCurrentClockFreqs is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlDeviceCurrentClockFreqs_t*) noexcept nogil>__nvmlDeviceGetCurrentClockFreqs)(
        device, currentClockFreqs)


cdef nvmlReturn_t _nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int* limit) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetPowerManagementLimit
    _check_or_init_nvml()
    if __nvmlDeviceGetPowerManagementLimit == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetPowerManagementLimit is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetPowerManagementLimit)(
        device, limit)


cdef nvmlReturn_t _nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device, unsigned int* minLimit, unsigned int* maxLimit) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetPowerManagementLimitConstraints
    _check_or_init_nvml()
    if __nvmlDeviceGetPowerManagementLimitConstraints == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetPowerManagementLimitConstraints is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, unsigned int*) noexcept nogil>__nvmlDeviceGetPowerManagementLimitConstraints)(
        device, minLimit, maxLimit)


cdef nvmlReturn_t _nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device, unsigned int* defaultLimit) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetPowerManagementDefaultLimit
    _check_or_init_nvml()
    if __nvmlDeviceGetPowerManagementDefaultLimit == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetPowerManagementDefaultLimit is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetPowerManagementDefaultLimit)(
        device, defaultLimit)


cdef nvmlReturn_t _nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int* power) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetPowerUsage
    _check_or_init_nvml()
    if __nvmlDeviceGetPowerUsage == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetPowerUsage is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetPowerUsage)(
        device, power)


cdef nvmlReturn_t _nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t device, unsigned long long* energy) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetTotalEnergyConsumption
    _check_or_init_nvml()
    if __nvmlDeviceGetTotalEnergyConsumption == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetTotalEnergyConsumption is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned long long*) noexcept nogil>__nvmlDeviceGetTotalEnergyConsumption)(
        device, energy)


cdef nvmlReturn_t _nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int* limit) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetEnforcedPowerLimit
    _check_or_init_nvml()
    if __nvmlDeviceGetEnforcedPowerLimit == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetEnforcedPowerLimit is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetEnforcedPowerLimit)(
        device, limit)


cdef nvmlReturn_t _nvmlDeviceGetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t* current, nvmlGpuOperationMode_t* pending) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetGpuOperationMode
    _check_or_init_nvml()
    if __nvmlDeviceGetGpuOperationMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetGpuOperationMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlGpuOperationMode_t*, nvmlGpuOperationMode_t*) noexcept nogil>__nvmlDeviceGetGpuOperationMode)(
        device, current, pending)


cdef nvmlReturn_t _nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t* memory) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetMemoryInfo_v2
    _check_or_init_nvml()
    if __nvmlDeviceGetMemoryInfo_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetMemoryInfo_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_v2_t*) noexcept nogil>__nvmlDeviceGetMemoryInfo_v2)(
        device, memory)


cdef nvmlReturn_t _nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t* mode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetComputeMode
    _check_or_init_nvml()
    if __nvmlDeviceGetComputeMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetComputeMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlComputeMode_t*) noexcept nogil>__nvmlDeviceGetComputeMode)(
        device, mode)


cdef nvmlReturn_t _nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major, int* minor) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetCudaComputeCapability
    _check_or_init_nvml()
    if __nvmlDeviceGetCudaComputeCapability == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetCudaComputeCapability is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, int*, int*) noexcept nogil>__nvmlDeviceGetCudaComputeCapability)(
        device, major, minor)


cdef nvmlReturn_t _nvmlDeviceGetDramEncryptionMode(nvmlDevice_t device, nvmlDramEncryptionInfo_t* current, nvmlDramEncryptionInfo_t* pending) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetDramEncryptionMode
    _check_or_init_nvml()
    if __nvmlDeviceGetDramEncryptionMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetDramEncryptionMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlDramEncryptionInfo_t*, nvmlDramEncryptionInfo_t*) noexcept nogil>__nvmlDeviceGetDramEncryptionMode)(
        device, current, pending)


cdef nvmlReturn_t _nvmlDeviceSetDramEncryptionMode(nvmlDevice_t device, const nvmlDramEncryptionInfo_t* dramEncryption) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetDramEncryptionMode
    _check_or_init_nvml()
    if __nvmlDeviceSetDramEncryptionMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetDramEncryptionMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, const nvmlDramEncryptionInfo_t*) noexcept nogil>__nvmlDeviceSetDramEncryptionMode)(
        device, dramEncryption)


cdef nvmlReturn_t _nvmlDeviceGetEccMode(nvmlDevice_t device, nvmlEnableState_t* current, nvmlEnableState_t* pending) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetEccMode
    _check_or_init_nvml()
    if __nvmlDeviceGetEccMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetEccMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t*, nvmlEnableState_t*) noexcept nogil>__nvmlDeviceGetEccMode)(
        device, current, pending)


cdef nvmlReturn_t _nvmlDeviceGetDefaultEccMode(nvmlDevice_t device, nvmlEnableState_t* defaultMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetDefaultEccMode
    _check_or_init_nvml()
    if __nvmlDeviceGetDefaultEccMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetDefaultEccMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t*) noexcept nogil>__nvmlDeviceGetDefaultEccMode)(
        device, defaultMode)


cdef nvmlReturn_t _nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int* boardId) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetBoardId
    _check_or_init_nvml()
    if __nvmlDeviceGetBoardId == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetBoardId is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetBoardId)(
        device, boardId)


cdef nvmlReturn_t _nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device, unsigned int* multiGpuBool) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetMultiGpuBoard
    _check_or_init_nvml()
    if __nvmlDeviceGetMultiGpuBoard == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetMultiGpuBoard is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetMultiGpuBoard)(
        device, multiGpuBool)


cdef nvmlReturn_t _nvmlDeviceGetTotalEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long* eccCounts) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetTotalEccErrors
    _check_or_init_nvml()
    if __nvmlDeviceGetTotalEccErrors == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetTotalEccErrors is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlMemoryErrorType_t, nvmlEccCounterType_t, unsigned long long*) noexcept nogil>__nvmlDeviceGetTotalEccErrors)(
        device, errorType, counterType, eccCounts)


cdef nvmlReturn_t _nvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlMemoryLocation_t locationType, unsigned long long* count) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetMemoryErrorCounter
    _check_or_init_nvml()
    if __nvmlDeviceGetMemoryErrorCounter == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetMemoryErrorCounter is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlMemoryErrorType_t, nvmlEccCounterType_t, nvmlMemoryLocation_t, unsigned long long*) noexcept nogil>__nvmlDeviceGetMemoryErrorCounter)(
        device, errorType, counterType, locationType, count)


cdef nvmlReturn_t _nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t* utilization) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetUtilizationRates
    _check_or_init_nvml()
    if __nvmlDeviceGetUtilizationRates == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetUtilizationRates is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlUtilization_t*) noexcept nogil>__nvmlDeviceGetUtilizationRates)(
        device, utilization)


cdef nvmlReturn_t _nvmlDeviceGetEncoderUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetEncoderUtilization
    _check_or_init_nvml()
    if __nvmlDeviceGetEncoderUtilization == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetEncoderUtilization is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, unsigned int*) noexcept nogil>__nvmlDeviceGetEncoderUtilization)(
        device, utilization, samplingPeriodUs)


cdef nvmlReturn_t _nvmlDeviceGetEncoderCapacity(nvmlDevice_t device, nvmlEncoderType_t encoderQueryType, unsigned int* encoderCapacity) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetEncoderCapacity
    _check_or_init_nvml()
    if __nvmlDeviceGetEncoderCapacity == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetEncoderCapacity is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlEncoderType_t, unsigned int*) noexcept nogil>__nvmlDeviceGetEncoderCapacity)(
        device, encoderQueryType, encoderCapacity)


cdef nvmlReturn_t _nvmlDeviceGetEncoderStats(nvmlDevice_t device, unsigned int* sessionCount, unsigned int* averageFps, unsigned int* averageLatency) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetEncoderStats
    _check_or_init_nvml()
    if __nvmlDeviceGetEncoderStats == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetEncoderStats is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, unsigned int*, unsigned int*) noexcept nogil>__nvmlDeviceGetEncoderStats)(
        device, sessionCount, averageFps, averageLatency)


cdef nvmlReturn_t _nvmlDeviceGetEncoderSessions(nvmlDevice_t device, unsigned int* sessionCount, nvmlEncoderSessionInfo_t* sessionInfos) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetEncoderSessions
    _check_or_init_nvml()
    if __nvmlDeviceGetEncoderSessions == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetEncoderSessions is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, nvmlEncoderSessionInfo_t*) noexcept nogil>__nvmlDeviceGetEncoderSessions)(
        device, sessionCount, sessionInfos)


cdef nvmlReturn_t _nvmlDeviceGetDecoderUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetDecoderUtilization
    _check_or_init_nvml()
    if __nvmlDeviceGetDecoderUtilization == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetDecoderUtilization is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, unsigned int*) noexcept nogil>__nvmlDeviceGetDecoderUtilization)(
        device, utilization, samplingPeriodUs)


cdef nvmlReturn_t _nvmlDeviceGetJpgUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetJpgUtilization
    _check_or_init_nvml()
    if __nvmlDeviceGetJpgUtilization == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetJpgUtilization is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, unsigned int*) noexcept nogil>__nvmlDeviceGetJpgUtilization)(
        device, utilization, samplingPeriodUs)


cdef nvmlReturn_t _nvmlDeviceGetOfaUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetOfaUtilization
    _check_or_init_nvml()
    if __nvmlDeviceGetOfaUtilization == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetOfaUtilization is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, unsigned int*) noexcept nogil>__nvmlDeviceGetOfaUtilization)(
        device, utilization, samplingPeriodUs)


cdef nvmlReturn_t _nvmlDeviceGetFBCStats(nvmlDevice_t device, nvmlFBCStats_t* fbcStats) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetFBCStats
    _check_or_init_nvml()
    if __nvmlDeviceGetFBCStats == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetFBCStats is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlFBCStats_t*) noexcept nogil>__nvmlDeviceGetFBCStats)(
        device, fbcStats)


cdef nvmlReturn_t _nvmlDeviceGetFBCSessions(nvmlDevice_t device, unsigned int* sessionCount, nvmlFBCSessionInfo_t* sessionInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetFBCSessions
    _check_or_init_nvml()
    if __nvmlDeviceGetFBCSessions == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetFBCSessions is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, nvmlFBCSessionInfo_t*) noexcept nogil>__nvmlDeviceGetFBCSessions)(
        device, sessionCount, sessionInfo)


cdef nvmlReturn_t _nvmlDeviceGetDriverModel_v2(nvmlDevice_t device, nvmlDriverModel_t* current, nvmlDriverModel_t* pending) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetDriverModel_v2
    _check_or_init_nvml()
    if __nvmlDeviceGetDriverModel_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetDriverModel_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlDriverModel_t*, nvmlDriverModel_t*) noexcept nogil>__nvmlDeviceGetDriverModel_v2)(
        device, current, pending)


cdef nvmlReturn_t _nvmlDeviceGetVbiosVersion(nvmlDevice_t device, char* version, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetVbiosVersion
    _check_or_init_nvml()
    if __nvmlDeviceGetVbiosVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetVbiosVersion is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, char*, unsigned int) noexcept nogil>__nvmlDeviceGetVbiosVersion)(
        device, version, length)


cdef nvmlReturn_t _nvmlDeviceGetBridgeChipInfo(nvmlDevice_t device, nvmlBridgeChipHierarchy_t* bridgeHierarchy) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetBridgeChipInfo
    _check_or_init_nvml()
    if __nvmlDeviceGetBridgeChipInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetBridgeChipInfo is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlBridgeChipHierarchy_t*) noexcept nogil>__nvmlDeviceGetBridgeChipInfo)(
        device, bridgeHierarchy)


cdef nvmlReturn_t _nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetComputeRunningProcesses_v3
    _check_or_init_nvml()
    if __nvmlDeviceGetComputeRunningProcesses_v3 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetComputeRunningProcesses_v3 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, nvmlProcessInfo_t*) noexcept nogil>__nvmlDeviceGetComputeRunningProcesses_v3)(
        device, infoCount, infos)


cdef nvmlReturn_t _nvmlDeviceGetGraphicsRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetGraphicsRunningProcesses_v3
    _check_or_init_nvml()
    if __nvmlDeviceGetGraphicsRunningProcesses_v3 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetGraphicsRunningProcesses_v3 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, nvmlProcessInfo_t*) noexcept nogil>__nvmlDeviceGetGraphicsRunningProcesses_v3)(
        device, infoCount, infos)


cdef nvmlReturn_t _nvmlDeviceGetMPSComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetMPSComputeRunningProcesses_v3
    _check_or_init_nvml()
    if __nvmlDeviceGetMPSComputeRunningProcesses_v3 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetMPSComputeRunningProcesses_v3 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, nvmlProcessInfo_t*) noexcept nogil>__nvmlDeviceGetMPSComputeRunningProcesses_v3)(
        device, infoCount, infos)


cdef nvmlReturn_t _nvmlDeviceGetRunningProcessDetailList(nvmlDevice_t device, nvmlProcessDetailList_t* plist) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetRunningProcessDetailList
    _check_or_init_nvml()
    if __nvmlDeviceGetRunningProcessDetailList == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetRunningProcessDetailList is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlProcessDetailList_t*) noexcept nogil>__nvmlDeviceGetRunningProcessDetailList)(
        device, plist)


cdef nvmlReturn_t _nvmlDeviceOnSameBoard(nvmlDevice_t device1, nvmlDevice_t device2, int* onSameBoard) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceOnSameBoard
    _check_or_init_nvml()
    if __nvmlDeviceOnSameBoard == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceOnSameBoard is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlDevice_t, int*) noexcept nogil>__nvmlDeviceOnSameBoard)(
        device1, device2, onSameBoard)


cdef nvmlReturn_t _nvmlDeviceGetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t* isRestricted) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetAPIRestriction
    _check_or_init_nvml()
    if __nvmlDeviceGetAPIRestriction == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetAPIRestriction is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlRestrictedAPI_t, nvmlEnableState_t*) noexcept nogil>__nvmlDeviceGetAPIRestriction)(
        device, apiType, isRestricted)


cdef nvmlReturn_t _nvmlDeviceGetSamples(nvmlDevice_t device, nvmlSamplingType_t type, unsigned long long lastSeenTimeStamp, nvmlValueType_t* sampleValType, unsigned int* sampleCount, nvmlSample_t* samples) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetSamples
    _check_or_init_nvml()
    if __nvmlDeviceGetSamples == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetSamples is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlSamplingType_t, unsigned long long, nvmlValueType_t*, unsigned int*, nvmlSample_t*) noexcept nogil>__nvmlDeviceGetSamples)(
        device, type, lastSeenTimeStamp, sampleValType, sampleCount, samples)


cdef nvmlReturn_t _nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t device, nvmlBAR1Memory_t* bar1Memory) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetBAR1MemoryInfo
    _check_or_init_nvml()
    if __nvmlDeviceGetBAR1MemoryInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetBAR1MemoryInfo is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlBAR1Memory_t*) noexcept nogil>__nvmlDeviceGetBAR1MemoryInfo)(
        device, bar1Memory)


cdef nvmlReturn_t _nvmlDeviceGetIrqNum(nvmlDevice_t device, unsigned int* irqNum) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetIrqNum
    _check_or_init_nvml()
    if __nvmlDeviceGetIrqNum == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetIrqNum is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetIrqNum)(
        device, irqNum)


cdef nvmlReturn_t _nvmlDeviceGetNumGpuCores(nvmlDevice_t device, unsigned int* numCores) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetNumGpuCores
    _check_or_init_nvml()
    if __nvmlDeviceGetNumGpuCores == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetNumGpuCores is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetNumGpuCores)(
        device, numCores)


cdef nvmlReturn_t _nvmlDeviceGetPowerSource(nvmlDevice_t device, nvmlPowerSource_t* powerSource) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetPowerSource
    _check_or_init_nvml()
    if __nvmlDeviceGetPowerSource == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetPowerSource is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlPowerSource_t*) noexcept nogil>__nvmlDeviceGetPowerSource)(
        device, powerSource)


cdef nvmlReturn_t _nvmlDeviceGetMemoryBusWidth(nvmlDevice_t device, unsigned int* busWidth) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetMemoryBusWidth
    _check_or_init_nvml()
    if __nvmlDeviceGetMemoryBusWidth == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetMemoryBusWidth is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetMemoryBusWidth)(
        device, busWidth)


cdef nvmlReturn_t _nvmlDeviceGetPcieLinkMaxSpeed(nvmlDevice_t device, unsigned int* maxSpeed) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetPcieLinkMaxSpeed
    _check_or_init_nvml()
    if __nvmlDeviceGetPcieLinkMaxSpeed == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetPcieLinkMaxSpeed is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetPcieLinkMaxSpeed)(
        device, maxSpeed)


cdef nvmlReturn_t _nvmlDeviceGetPcieSpeed(nvmlDevice_t device, unsigned int* pcieSpeed) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetPcieSpeed
    _check_or_init_nvml()
    if __nvmlDeviceGetPcieSpeed == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetPcieSpeed is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetPcieSpeed)(
        device, pcieSpeed)


cdef nvmlReturn_t _nvmlDeviceGetAdaptiveClockInfoStatus(nvmlDevice_t device, unsigned int* adaptiveClockStatus) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetAdaptiveClockInfoStatus
    _check_or_init_nvml()
    if __nvmlDeviceGetAdaptiveClockInfoStatus == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetAdaptiveClockInfoStatus is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetAdaptiveClockInfoStatus)(
        device, adaptiveClockStatus)


cdef nvmlReturn_t _nvmlDeviceGetBusType(nvmlDevice_t device, nvmlBusType_t* type) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetBusType
    _check_or_init_nvml()
    if __nvmlDeviceGetBusType == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetBusType is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlBusType_t*) noexcept nogil>__nvmlDeviceGetBusType)(
        device, type)


cdef nvmlReturn_t _nvmlDeviceGetGpuFabricInfoV(nvmlDevice_t device, nvmlGpuFabricInfoV_t* gpuFabricInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetGpuFabricInfoV
    _check_or_init_nvml()
    if __nvmlDeviceGetGpuFabricInfoV == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetGpuFabricInfoV is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlGpuFabricInfoV_t*) noexcept nogil>__nvmlDeviceGetGpuFabricInfoV)(
        device, gpuFabricInfo)


cdef nvmlReturn_t _nvmlSystemGetConfComputeCapabilities(nvmlConfComputeSystemCaps_t* capabilities) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemGetConfComputeCapabilities
    _check_or_init_nvml()
    if __nvmlSystemGetConfComputeCapabilities == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemGetConfComputeCapabilities is not found")
    return (<nvmlReturn_t (*)(nvmlConfComputeSystemCaps_t*) noexcept nogil>__nvmlSystemGetConfComputeCapabilities)(
        capabilities)


cdef nvmlReturn_t _nvmlSystemGetConfComputeState(nvmlConfComputeSystemState_t* state) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemGetConfComputeState
    _check_or_init_nvml()
    if __nvmlSystemGetConfComputeState == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemGetConfComputeState is not found")
    return (<nvmlReturn_t (*)(nvmlConfComputeSystemState_t*) noexcept nogil>__nvmlSystemGetConfComputeState)(
        state)


cdef nvmlReturn_t _nvmlDeviceGetConfComputeMemSizeInfo(nvmlDevice_t device, nvmlConfComputeMemSizeInfo_t* memInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetConfComputeMemSizeInfo
    _check_or_init_nvml()
    if __nvmlDeviceGetConfComputeMemSizeInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetConfComputeMemSizeInfo is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlConfComputeMemSizeInfo_t*) noexcept nogil>__nvmlDeviceGetConfComputeMemSizeInfo)(
        device, memInfo)


cdef nvmlReturn_t _nvmlSystemGetConfComputeGpusReadyState(unsigned int* isAcceptingWork) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemGetConfComputeGpusReadyState
    _check_or_init_nvml()
    if __nvmlSystemGetConfComputeGpusReadyState == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemGetConfComputeGpusReadyState is not found")
    return (<nvmlReturn_t (*)(unsigned int*) noexcept nogil>__nvmlSystemGetConfComputeGpusReadyState)(
        isAcceptingWork)


cdef nvmlReturn_t _nvmlDeviceGetConfComputeProtectedMemoryUsage(nvmlDevice_t device, nvmlMemory_t* memory) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetConfComputeProtectedMemoryUsage
    _check_or_init_nvml()
    if __nvmlDeviceGetConfComputeProtectedMemoryUsage == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetConfComputeProtectedMemoryUsage is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t*) noexcept nogil>__nvmlDeviceGetConfComputeProtectedMemoryUsage)(
        device, memory)


cdef nvmlReturn_t _nvmlDeviceGetConfComputeGpuCertificate(nvmlDevice_t device, nvmlConfComputeGpuCertificate_t* gpuCert) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetConfComputeGpuCertificate
    _check_or_init_nvml()
    if __nvmlDeviceGetConfComputeGpuCertificate == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetConfComputeGpuCertificate is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlConfComputeGpuCertificate_t*) noexcept nogil>__nvmlDeviceGetConfComputeGpuCertificate)(
        device, gpuCert)


cdef nvmlReturn_t _nvmlDeviceGetConfComputeGpuAttestationReport(nvmlDevice_t device, nvmlConfComputeGpuAttestationReport_t* gpuAtstReport) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetConfComputeGpuAttestationReport
    _check_or_init_nvml()
    if __nvmlDeviceGetConfComputeGpuAttestationReport == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetConfComputeGpuAttestationReport is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlConfComputeGpuAttestationReport_t*) noexcept nogil>__nvmlDeviceGetConfComputeGpuAttestationReport)(
        device, gpuAtstReport)


cdef nvmlReturn_t _nvmlSystemGetConfComputeKeyRotationThresholdInfo(nvmlConfComputeGetKeyRotationThresholdInfo_t* pKeyRotationThrInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemGetConfComputeKeyRotationThresholdInfo
    _check_or_init_nvml()
    if __nvmlSystemGetConfComputeKeyRotationThresholdInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemGetConfComputeKeyRotationThresholdInfo is not found")
    return (<nvmlReturn_t (*)(nvmlConfComputeGetKeyRotationThresholdInfo_t*) noexcept nogil>__nvmlSystemGetConfComputeKeyRotationThresholdInfo)(
        pKeyRotationThrInfo)


cdef nvmlReturn_t _nvmlDeviceSetConfComputeUnprotectedMemSize(nvmlDevice_t device, unsigned long long sizeKiB) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetConfComputeUnprotectedMemSize
    _check_or_init_nvml()
    if __nvmlDeviceSetConfComputeUnprotectedMemSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetConfComputeUnprotectedMemSize is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned long long) noexcept nogil>__nvmlDeviceSetConfComputeUnprotectedMemSize)(
        device, sizeKiB)


cdef nvmlReturn_t _nvmlSystemSetConfComputeGpusReadyState(unsigned int isAcceptingWork) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemSetConfComputeGpusReadyState
    _check_or_init_nvml()
    if __nvmlSystemSetConfComputeGpusReadyState == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemSetConfComputeGpusReadyState is not found")
    return (<nvmlReturn_t (*)(unsigned int) noexcept nogil>__nvmlSystemSetConfComputeGpusReadyState)(
        isAcceptingWork)


cdef nvmlReturn_t _nvmlSystemSetConfComputeKeyRotationThresholdInfo(nvmlConfComputeSetKeyRotationThresholdInfo_t* pKeyRotationThrInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemSetConfComputeKeyRotationThresholdInfo
    _check_or_init_nvml()
    if __nvmlSystemSetConfComputeKeyRotationThresholdInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemSetConfComputeKeyRotationThresholdInfo is not found")
    return (<nvmlReturn_t (*)(nvmlConfComputeSetKeyRotationThresholdInfo_t*) noexcept nogil>__nvmlSystemSetConfComputeKeyRotationThresholdInfo)(
        pKeyRotationThrInfo)


cdef nvmlReturn_t _nvmlSystemGetConfComputeSettings(nvmlSystemConfComputeSettings_t* settings) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemGetConfComputeSettings
    _check_or_init_nvml()
    if __nvmlSystemGetConfComputeSettings == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemGetConfComputeSettings is not found")
    return (<nvmlReturn_t (*)(nvmlSystemConfComputeSettings_t*) noexcept nogil>__nvmlSystemGetConfComputeSettings)(
        settings)


cdef nvmlReturn_t _nvmlDeviceGetGspFirmwareVersion(nvmlDevice_t device, char* version) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetGspFirmwareVersion
    _check_or_init_nvml()
    if __nvmlDeviceGetGspFirmwareVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetGspFirmwareVersion is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, char*) noexcept nogil>__nvmlDeviceGetGspFirmwareVersion)(
        device, version)


cdef nvmlReturn_t _nvmlDeviceGetGspFirmwareMode(nvmlDevice_t device, unsigned int* isEnabled, unsigned int* defaultMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetGspFirmwareMode
    _check_or_init_nvml()
    if __nvmlDeviceGetGspFirmwareMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetGspFirmwareMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, unsigned int*) noexcept nogil>__nvmlDeviceGetGspFirmwareMode)(
        device, isEnabled, defaultMode)


cdef nvmlReturn_t _nvmlDeviceGetSramEccErrorStatus(nvmlDevice_t device, nvmlEccSramErrorStatus_t* status) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetSramEccErrorStatus
    _check_or_init_nvml()
    if __nvmlDeviceGetSramEccErrorStatus == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetSramEccErrorStatus is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlEccSramErrorStatus_t*) noexcept nogil>__nvmlDeviceGetSramEccErrorStatus)(
        device, status)


cdef nvmlReturn_t _nvmlDeviceGetAccountingMode(nvmlDevice_t device, nvmlEnableState_t* mode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetAccountingMode
    _check_or_init_nvml()
    if __nvmlDeviceGetAccountingMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetAccountingMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t*) noexcept nogil>__nvmlDeviceGetAccountingMode)(
        device, mode)


cdef nvmlReturn_t _nvmlDeviceGetAccountingStats(nvmlDevice_t device, unsigned int pid, nvmlAccountingStats_t* stats) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetAccountingStats
    _check_or_init_nvml()
    if __nvmlDeviceGetAccountingStats == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetAccountingStats is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlAccountingStats_t*) noexcept nogil>__nvmlDeviceGetAccountingStats)(
        device, pid, stats)


cdef nvmlReturn_t _nvmlDeviceGetAccountingPids(nvmlDevice_t device, unsigned int* count, unsigned int* pids) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetAccountingPids
    _check_or_init_nvml()
    if __nvmlDeviceGetAccountingPids == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetAccountingPids is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, unsigned int*) noexcept nogil>__nvmlDeviceGetAccountingPids)(
        device, count, pids)


cdef nvmlReturn_t _nvmlDeviceGetAccountingBufferSize(nvmlDevice_t device, unsigned int* bufferSize) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetAccountingBufferSize
    _check_or_init_nvml()
    if __nvmlDeviceGetAccountingBufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetAccountingBufferSize is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetAccountingBufferSize)(
        device, bufferSize)


cdef nvmlReturn_t _nvmlDeviceGetRetiredPages(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int* pageCount, unsigned long long* addresses) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetRetiredPages
    _check_or_init_nvml()
    if __nvmlDeviceGetRetiredPages == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetRetiredPages is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlPageRetirementCause_t, unsigned int*, unsigned long long*) noexcept nogil>__nvmlDeviceGetRetiredPages)(
        device, cause, pageCount, addresses)


cdef nvmlReturn_t _nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int* pageCount, unsigned long long* addresses, unsigned long long* timestamps) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetRetiredPages_v2
    _check_or_init_nvml()
    if __nvmlDeviceGetRetiredPages_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetRetiredPages_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlPageRetirementCause_t, unsigned int*, unsigned long long*, unsigned long long*) noexcept nogil>__nvmlDeviceGetRetiredPages_v2)(
        device, cause, pageCount, addresses, timestamps)


cdef nvmlReturn_t _nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device, nvmlEnableState_t* isPending) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetRetiredPagesPendingStatus
    _check_or_init_nvml()
    if __nvmlDeviceGetRetiredPagesPendingStatus == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetRetiredPagesPendingStatus is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t*) noexcept nogil>__nvmlDeviceGetRetiredPagesPendingStatus)(
        device, isPending)


cdef nvmlReturn_t _nvmlDeviceGetRemappedRows(nvmlDevice_t device, unsigned int* corrRows, unsigned int* uncRows, unsigned int* isPending, unsigned int* failureOccurred) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetRemappedRows
    _check_or_init_nvml()
    if __nvmlDeviceGetRemappedRows == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetRemappedRows is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, unsigned int*, unsigned int*, unsigned int*) noexcept nogil>__nvmlDeviceGetRemappedRows)(
        device, corrRows, uncRows, isPending, failureOccurred)


cdef nvmlReturn_t _nvmlDeviceGetRowRemapperHistogram(nvmlDevice_t device, nvmlRowRemapperHistogramValues_t* values) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetRowRemapperHistogram
    _check_or_init_nvml()
    if __nvmlDeviceGetRowRemapperHistogram == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetRowRemapperHistogram is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlRowRemapperHistogramValues_t*) noexcept nogil>__nvmlDeviceGetRowRemapperHistogram)(
        device, values)


cdef nvmlReturn_t _nvmlDeviceGetArchitecture(nvmlDevice_t device, nvmlDeviceArchitecture_t* arch) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetArchitecture
    _check_or_init_nvml()
    if __nvmlDeviceGetArchitecture == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetArchitecture is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlDeviceArchitecture_t*) noexcept nogil>__nvmlDeviceGetArchitecture)(
        device, arch)


cdef nvmlReturn_t _nvmlDeviceGetClkMonStatus(nvmlDevice_t device, nvmlClkMonStatus_t* status) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetClkMonStatus
    _check_or_init_nvml()
    if __nvmlDeviceGetClkMonStatus == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetClkMonStatus is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlClkMonStatus_t*) noexcept nogil>__nvmlDeviceGetClkMonStatus)(
        device, status)


cdef nvmlReturn_t _nvmlDeviceGetProcessUtilization(nvmlDevice_t device, nvmlProcessUtilizationSample_t* utilization, unsigned int* processSamplesCount, unsigned long long lastSeenTimeStamp) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetProcessUtilization
    _check_or_init_nvml()
    if __nvmlDeviceGetProcessUtilization == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetProcessUtilization is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlProcessUtilizationSample_t*, unsigned int*, unsigned long long) noexcept nogil>__nvmlDeviceGetProcessUtilization)(
        device, utilization, processSamplesCount, lastSeenTimeStamp)


cdef nvmlReturn_t _nvmlDeviceGetProcessesUtilizationInfo(nvmlDevice_t device, nvmlProcessesUtilizationInfo_t* procesesUtilInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetProcessesUtilizationInfo
    _check_or_init_nvml()
    if __nvmlDeviceGetProcessesUtilizationInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetProcessesUtilizationInfo is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlProcessesUtilizationInfo_t*) noexcept nogil>__nvmlDeviceGetProcessesUtilizationInfo)(
        device, procesesUtilInfo)


cdef nvmlReturn_t _nvmlDeviceGetPlatformInfo(nvmlDevice_t device, nvmlPlatformInfo_t* platformInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetPlatformInfo
    _check_or_init_nvml()
    if __nvmlDeviceGetPlatformInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetPlatformInfo is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlPlatformInfo_t*) noexcept nogil>__nvmlDeviceGetPlatformInfo)(
        device, platformInfo)


cdef nvmlReturn_t _nvmlUnitSetLedState(nvmlUnit_t unit, nvmlLedColor_t color) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlUnitSetLedState
    _check_or_init_nvml()
    if __nvmlUnitSetLedState == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlUnitSetLedState is not found")
    return (<nvmlReturn_t (*)(nvmlUnit_t, nvmlLedColor_t) noexcept nogil>__nvmlUnitSetLedState)(
        unit, color)


cdef nvmlReturn_t _nvmlDeviceSetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t mode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetPersistenceMode
    _check_or_init_nvml()
    if __nvmlDeviceSetPersistenceMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetPersistenceMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t) noexcept nogil>__nvmlDeviceSetPersistenceMode)(
        device, mode)


cdef nvmlReturn_t _nvmlDeviceSetComputeMode(nvmlDevice_t device, nvmlComputeMode_t mode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetComputeMode
    _check_or_init_nvml()
    if __nvmlDeviceSetComputeMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetComputeMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlComputeMode_t) noexcept nogil>__nvmlDeviceSetComputeMode)(
        device, mode)


cdef nvmlReturn_t _nvmlDeviceSetEccMode(nvmlDevice_t device, nvmlEnableState_t ecc) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetEccMode
    _check_or_init_nvml()
    if __nvmlDeviceSetEccMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetEccMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t) noexcept nogil>__nvmlDeviceSetEccMode)(
        device, ecc)


cdef nvmlReturn_t _nvmlDeviceClearEccErrorCounts(nvmlDevice_t device, nvmlEccCounterType_t counterType) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceClearEccErrorCounts
    _check_or_init_nvml()
    if __nvmlDeviceClearEccErrorCounts == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceClearEccErrorCounts is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlEccCounterType_t) noexcept nogil>__nvmlDeviceClearEccErrorCounts)(
        device, counterType)


cdef nvmlReturn_t _nvmlDeviceSetDriverModel(nvmlDevice_t device, nvmlDriverModel_t driverModel, unsigned int flags) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetDriverModel
    _check_or_init_nvml()
    if __nvmlDeviceSetDriverModel == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetDriverModel is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlDriverModel_t, unsigned int) noexcept nogil>__nvmlDeviceSetDriverModel)(
        device, driverModel, flags)


cdef nvmlReturn_t _nvmlDeviceSetGpuLockedClocks(nvmlDevice_t device, unsigned int minGpuClockMHz, unsigned int maxGpuClockMHz) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetGpuLockedClocks
    _check_or_init_nvml()
    if __nvmlDeviceSetGpuLockedClocks == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetGpuLockedClocks is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int) noexcept nogil>__nvmlDeviceSetGpuLockedClocks)(
        device, minGpuClockMHz, maxGpuClockMHz)


cdef nvmlReturn_t _nvmlDeviceResetGpuLockedClocks(nvmlDevice_t device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceResetGpuLockedClocks
    _check_or_init_nvml()
    if __nvmlDeviceResetGpuLockedClocks == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceResetGpuLockedClocks is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t) noexcept nogil>__nvmlDeviceResetGpuLockedClocks)(
        device)


cdef nvmlReturn_t _nvmlDeviceSetMemoryLockedClocks(nvmlDevice_t device, unsigned int minMemClockMHz, unsigned int maxMemClockMHz) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetMemoryLockedClocks
    _check_or_init_nvml()
    if __nvmlDeviceSetMemoryLockedClocks == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetMemoryLockedClocks is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int) noexcept nogil>__nvmlDeviceSetMemoryLockedClocks)(
        device, minMemClockMHz, maxMemClockMHz)


cdef nvmlReturn_t _nvmlDeviceResetMemoryLockedClocks(nvmlDevice_t device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceResetMemoryLockedClocks
    _check_or_init_nvml()
    if __nvmlDeviceResetMemoryLockedClocks == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceResetMemoryLockedClocks is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t) noexcept nogil>__nvmlDeviceResetMemoryLockedClocks)(
        device)


cdef nvmlReturn_t _nvmlDeviceSetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetAutoBoostedClocksEnabled
    _check_or_init_nvml()
    if __nvmlDeviceSetAutoBoostedClocksEnabled == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetAutoBoostedClocksEnabled is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t) noexcept nogil>__nvmlDeviceSetAutoBoostedClocksEnabled)(
        device, enabled)


cdef nvmlReturn_t _nvmlDeviceSetDefaultAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled, unsigned int flags) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetDefaultAutoBoostedClocksEnabled
    _check_or_init_nvml()
    if __nvmlDeviceSetDefaultAutoBoostedClocksEnabled == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetDefaultAutoBoostedClocksEnabled is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t, unsigned int) noexcept nogil>__nvmlDeviceSetDefaultAutoBoostedClocksEnabled)(
        device, enabled, flags)


cdef nvmlReturn_t _nvmlDeviceSetDefaultFanSpeed_v2(nvmlDevice_t device, unsigned int fan) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetDefaultFanSpeed_v2
    _check_or_init_nvml()
    if __nvmlDeviceSetDefaultFanSpeed_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetDefaultFanSpeed_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int) noexcept nogil>__nvmlDeviceSetDefaultFanSpeed_v2)(
        device, fan)


cdef nvmlReturn_t _nvmlDeviceSetFanControlPolicy(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t policy) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetFanControlPolicy
    _check_or_init_nvml()
    if __nvmlDeviceSetFanControlPolicy == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetFanControlPolicy is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlFanControlPolicy_t) noexcept nogil>__nvmlDeviceSetFanControlPolicy)(
        device, fan, policy)


cdef nvmlReturn_t _nvmlDeviceSetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, int* temp) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetTemperatureThreshold
    _check_or_init_nvml()
    if __nvmlDeviceSetTemperatureThreshold == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetTemperatureThreshold is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlTemperatureThresholds_t, int*) noexcept nogil>__nvmlDeviceSetTemperatureThreshold)(
        device, thresholdType, temp)


cdef nvmlReturn_t _nvmlDeviceSetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t mode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetGpuOperationMode
    _check_or_init_nvml()
    if __nvmlDeviceSetGpuOperationMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetGpuOperationMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlGpuOperationMode_t) noexcept nogil>__nvmlDeviceSetGpuOperationMode)(
        device, mode)


cdef nvmlReturn_t _nvmlDeviceSetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t isRestricted) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetAPIRestriction
    _check_or_init_nvml()
    if __nvmlDeviceSetAPIRestriction == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetAPIRestriction is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlRestrictedAPI_t, nvmlEnableState_t) noexcept nogil>__nvmlDeviceSetAPIRestriction)(
        device, apiType, isRestricted)


cdef nvmlReturn_t _nvmlDeviceSetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int speed) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetFanSpeed_v2
    _check_or_init_nvml()
    if __nvmlDeviceSetFanSpeed_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetFanSpeed_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int) noexcept nogil>__nvmlDeviceSetFanSpeed_v2)(
        device, fan, speed)


cdef nvmlReturn_t _nvmlDeviceSetAccountingMode(nvmlDevice_t device, nvmlEnableState_t mode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetAccountingMode
    _check_or_init_nvml()
    if __nvmlDeviceSetAccountingMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetAccountingMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t) noexcept nogil>__nvmlDeviceSetAccountingMode)(
        device, mode)


cdef nvmlReturn_t _nvmlDeviceClearAccountingPids(nvmlDevice_t device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceClearAccountingPids
    _check_or_init_nvml()
    if __nvmlDeviceClearAccountingPids == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceClearAccountingPids is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t) noexcept nogil>__nvmlDeviceClearAccountingPids)(
        device)


cdef nvmlReturn_t _nvmlDeviceSetPowerManagementLimit_v2(nvmlDevice_t device, nvmlPowerValue_v2_t* powerValue) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetPowerManagementLimit_v2
    _check_or_init_nvml()
    if __nvmlDeviceSetPowerManagementLimit_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetPowerManagementLimit_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlPowerValue_v2_t*) noexcept nogil>__nvmlDeviceSetPowerManagementLimit_v2)(
        device, powerValue)


cdef nvmlReturn_t _nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t* isActive) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetNvLinkState
    _check_or_init_nvml()
    if __nvmlDeviceGetNvLinkState == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetNvLinkState is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlEnableState_t*) noexcept nogil>__nvmlDeviceGetNvLinkState)(
        device, link, isActive)


cdef nvmlReturn_t _nvmlDeviceGetNvLinkVersion(nvmlDevice_t device, unsigned int link, unsigned int* version) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetNvLinkVersion
    _check_or_init_nvml()
    if __nvmlDeviceGetNvLinkVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetNvLinkVersion is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int*) noexcept nogil>__nvmlDeviceGetNvLinkVersion)(
        device, link, version)


cdef nvmlReturn_t _nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int* capResult) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetNvLinkCapability
    _check_or_init_nvml()
    if __nvmlDeviceGetNvLinkCapability == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetNvLinkCapability is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlNvLinkCapability_t, unsigned int*) noexcept nogil>__nvmlDeviceGetNvLinkCapability)(
        device, link, capability, capResult)


cdef nvmlReturn_t _nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t* pci) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetNvLinkRemotePciInfo_v2
    _check_or_init_nvml()
    if __nvmlDeviceGetNvLinkRemotePciInfo_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetNvLinkRemotePciInfo_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlPciInfo_t*) noexcept nogil>__nvmlDeviceGetNvLinkRemotePciInfo_v2)(
        device, link, pci)


cdef nvmlReturn_t _nvmlDeviceGetNvLinkErrorCounter(nvmlDevice_t device, unsigned int link, nvmlNvLinkErrorCounter_t counter, unsigned long long* counterValue) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetNvLinkErrorCounter
    _check_or_init_nvml()
    if __nvmlDeviceGetNvLinkErrorCounter == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetNvLinkErrorCounter is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlNvLinkErrorCounter_t, unsigned long long*) noexcept nogil>__nvmlDeviceGetNvLinkErrorCounter)(
        device, link, counter, counterValue)


cdef nvmlReturn_t _nvmlDeviceResetNvLinkErrorCounters(nvmlDevice_t device, unsigned int link) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceResetNvLinkErrorCounters
    _check_or_init_nvml()
    if __nvmlDeviceResetNvLinkErrorCounters == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceResetNvLinkErrorCounters is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int) noexcept nogil>__nvmlDeviceResetNvLinkErrorCounters)(
        device, link)


cdef nvmlReturn_t _nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link, nvmlIntNvLinkDeviceType_t* pNvLinkDeviceType) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetNvLinkRemoteDeviceType
    _check_or_init_nvml()
    if __nvmlDeviceGetNvLinkRemoteDeviceType == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetNvLinkRemoteDeviceType is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlIntNvLinkDeviceType_t*) noexcept nogil>__nvmlDeviceGetNvLinkRemoteDeviceType)(
        device, link, pNvLinkDeviceType)


cdef nvmlReturn_t _nvmlDeviceSetNvLinkDeviceLowPowerThreshold(nvmlDevice_t device, nvmlNvLinkPowerThres_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetNvLinkDeviceLowPowerThreshold
    _check_or_init_nvml()
    if __nvmlDeviceSetNvLinkDeviceLowPowerThreshold == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetNvLinkDeviceLowPowerThreshold is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlNvLinkPowerThres_t*) noexcept nogil>__nvmlDeviceSetNvLinkDeviceLowPowerThreshold)(
        device, info)


cdef nvmlReturn_t _nvmlSystemSetNvlinkBwMode(unsigned int nvlinkBwMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemSetNvlinkBwMode
    _check_or_init_nvml()
    if __nvmlSystemSetNvlinkBwMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemSetNvlinkBwMode is not found")
    return (<nvmlReturn_t (*)(unsigned int) noexcept nogil>__nvmlSystemSetNvlinkBwMode)(
        nvlinkBwMode)


cdef nvmlReturn_t _nvmlSystemGetNvlinkBwMode(unsigned int* nvlinkBwMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemGetNvlinkBwMode
    _check_or_init_nvml()
    if __nvmlSystemGetNvlinkBwMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemGetNvlinkBwMode is not found")
    return (<nvmlReturn_t (*)(unsigned int*) noexcept nogil>__nvmlSystemGetNvlinkBwMode)(
        nvlinkBwMode)


cdef nvmlReturn_t _nvmlDeviceGetNvlinkSupportedBwModes(nvmlDevice_t device, nvmlNvlinkSupportedBwModes_t* supportedBwMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetNvlinkSupportedBwModes
    _check_or_init_nvml()
    if __nvmlDeviceGetNvlinkSupportedBwModes == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetNvlinkSupportedBwModes is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlNvlinkSupportedBwModes_t*) noexcept nogil>__nvmlDeviceGetNvlinkSupportedBwModes)(
        device, supportedBwMode)


cdef nvmlReturn_t _nvmlDeviceGetNvlinkBwMode(nvmlDevice_t device, nvmlNvlinkGetBwMode_t* getBwMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetNvlinkBwMode
    _check_or_init_nvml()
    if __nvmlDeviceGetNvlinkBwMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetNvlinkBwMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlNvlinkGetBwMode_t*) noexcept nogil>__nvmlDeviceGetNvlinkBwMode)(
        device, getBwMode)


cdef nvmlReturn_t _nvmlDeviceSetNvlinkBwMode(nvmlDevice_t device, nvmlNvlinkSetBwMode_t* setBwMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetNvlinkBwMode
    _check_or_init_nvml()
    if __nvmlDeviceSetNvlinkBwMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetNvlinkBwMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlNvlinkSetBwMode_t*) noexcept nogil>__nvmlDeviceSetNvlinkBwMode)(
        device, setBwMode)


cdef nvmlReturn_t _nvmlEventSetCreate(nvmlEventSet_t* set) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlEventSetCreate
    _check_or_init_nvml()
    if __nvmlEventSetCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlEventSetCreate is not found")
    return (<nvmlReturn_t (*)(nvmlEventSet_t*) noexcept nogil>__nvmlEventSetCreate)(
        set)


cdef nvmlReturn_t _nvmlDeviceRegisterEvents(nvmlDevice_t device, unsigned long long eventTypes, nvmlEventSet_t set) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceRegisterEvents
    _check_or_init_nvml()
    if __nvmlDeviceRegisterEvents == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceRegisterEvents is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned long long, nvmlEventSet_t) noexcept nogil>__nvmlDeviceRegisterEvents)(
        device, eventTypes, set)


cdef nvmlReturn_t _nvmlDeviceGetSupportedEventTypes(nvmlDevice_t device, unsigned long long* eventTypes) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetSupportedEventTypes
    _check_or_init_nvml()
    if __nvmlDeviceGetSupportedEventTypes == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetSupportedEventTypes is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned long long*) noexcept nogil>__nvmlDeviceGetSupportedEventTypes)(
        device, eventTypes)


cdef nvmlReturn_t _nvmlEventSetWait_v2(nvmlEventSet_t set, nvmlEventData_t* data, unsigned int timeoutms) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlEventSetWait_v2
    _check_or_init_nvml()
    if __nvmlEventSetWait_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlEventSetWait_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlEventSet_t, nvmlEventData_t*, unsigned int) noexcept nogil>__nvmlEventSetWait_v2)(
        set, data, timeoutms)


cdef nvmlReturn_t _nvmlEventSetFree(nvmlEventSet_t set) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlEventSetFree
    _check_or_init_nvml()
    if __nvmlEventSetFree == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlEventSetFree is not found")
    return (<nvmlReturn_t (*)(nvmlEventSet_t) noexcept nogil>__nvmlEventSetFree)(
        set)


cdef nvmlReturn_t _nvmlSystemEventSetCreate(nvmlSystemEventSetCreateRequest_t* request) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemEventSetCreate
    _check_or_init_nvml()
    if __nvmlSystemEventSetCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemEventSetCreate is not found")
    return (<nvmlReturn_t (*)(nvmlSystemEventSetCreateRequest_t*) noexcept nogil>__nvmlSystemEventSetCreate)(
        request)


cdef nvmlReturn_t _nvmlSystemEventSetFree(nvmlSystemEventSetFreeRequest_t* request) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemEventSetFree
    _check_or_init_nvml()
    if __nvmlSystemEventSetFree == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemEventSetFree is not found")
    return (<nvmlReturn_t (*)(nvmlSystemEventSetFreeRequest_t*) noexcept nogil>__nvmlSystemEventSetFree)(
        request)


cdef nvmlReturn_t _nvmlSystemRegisterEvents(nvmlSystemRegisterEventRequest_t* request) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemRegisterEvents
    _check_or_init_nvml()
    if __nvmlSystemRegisterEvents == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemRegisterEvents is not found")
    return (<nvmlReturn_t (*)(nvmlSystemRegisterEventRequest_t*) noexcept nogil>__nvmlSystemRegisterEvents)(
        request)


cdef nvmlReturn_t _nvmlSystemEventSetWait(nvmlSystemEventSetWaitRequest_t* request) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemEventSetWait
    _check_or_init_nvml()
    if __nvmlSystemEventSetWait == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemEventSetWait is not found")
    return (<nvmlReturn_t (*)(nvmlSystemEventSetWaitRequest_t*) noexcept nogil>__nvmlSystemEventSetWait)(
        request)


cdef nvmlReturn_t _nvmlDeviceModifyDrainState(nvmlPciInfo_t* pciInfo, nvmlEnableState_t newState) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceModifyDrainState
    _check_or_init_nvml()
    if __nvmlDeviceModifyDrainState == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceModifyDrainState is not found")
    return (<nvmlReturn_t (*)(nvmlPciInfo_t*, nvmlEnableState_t) noexcept nogil>__nvmlDeviceModifyDrainState)(
        pciInfo, newState)


cdef nvmlReturn_t _nvmlDeviceQueryDrainState(nvmlPciInfo_t* pciInfo, nvmlEnableState_t* currentState) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceQueryDrainState
    _check_or_init_nvml()
    if __nvmlDeviceQueryDrainState == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceQueryDrainState is not found")
    return (<nvmlReturn_t (*)(nvmlPciInfo_t*, nvmlEnableState_t*) noexcept nogil>__nvmlDeviceQueryDrainState)(
        pciInfo, currentState)


cdef nvmlReturn_t _nvmlDeviceRemoveGpu_v2(nvmlPciInfo_t* pciInfo, nvmlDetachGpuState_t gpuState, nvmlPcieLinkState_t linkState) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceRemoveGpu_v2
    _check_or_init_nvml()
    if __nvmlDeviceRemoveGpu_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceRemoveGpu_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlPciInfo_t*, nvmlDetachGpuState_t, nvmlPcieLinkState_t) noexcept nogil>__nvmlDeviceRemoveGpu_v2)(
        pciInfo, gpuState, linkState)


cdef nvmlReturn_t _nvmlDeviceDiscoverGpus(nvmlPciInfo_t* pciInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceDiscoverGpus
    _check_or_init_nvml()
    if __nvmlDeviceDiscoverGpus == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceDiscoverGpus is not found")
    return (<nvmlReturn_t (*)(nvmlPciInfo_t*) noexcept nogil>__nvmlDeviceDiscoverGpus)(
        pciInfo)


cdef nvmlReturn_t _nvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetFieldValues
    _check_or_init_nvml()
    if __nvmlDeviceGetFieldValues == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetFieldValues is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, int, nvmlFieldValue_t*) noexcept nogil>__nvmlDeviceGetFieldValues)(
        device, valuesCount, values)


cdef nvmlReturn_t _nvmlDeviceClearFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceClearFieldValues
    _check_or_init_nvml()
    if __nvmlDeviceClearFieldValues == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceClearFieldValues is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, int, nvmlFieldValue_t*) noexcept nogil>__nvmlDeviceClearFieldValues)(
        device, valuesCount, values)


cdef nvmlReturn_t _nvmlDeviceGetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t* pVirtualMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetVirtualizationMode
    _check_or_init_nvml()
    if __nvmlDeviceGetVirtualizationMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetVirtualizationMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlGpuVirtualizationMode_t*) noexcept nogil>__nvmlDeviceGetVirtualizationMode)(
        device, pVirtualMode)


cdef nvmlReturn_t _nvmlDeviceGetHostVgpuMode(nvmlDevice_t device, nvmlHostVgpuMode_t* pHostVgpuMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetHostVgpuMode
    _check_or_init_nvml()
    if __nvmlDeviceGetHostVgpuMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetHostVgpuMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlHostVgpuMode_t*) noexcept nogil>__nvmlDeviceGetHostVgpuMode)(
        device, pHostVgpuMode)


cdef nvmlReturn_t _nvmlDeviceSetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t virtualMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetVirtualizationMode
    _check_or_init_nvml()
    if __nvmlDeviceSetVirtualizationMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetVirtualizationMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlGpuVirtualizationMode_t) noexcept nogil>__nvmlDeviceSetVirtualizationMode)(
        device, virtualMode)


cdef nvmlReturn_t _nvmlDeviceGetVgpuHeterogeneousMode(nvmlDevice_t device, nvmlVgpuHeterogeneousMode_t* pHeterogeneousMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetVgpuHeterogeneousMode
    _check_or_init_nvml()
    if __nvmlDeviceGetVgpuHeterogeneousMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetVgpuHeterogeneousMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlVgpuHeterogeneousMode_t*) noexcept nogil>__nvmlDeviceGetVgpuHeterogeneousMode)(
        device, pHeterogeneousMode)


cdef nvmlReturn_t _nvmlDeviceSetVgpuHeterogeneousMode(nvmlDevice_t device, const nvmlVgpuHeterogeneousMode_t* pHeterogeneousMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetVgpuHeterogeneousMode
    _check_or_init_nvml()
    if __nvmlDeviceSetVgpuHeterogeneousMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetVgpuHeterogeneousMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, const nvmlVgpuHeterogeneousMode_t*) noexcept nogil>__nvmlDeviceSetVgpuHeterogeneousMode)(
        device, pHeterogeneousMode)


cdef nvmlReturn_t _nvmlVgpuInstanceGetPlacementId(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuPlacementId_t* pPlacement) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetPlacementId
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetPlacementId == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetPlacementId is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, nvmlVgpuPlacementId_t*) noexcept nogil>__nvmlVgpuInstanceGetPlacementId)(
        vgpuInstance, pPlacement)


cdef nvmlReturn_t _nvmlDeviceGetVgpuTypeSupportedPlacements(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuPlacementList_t* pPlacementList) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetVgpuTypeSupportedPlacements
    _check_or_init_nvml()
    if __nvmlDeviceGetVgpuTypeSupportedPlacements == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetVgpuTypeSupportedPlacements is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlVgpuTypeId_t, nvmlVgpuPlacementList_t*) noexcept nogil>__nvmlDeviceGetVgpuTypeSupportedPlacements)(
        device, vgpuTypeId, pPlacementList)


cdef nvmlReturn_t _nvmlDeviceGetVgpuTypeCreatablePlacements(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuPlacementList_t* pPlacementList) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetVgpuTypeCreatablePlacements
    _check_or_init_nvml()
    if __nvmlDeviceGetVgpuTypeCreatablePlacements == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetVgpuTypeCreatablePlacements is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlVgpuTypeId_t, nvmlVgpuPlacementList_t*) noexcept nogil>__nvmlDeviceGetVgpuTypeCreatablePlacements)(
        device, vgpuTypeId, pPlacementList)


cdef nvmlReturn_t _nvmlVgpuTypeGetGspHeapSize(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* gspHeapSize) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuTypeGetGspHeapSize
    _check_or_init_nvml()
    if __nvmlVgpuTypeGetGspHeapSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuTypeGetGspHeapSize is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuTypeId_t, unsigned long long*) noexcept nogil>__nvmlVgpuTypeGetGspHeapSize)(
        vgpuTypeId, gspHeapSize)


cdef nvmlReturn_t _nvmlVgpuTypeGetFbReservation(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* fbReservation) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuTypeGetFbReservation
    _check_or_init_nvml()
    if __nvmlVgpuTypeGetFbReservation == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuTypeGetFbReservation is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuTypeId_t, unsigned long long*) noexcept nogil>__nvmlVgpuTypeGetFbReservation)(
        vgpuTypeId, fbReservation)


cdef nvmlReturn_t _nvmlVgpuInstanceGetRuntimeStateSize(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuRuntimeState_t* pState) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetRuntimeStateSize
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetRuntimeStateSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetRuntimeStateSize is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, nvmlVgpuRuntimeState_t*) noexcept nogil>__nvmlVgpuInstanceGetRuntimeStateSize)(
        vgpuInstance, pState)


cdef nvmlReturn_t _nvmlDeviceSetVgpuCapabilities(nvmlDevice_t device, nvmlDeviceVgpuCapability_t capability, nvmlEnableState_t state) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetVgpuCapabilities
    _check_or_init_nvml()
    if __nvmlDeviceSetVgpuCapabilities == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetVgpuCapabilities is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlDeviceVgpuCapability_t, nvmlEnableState_t) noexcept nogil>__nvmlDeviceSetVgpuCapabilities)(
        device, capability, state)


cdef nvmlReturn_t _nvmlDeviceGetGridLicensableFeatures_v4(nvmlDevice_t device, nvmlGridLicensableFeatures_t* pGridLicensableFeatures) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetGridLicensableFeatures_v4
    _check_or_init_nvml()
    if __nvmlDeviceGetGridLicensableFeatures_v4 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetGridLicensableFeatures_v4 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlGridLicensableFeatures_t*) noexcept nogil>__nvmlDeviceGetGridLicensableFeatures_v4)(
        device, pGridLicensableFeatures)


cdef nvmlReturn_t _nvmlGetVgpuDriverCapabilities(nvmlVgpuDriverCapability_t capability, unsigned int* capResult) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGetVgpuDriverCapabilities
    _check_or_init_nvml()
    if __nvmlGetVgpuDriverCapabilities == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGetVgpuDriverCapabilities is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuDriverCapability_t, unsigned int*) noexcept nogil>__nvmlGetVgpuDriverCapabilities)(
        capability, capResult)


cdef nvmlReturn_t _nvmlDeviceGetVgpuCapabilities(nvmlDevice_t device, nvmlDeviceVgpuCapability_t capability, unsigned int* capResult) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetVgpuCapabilities
    _check_or_init_nvml()
    if __nvmlDeviceGetVgpuCapabilities == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetVgpuCapabilities is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlDeviceVgpuCapability_t, unsigned int*) noexcept nogil>__nvmlDeviceGetVgpuCapabilities)(
        device, capability, capResult)


cdef nvmlReturn_t _nvmlDeviceGetSupportedVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuTypeId_t* vgpuTypeIds) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetSupportedVgpus
    _check_or_init_nvml()
    if __nvmlDeviceGetSupportedVgpus == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetSupportedVgpus is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, nvmlVgpuTypeId_t*) noexcept nogil>__nvmlDeviceGetSupportedVgpus)(
        device, vgpuCount, vgpuTypeIds)


cdef nvmlReturn_t _nvmlDeviceGetCreatableVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuTypeId_t* vgpuTypeIds) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetCreatableVgpus
    _check_or_init_nvml()
    if __nvmlDeviceGetCreatableVgpus == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetCreatableVgpus is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, nvmlVgpuTypeId_t*) noexcept nogil>__nvmlDeviceGetCreatableVgpus)(
        device, vgpuCount, vgpuTypeIds)


cdef nvmlReturn_t _nvmlVgpuTypeGetClass(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeClass, unsigned int* size) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuTypeGetClass
    _check_or_init_nvml()
    if __nvmlVgpuTypeGetClass == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuTypeGetClass is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuTypeId_t, char*, unsigned int*) noexcept nogil>__nvmlVgpuTypeGetClass)(
        vgpuTypeId, vgpuTypeClass, size)


cdef nvmlReturn_t _nvmlVgpuTypeGetName(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeName, unsigned int* size) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuTypeGetName
    _check_or_init_nvml()
    if __nvmlVgpuTypeGetName == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuTypeGetName is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuTypeId_t, char*, unsigned int*) noexcept nogil>__nvmlVgpuTypeGetName)(
        vgpuTypeId, vgpuTypeName, size)


cdef nvmlReturn_t _nvmlVgpuTypeGetGpuInstanceProfileId(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* gpuInstanceProfileId) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuTypeGetGpuInstanceProfileId
    _check_or_init_nvml()
    if __nvmlVgpuTypeGetGpuInstanceProfileId == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuTypeGetGpuInstanceProfileId is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuTypeId_t, unsigned int*) noexcept nogil>__nvmlVgpuTypeGetGpuInstanceProfileId)(
        vgpuTypeId, gpuInstanceProfileId)


cdef nvmlReturn_t _nvmlVgpuTypeGetDeviceID(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* deviceID, unsigned long long* subsystemID) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuTypeGetDeviceID
    _check_or_init_nvml()
    if __nvmlVgpuTypeGetDeviceID == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuTypeGetDeviceID is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuTypeId_t, unsigned long long*, unsigned long long*) noexcept nogil>__nvmlVgpuTypeGetDeviceID)(
        vgpuTypeId, deviceID, subsystemID)


cdef nvmlReturn_t _nvmlVgpuTypeGetFramebufferSize(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* fbSize) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuTypeGetFramebufferSize
    _check_or_init_nvml()
    if __nvmlVgpuTypeGetFramebufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuTypeGetFramebufferSize is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuTypeId_t, unsigned long long*) noexcept nogil>__nvmlVgpuTypeGetFramebufferSize)(
        vgpuTypeId, fbSize)


cdef nvmlReturn_t _nvmlVgpuTypeGetNumDisplayHeads(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* numDisplayHeads) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuTypeGetNumDisplayHeads
    _check_or_init_nvml()
    if __nvmlVgpuTypeGetNumDisplayHeads == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuTypeGetNumDisplayHeads is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuTypeId_t, unsigned int*) noexcept nogil>__nvmlVgpuTypeGetNumDisplayHeads)(
        vgpuTypeId, numDisplayHeads)


cdef nvmlReturn_t _nvmlVgpuTypeGetResolution(nvmlVgpuTypeId_t vgpuTypeId, unsigned int displayIndex, unsigned int* xdim, unsigned int* ydim) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuTypeGetResolution
    _check_or_init_nvml()
    if __nvmlVgpuTypeGetResolution == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuTypeGetResolution is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuTypeId_t, unsigned int, unsigned int*, unsigned int*) noexcept nogil>__nvmlVgpuTypeGetResolution)(
        vgpuTypeId, displayIndex, xdim, ydim)


cdef nvmlReturn_t _nvmlVgpuTypeGetLicense(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeLicenseString, unsigned int size) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuTypeGetLicense
    _check_or_init_nvml()
    if __nvmlVgpuTypeGetLicense == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuTypeGetLicense is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuTypeId_t, char*, unsigned int) noexcept nogil>__nvmlVgpuTypeGetLicense)(
        vgpuTypeId, vgpuTypeLicenseString, size)


cdef nvmlReturn_t _nvmlVgpuTypeGetFrameRateLimit(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* frameRateLimit) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuTypeGetFrameRateLimit
    _check_or_init_nvml()
    if __nvmlVgpuTypeGetFrameRateLimit == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuTypeGetFrameRateLimit is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuTypeId_t, unsigned int*) noexcept nogil>__nvmlVgpuTypeGetFrameRateLimit)(
        vgpuTypeId, frameRateLimit)


cdef nvmlReturn_t _nvmlVgpuTypeGetMaxInstances(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, unsigned int* vgpuInstanceCount) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuTypeGetMaxInstances
    _check_or_init_nvml()
    if __nvmlVgpuTypeGetMaxInstances == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuTypeGetMaxInstances is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlVgpuTypeId_t, unsigned int*) noexcept nogil>__nvmlVgpuTypeGetMaxInstances)(
        device, vgpuTypeId, vgpuInstanceCount)


cdef nvmlReturn_t _nvmlVgpuTypeGetMaxInstancesPerVm(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* vgpuInstanceCountPerVm) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuTypeGetMaxInstancesPerVm
    _check_or_init_nvml()
    if __nvmlVgpuTypeGetMaxInstancesPerVm == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuTypeGetMaxInstancesPerVm is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuTypeId_t, unsigned int*) noexcept nogil>__nvmlVgpuTypeGetMaxInstancesPerVm)(
        vgpuTypeId, vgpuInstanceCountPerVm)


cdef nvmlReturn_t _nvmlVgpuTypeGetBAR1Info(nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuTypeBar1Info_t* bar1Info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuTypeGetBAR1Info
    _check_or_init_nvml()
    if __nvmlVgpuTypeGetBAR1Info == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuTypeGetBAR1Info is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuTypeId_t, nvmlVgpuTypeBar1Info_t*) noexcept nogil>__nvmlVgpuTypeGetBAR1Info)(
        vgpuTypeId, bar1Info)


cdef nvmlReturn_t _nvmlDeviceGetActiveVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuInstance_t* vgpuInstances) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetActiveVgpus
    _check_or_init_nvml()
    if __nvmlDeviceGetActiveVgpus == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetActiveVgpus is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, nvmlVgpuInstance_t*) noexcept nogil>__nvmlDeviceGetActiveVgpus)(
        device, vgpuCount, vgpuInstances)


cdef nvmlReturn_t _nvmlVgpuInstanceGetVmID(nvmlVgpuInstance_t vgpuInstance, char* vmId, unsigned int size, nvmlVgpuVmIdType_t* vmIdType) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetVmID
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetVmID == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetVmID is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, char*, unsigned int, nvmlVgpuVmIdType_t*) noexcept nogil>__nvmlVgpuInstanceGetVmID)(
        vgpuInstance, vmId, size, vmIdType)


cdef nvmlReturn_t _nvmlVgpuInstanceGetUUID(nvmlVgpuInstance_t vgpuInstance, char* uuid, unsigned int size) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetUUID
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetUUID == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetUUID is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, char*, unsigned int) noexcept nogil>__nvmlVgpuInstanceGetUUID)(
        vgpuInstance, uuid, size)


cdef nvmlReturn_t _nvmlVgpuInstanceGetVmDriverVersion(nvmlVgpuInstance_t vgpuInstance, char* version, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetVmDriverVersion
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetVmDriverVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetVmDriverVersion is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, char*, unsigned int) noexcept nogil>__nvmlVgpuInstanceGetVmDriverVersion)(
        vgpuInstance, version, length)


cdef nvmlReturn_t _nvmlVgpuInstanceGetFbUsage(nvmlVgpuInstance_t vgpuInstance, unsigned long long* fbUsage) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetFbUsage
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetFbUsage == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetFbUsage is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned long long*) noexcept nogil>__nvmlVgpuInstanceGetFbUsage)(
        vgpuInstance, fbUsage)


cdef nvmlReturn_t _nvmlVgpuInstanceGetLicenseStatus(nvmlVgpuInstance_t vgpuInstance, unsigned int* licensed) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetLicenseStatus
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetLicenseStatus == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetLicenseStatus is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned int*) noexcept nogil>__nvmlVgpuInstanceGetLicenseStatus)(
        vgpuInstance, licensed)


cdef nvmlReturn_t _nvmlVgpuInstanceGetType(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuTypeId_t* vgpuTypeId) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetType
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetType == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetType is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, nvmlVgpuTypeId_t*) noexcept nogil>__nvmlVgpuInstanceGetType)(
        vgpuInstance, vgpuTypeId)


cdef nvmlReturn_t _nvmlVgpuInstanceGetFrameRateLimit(nvmlVgpuInstance_t vgpuInstance, unsigned int* frameRateLimit) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetFrameRateLimit
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetFrameRateLimit == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetFrameRateLimit is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned int*) noexcept nogil>__nvmlVgpuInstanceGetFrameRateLimit)(
        vgpuInstance, frameRateLimit)


cdef nvmlReturn_t _nvmlVgpuInstanceGetEccMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t* eccMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetEccMode
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetEccMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetEccMode is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, nvmlEnableState_t*) noexcept nogil>__nvmlVgpuInstanceGetEccMode)(
        vgpuInstance, eccMode)


cdef nvmlReturn_t _nvmlVgpuInstanceGetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int* encoderCapacity) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetEncoderCapacity
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetEncoderCapacity == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetEncoderCapacity is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned int*) noexcept nogil>__nvmlVgpuInstanceGetEncoderCapacity)(
        vgpuInstance, encoderCapacity)


cdef nvmlReturn_t _nvmlVgpuInstanceSetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int encoderCapacity) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceSetEncoderCapacity
    _check_or_init_nvml()
    if __nvmlVgpuInstanceSetEncoderCapacity == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceSetEncoderCapacity is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned int) noexcept nogil>__nvmlVgpuInstanceSetEncoderCapacity)(
        vgpuInstance, encoderCapacity)


cdef nvmlReturn_t _nvmlVgpuInstanceGetEncoderStats(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, unsigned int* averageFps, unsigned int* averageLatency) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetEncoderStats
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetEncoderStats == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetEncoderStats is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned int*, unsigned int*, unsigned int*) noexcept nogil>__nvmlVgpuInstanceGetEncoderStats)(
        vgpuInstance, sessionCount, averageFps, averageLatency)


cdef nvmlReturn_t _nvmlVgpuInstanceGetEncoderSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, nvmlEncoderSessionInfo_t* sessionInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetEncoderSessions
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetEncoderSessions == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetEncoderSessions is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned int*, nvmlEncoderSessionInfo_t*) noexcept nogil>__nvmlVgpuInstanceGetEncoderSessions)(
        vgpuInstance, sessionCount, sessionInfo)


cdef nvmlReturn_t _nvmlVgpuInstanceGetFBCStats(nvmlVgpuInstance_t vgpuInstance, nvmlFBCStats_t* fbcStats) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetFBCStats
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetFBCStats == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetFBCStats is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, nvmlFBCStats_t*) noexcept nogil>__nvmlVgpuInstanceGetFBCStats)(
        vgpuInstance, fbcStats)


cdef nvmlReturn_t _nvmlVgpuInstanceGetFBCSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, nvmlFBCSessionInfo_t* sessionInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetFBCSessions
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetFBCSessions == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetFBCSessions is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned int*, nvmlFBCSessionInfo_t*) noexcept nogil>__nvmlVgpuInstanceGetFBCSessions)(
        vgpuInstance, sessionCount, sessionInfo)


cdef nvmlReturn_t _nvmlVgpuInstanceGetGpuInstanceId(nvmlVgpuInstance_t vgpuInstance, unsigned int* gpuInstanceId) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetGpuInstanceId
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetGpuInstanceId == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetGpuInstanceId is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned int*) noexcept nogil>__nvmlVgpuInstanceGetGpuInstanceId)(
        vgpuInstance, gpuInstanceId)


cdef nvmlReturn_t _nvmlVgpuInstanceGetGpuPciId(nvmlVgpuInstance_t vgpuInstance, char* vgpuPciId, unsigned int* length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetGpuPciId
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetGpuPciId == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetGpuPciId is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, char*, unsigned int*) noexcept nogil>__nvmlVgpuInstanceGetGpuPciId)(
        vgpuInstance, vgpuPciId, length)


cdef nvmlReturn_t _nvmlVgpuTypeGetCapabilities(nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuCapability_t capability, unsigned int* capResult) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuTypeGetCapabilities
    _check_or_init_nvml()
    if __nvmlVgpuTypeGetCapabilities == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuTypeGetCapabilities is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuTypeId_t, nvmlVgpuCapability_t, unsigned int*) noexcept nogil>__nvmlVgpuTypeGetCapabilities)(
        vgpuTypeId, capability, capResult)


cdef nvmlReturn_t _nvmlVgpuInstanceGetMdevUUID(nvmlVgpuInstance_t vgpuInstance, char* mdevUuid, unsigned int size) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetMdevUUID
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetMdevUUID == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetMdevUUID is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, char*, unsigned int) noexcept nogil>__nvmlVgpuInstanceGetMdevUUID)(
        vgpuInstance, mdevUuid, size)


cdef nvmlReturn_t _nvmlGpuInstanceGetCreatableVgpus(nvmlGpuInstance_t gpuInstance, nvmlVgpuTypeIdInfo_t* pVgpus) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGpuInstanceGetCreatableVgpus
    _check_or_init_nvml()
    if __nvmlGpuInstanceGetCreatableVgpus == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGpuInstanceGetCreatableVgpus is not found")
    return (<nvmlReturn_t (*)(nvmlGpuInstance_t, nvmlVgpuTypeIdInfo_t*) noexcept nogil>__nvmlGpuInstanceGetCreatableVgpus)(
        gpuInstance, pVgpus)


cdef nvmlReturn_t _nvmlVgpuTypeGetMaxInstancesPerGpuInstance(nvmlVgpuTypeMaxInstance_t* pMaxInstance) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuTypeGetMaxInstancesPerGpuInstance
    _check_or_init_nvml()
    if __nvmlVgpuTypeGetMaxInstancesPerGpuInstance == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuTypeGetMaxInstancesPerGpuInstance is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuTypeMaxInstance_t*) noexcept nogil>__nvmlVgpuTypeGetMaxInstancesPerGpuInstance)(
        pMaxInstance)


cdef nvmlReturn_t _nvmlGpuInstanceGetActiveVgpus(nvmlGpuInstance_t gpuInstance, nvmlActiveVgpuInstanceInfo_t* pVgpuInstanceInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGpuInstanceGetActiveVgpus
    _check_or_init_nvml()
    if __nvmlGpuInstanceGetActiveVgpus == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGpuInstanceGetActiveVgpus is not found")
    return (<nvmlReturn_t (*)(nvmlGpuInstance_t, nvmlActiveVgpuInstanceInfo_t*) noexcept nogil>__nvmlGpuInstanceGetActiveVgpus)(
        gpuInstance, pVgpuInstanceInfo)


cdef nvmlReturn_t _nvmlGpuInstanceSetVgpuSchedulerState(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerState_t* pScheduler) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGpuInstanceSetVgpuSchedulerState
    _check_or_init_nvml()
    if __nvmlGpuInstanceSetVgpuSchedulerState == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGpuInstanceSetVgpuSchedulerState is not found")
    return (<nvmlReturn_t (*)(nvmlGpuInstance_t, nvmlVgpuSchedulerState_t*) noexcept nogil>__nvmlGpuInstanceSetVgpuSchedulerState)(
        gpuInstance, pScheduler)


cdef nvmlReturn_t _nvmlGpuInstanceGetVgpuSchedulerState(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerStateInfo_t* pSchedulerStateInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGpuInstanceGetVgpuSchedulerState
    _check_or_init_nvml()
    if __nvmlGpuInstanceGetVgpuSchedulerState == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGpuInstanceGetVgpuSchedulerState is not found")
    return (<nvmlReturn_t (*)(nvmlGpuInstance_t, nvmlVgpuSchedulerStateInfo_t*) noexcept nogil>__nvmlGpuInstanceGetVgpuSchedulerState)(
        gpuInstance, pSchedulerStateInfo)


cdef nvmlReturn_t _nvmlGpuInstanceGetVgpuSchedulerLog(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerLogInfo_t* pSchedulerLogInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGpuInstanceGetVgpuSchedulerLog
    _check_or_init_nvml()
    if __nvmlGpuInstanceGetVgpuSchedulerLog == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGpuInstanceGetVgpuSchedulerLog is not found")
    return (<nvmlReturn_t (*)(nvmlGpuInstance_t, nvmlVgpuSchedulerLogInfo_t*) noexcept nogil>__nvmlGpuInstanceGetVgpuSchedulerLog)(
        gpuInstance, pSchedulerLogInfo)


cdef nvmlReturn_t _nvmlGpuInstanceGetVgpuTypeCreatablePlacements(nvmlGpuInstance_t gpuInstance, nvmlVgpuCreatablePlacementInfo_t* pCreatablePlacementInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGpuInstanceGetVgpuTypeCreatablePlacements
    _check_or_init_nvml()
    if __nvmlGpuInstanceGetVgpuTypeCreatablePlacements == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGpuInstanceGetVgpuTypeCreatablePlacements is not found")
    return (<nvmlReturn_t (*)(nvmlGpuInstance_t, nvmlVgpuCreatablePlacementInfo_t*) noexcept nogil>__nvmlGpuInstanceGetVgpuTypeCreatablePlacements)(
        gpuInstance, pCreatablePlacementInfo)


cdef nvmlReturn_t _nvmlGpuInstanceGetVgpuHeterogeneousMode(nvmlGpuInstance_t gpuInstance, nvmlVgpuHeterogeneousMode_t* pHeterogeneousMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGpuInstanceGetVgpuHeterogeneousMode
    _check_or_init_nvml()
    if __nvmlGpuInstanceGetVgpuHeterogeneousMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGpuInstanceGetVgpuHeterogeneousMode is not found")
    return (<nvmlReturn_t (*)(nvmlGpuInstance_t, nvmlVgpuHeterogeneousMode_t*) noexcept nogil>__nvmlGpuInstanceGetVgpuHeterogeneousMode)(
        gpuInstance, pHeterogeneousMode)


cdef nvmlReturn_t _nvmlGpuInstanceSetVgpuHeterogeneousMode(nvmlGpuInstance_t gpuInstance, const nvmlVgpuHeterogeneousMode_t* pHeterogeneousMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGpuInstanceSetVgpuHeterogeneousMode
    _check_or_init_nvml()
    if __nvmlGpuInstanceSetVgpuHeterogeneousMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGpuInstanceSetVgpuHeterogeneousMode is not found")
    return (<nvmlReturn_t (*)(nvmlGpuInstance_t, const nvmlVgpuHeterogeneousMode_t*) noexcept nogil>__nvmlGpuInstanceSetVgpuHeterogeneousMode)(
        gpuInstance, pHeterogeneousMode)


cdef nvmlReturn_t _nvmlVgpuInstanceGetMetadata(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuMetadata_t* vgpuMetadata, unsigned int* bufferSize) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetMetadata
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetMetadata == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetMetadata is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, nvmlVgpuMetadata_t*, unsigned int*) noexcept nogil>__nvmlVgpuInstanceGetMetadata)(
        vgpuInstance, vgpuMetadata, bufferSize)


cdef nvmlReturn_t _nvmlDeviceGetVgpuMetadata(nvmlDevice_t device, nvmlVgpuPgpuMetadata_t* pgpuMetadata, unsigned int* bufferSize) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetVgpuMetadata
    _check_or_init_nvml()
    if __nvmlDeviceGetVgpuMetadata == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetVgpuMetadata is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlVgpuPgpuMetadata_t*, unsigned int*) noexcept nogil>__nvmlDeviceGetVgpuMetadata)(
        device, pgpuMetadata, bufferSize)


cdef nvmlReturn_t _nvmlGetVgpuCompatibility(nvmlVgpuMetadata_t* vgpuMetadata, nvmlVgpuPgpuMetadata_t* pgpuMetadata, nvmlVgpuPgpuCompatibility_t* compatibilityInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGetVgpuCompatibility
    _check_or_init_nvml()
    if __nvmlGetVgpuCompatibility == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGetVgpuCompatibility is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuMetadata_t*, nvmlVgpuPgpuMetadata_t*, nvmlVgpuPgpuCompatibility_t*) noexcept nogil>__nvmlGetVgpuCompatibility)(
        vgpuMetadata, pgpuMetadata, compatibilityInfo)


cdef nvmlReturn_t _nvmlDeviceGetPgpuMetadataString(nvmlDevice_t device, char* pgpuMetadata, unsigned int* bufferSize) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetPgpuMetadataString
    _check_or_init_nvml()
    if __nvmlDeviceGetPgpuMetadataString == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetPgpuMetadataString is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, char*, unsigned int*) noexcept nogil>__nvmlDeviceGetPgpuMetadataString)(
        device, pgpuMetadata, bufferSize)


cdef nvmlReturn_t _nvmlDeviceGetVgpuSchedulerLog(nvmlDevice_t device, nvmlVgpuSchedulerLog_t* pSchedulerLog) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetVgpuSchedulerLog
    _check_or_init_nvml()
    if __nvmlDeviceGetVgpuSchedulerLog == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetVgpuSchedulerLog is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlVgpuSchedulerLog_t*) noexcept nogil>__nvmlDeviceGetVgpuSchedulerLog)(
        device, pSchedulerLog)


cdef nvmlReturn_t _nvmlDeviceGetVgpuSchedulerState(nvmlDevice_t device, nvmlVgpuSchedulerGetState_t* pSchedulerState) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetVgpuSchedulerState
    _check_or_init_nvml()
    if __nvmlDeviceGetVgpuSchedulerState == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetVgpuSchedulerState is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlVgpuSchedulerGetState_t*) noexcept nogil>__nvmlDeviceGetVgpuSchedulerState)(
        device, pSchedulerState)


cdef nvmlReturn_t _nvmlDeviceGetVgpuSchedulerCapabilities(nvmlDevice_t device, nvmlVgpuSchedulerCapabilities_t* pCapabilities) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetVgpuSchedulerCapabilities
    _check_or_init_nvml()
    if __nvmlDeviceGetVgpuSchedulerCapabilities == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetVgpuSchedulerCapabilities is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlVgpuSchedulerCapabilities_t*) noexcept nogil>__nvmlDeviceGetVgpuSchedulerCapabilities)(
        device, pCapabilities)


cdef nvmlReturn_t _nvmlDeviceSetVgpuSchedulerState(nvmlDevice_t device, nvmlVgpuSchedulerSetState_t* pSchedulerState) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetVgpuSchedulerState
    _check_or_init_nvml()
    if __nvmlDeviceSetVgpuSchedulerState == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetVgpuSchedulerState is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlVgpuSchedulerSetState_t*) noexcept nogil>__nvmlDeviceSetVgpuSchedulerState)(
        device, pSchedulerState)


cdef nvmlReturn_t _nvmlGetVgpuVersion(nvmlVgpuVersion_t* supported, nvmlVgpuVersion_t* current) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGetVgpuVersion
    _check_or_init_nvml()
    if __nvmlGetVgpuVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGetVgpuVersion is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuVersion_t*, nvmlVgpuVersion_t*) noexcept nogil>__nvmlGetVgpuVersion)(
        supported, current)


cdef nvmlReturn_t _nvmlSetVgpuVersion(nvmlVgpuVersion_t* vgpuVersion) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSetVgpuVersion
    _check_or_init_nvml()
    if __nvmlSetVgpuVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSetVgpuVersion is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuVersion_t*) noexcept nogil>__nvmlSetVgpuVersion)(
        vgpuVersion)


cdef nvmlReturn_t _nvmlDeviceGetVgpuUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, nvmlValueType_t* sampleValType, unsigned int* vgpuInstanceSamplesCount, nvmlVgpuInstanceUtilizationSample_t* utilizationSamples) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetVgpuUtilization
    _check_or_init_nvml()
    if __nvmlDeviceGetVgpuUtilization == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetVgpuUtilization is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned long long, nvmlValueType_t*, unsigned int*, nvmlVgpuInstanceUtilizationSample_t*) noexcept nogil>__nvmlDeviceGetVgpuUtilization)(
        device, lastSeenTimeStamp, sampleValType, vgpuInstanceSamplesCount, utilizationSamples)


cdef nvmlReturn_t _nvmlDeviceGetVgpuInstancesUtilizationInfo(nvmlDevice_t device, nvmlVgpuInstancesUtilizationInfo_t* vgpuUtilInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetVgpuInstancesUtilizationInfo
    _check_or_init_nvml()
    if __nvmlDeviceGetVgpuInstancesUtilizationInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetVgpuInstancesUtilizationInfo is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlVgpuInstancesUtilizationInfo_t*) noexcept nogil>__nvmlDeviceGetVgpuInstancesUtilizationInfo)(
        device, vgpuUtilInfo)


cdef nvmlReturn_t _nvmlDeviceGetVgpuProcessUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, unsigned int* vgpuProcessSamplesCount, nvmlVgpuProcessUtilizationSample_t* utilizationSamples) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetVgpuProcessUtilization
    _check_or_init_nvml()
    if __nvmlDeviceGetVgpuProcessUtilization == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetVgpuProcessUtilization is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned long long, unsigned int*, nvmlVgpuProcessUtilizationSample_t*) noexcept nogil>__nvmlDeviceGetVgpuProcessUtilization)(
        device, lastSeenTimeStamp, vgpuProcessSamplesCount, utilizationSamples)


cdef nvmlReturn_t _nvmlDeviceGetVgpuProcessesUtilizationInfo(nvmlDevice_t device, nvmlVgpuProcessesUtilizationInfo_t* vgpuProcUtilInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetVgpuProcessesUtilizationInfo
    _check_or_init_nvml()
    if __nvmlDeviceGetVgpuProcessesUtilizationInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetVgpuProcessesUtilizationInfo is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlVgpuProcessesUtilizationInfo_t*) noexcept nogil>__nvmlDeviceGetVgpuProcessesUtilizationInfo)(
        device, vgpuProcUtilInfo)


cdef nvmlReturn_t _nvmlVgpuInstanceGetAccountingMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t* mode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetAccountingMode
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetAccountingMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetAccountingMode is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, nvmlEnableState_t*) noexcept nogil>__nvmlVgpuInstanceGetAccountingMode)(
        vgpuInstance, mode)


cdef nvmlReturn_t _nvmlVgpuInstanceGetAccountingPids(nvmlVgpuInstance_t vgpuInstance, unsigned int* count, unsigned int* pids) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetAccountingPids
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetAccountingPids == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetAccountingPids is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned int*, unsigned int*) noexcept nogil>__nvmlVgpuInstanceGetAccountingPids)(
        vgpuInstance, count, pids)


cdef nvmlReturn_t _nvmlVgpuInstanceGetAccountingStats(nvmlVgpuInstance_t vgpuInstance, unsigned int pid, nvmlAccountingStats_t* stats) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetAccountingStats
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetAccountingStats == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetAccountingStats is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned int, nvmlAccountingStats_t*) noexcept nogil>__nvmlVgpuInstanceGetAccountingStats)(
        vgpuInstance, pid, stats)


cdef nvmlReturn_t _nvmlVgpuInstanceClearAccountingPids(nvmlVgpuInstance_t vgpuInstance) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceClearAccountingPids
    _check_or_init_nvml()
    if __nvmlVgpuInstanceClearAccountingPids == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceClearAccountingPids is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t) noexcept nogil>__nvmlVgpuInstanceClearAccountingPids)(
        vgpuInstance)


cdef nvmlReturn_t _nvmlVgpuInstanceGetLicenseInfo_v2(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuLicenseInfo_t* licenseInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlVgpuInstanceGetLicenseInfo_v2
    _check_or_init_nvml()
    if __nvmlVgpuInstanceGetLicenseInfo_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlVgpuInstanceGetLicenseInfo_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlVgpuInstance_t, nvmlVgpuLicenseInfo_t*) noexcept nogil>__nvmlVgpuInstanceGetLicenseInfo_v2)(
        vgpuInstance, licenseInfo)


cdef nvmlReturn_t _nvmlGetExcludedDeviceCount(unsigned int* deviceCount) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGetExcludedDeviceCount
    _check_or_init_nvml()
    if __nvmlGetExcludedDeviceCount == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGetExcludedDeviceCount is not found")
    return (<nvmlReturn_t (*)(unsigned int*) noexcept nogil>__nvmlGetExcludedDeviceCount)(
        deviceCount)


cdef nvmlReturn_t _nvmlGetExcludedDeviceInfoByIndex(unsigned int index, nvmlExcludedDeviceInfo_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGetExcludedDeviceInfoByIndex
    _check_or_init_nvml()
    if __nvmlGetExcludedDeviceInfoByIndex == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGetExcludedDeviceInfoByIndex is not found")
    return (<nvmlReturn_t (*)(unsigned int, nvmlExcludedDeviceInfo_t*) noexcept nogil>__nvmlGetExcludedDeviceInfoByIndex)(
        index, info)


cdef nvmlReturn_t _nvmlDeviceSetMigMode(nvmlDevice_t device, unsigned int mode, nvmlReturn_t* activationStatus) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetMigMode
    _check_or_init_nvml()
    if __nvmlDeviceSetMigMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetMigMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlReturn_t*) noexcept nogil>__nvmlDeviceSetMigMode)(
        device, mode, activationStatus)


cdef nvmlReturn_t _nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int* currentMode, unsigned int* pendingMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetMigMode
    _check_or_init_nvml()
    if __nvmlDeviceGetMigMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetMigMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, unsigned int*) noexcept nogil>__nvmlDeviceGetMigMode)(
        device, currentMode, pendingMode)


cdef nvmlReturn_t _nvmlDeviceGetGpuInstanceProfileInfoV(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_v2_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetGpuInstanceProfileInfoV
    _check_or_init_nvml()
    if __nvmlDeviceGetGpuInstanceProfileInfoV == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetGpuInstanceProfileInfoV is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlGpuInstanceProfileInfo_v2_t*) noexcept nogil>__nvmlDeviceGetGpuInstanceProfileInfoV)(
        device, profile, info)


cdef nvmlReturn_t _nvmlDeviceGetGpuInstancePossiblePlacements_v2(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstancePlacement_t* placements, unsigned int* count) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetGpuInstancePossiblePlacements_v2
    _check_or_init_nvml()
    if __nvmlDeviceGetGpuInstancePossiblePlacements_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetGpuInstancePossiblePlacements_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlGpuInstancePlacement_t*, unsigned int*) noexcept nogil>__nvmlDeviceGetGpuInstancePossiblePlacements_v2)(
        device, profileId, placements, count)


cdef nvmlReturn_t _nvmlDeviceGetGpuInstanceRemainingCapacity(nvmlDevice_t device, unsigned int profileId, unsigned int* count) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetGpuInstanceRemainingCapacity
    _check_or_init_nvml()
    if __nvmlDeviceGetGpuInstanceRemainingCapacity == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetGpuInstanceRemainingCapacity is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int*) noexcept nogil>__nvmlDeviceGetGpuInstanceRemainingCapacity)(
        device, profileId, count)


cdef nvmlReturn_t _nvmlDeviceCreateGpuInstance(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t* gpuInstance) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceCreateGpuInstance
    _check_or_init_nvml()
    if __nvmlDeviceCreateGpuInstance == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceCreateGpuInstance is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlGpuInstance_t*) noexcept nogil>__nvmlDeviceCreateGpuInstance)(
        device, profileId, gpuInstance)


cdef nvmlReturn_t _nvmlDeviceCreateGpuInstanceWithPlacement(nvmlDevice_t device, unsigned int profileId, const nvmlGpuInstancePlacement_t* placement, nvmlGpuInstance_t* gpuInstance) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceCreateGpuInstanceWithPlacement
    _check_or_init_nvml()
    if __nvmlDeviceCreateGpuInstanceWithPlacement == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceCreateGpuInstanceWithPlacement is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, const nvmlGpuInstancePlacement_t*, nvmlGpuInstance_t*) noexcept nogil>__nvmlDeviceCreateGpuInstanceWithPlacement)(
        device, profileId, placement, gpuInstance)


cdef nvmlReturn_t _nvmlGpuInstanceDestroy(nvmlGpuInstance_t gpuInstance) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGpuInstanceDestroy
    _check_or_init_nvml()
    if __nvmlGpuInstanceDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGpuInstanceDestroy is not found")
    return (<nvmlReturn_t (*)(nvmlGpuInstance_t) noexcept nogil>__nvmlGpuInstanceDestroy)(
        gpuInstance)


cdef nvmlReturn_t _nvmlDeviceGetGpuInstances(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t* gpuInstances, unsigned int* count) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetGpuInstances
    _check_or_init_nvml()
    if __nvmlDeviceGetGpuInstances == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetGpuInstances is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlGpuInstance_t*, unsigned int*) noexcept nogil>__nvmlDeviceGetGpuInstances)(
        device, profileId, gpuInstances, count)


cdef nvmlReturn_t _nvmlDeviceGetGpuInstanceById(nvmlDevice_t device, unsigned int id, nvmlGpuInstance_t* gpuInstance) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetGpuInstanceById
    _check_or_init_nvml()
    if __nvmlDeviceGetGpuInstanceById == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetGpuInstanceById is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlGpuInstance_t*) noexcept nogil>__nvmlDeviceGetGpuInstanceById)(
        device, id, gpuInstance)


cdef nvmlReturn_t _nvmlGpuInstanceGetInfo(nvmlGpuInstance_t gpuInstance, nvmlGpuInstanceInfo_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGpuInstanceGetInfo
    _check_or_init_nvml()
    if __nvmlGpuInstanceGetInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGpuInstanceGetInfo is not found")
    return (<nvmlReturn_t (*)(nvmlGpuInstance_t, nvmlGpuInstanceInfo_t*) noexcept nogil>__nvmlGpuInstanceGetInfo)(
        gpuInstance, info)


cdef nvmlReturn_t _nvmlGpuInstanceGetComputeInstanceProfileInfoV(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_v2_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGpuInstanceGetComputeInstanceProfileInfoV
    _check_or_init_nvml()
    if __nvmlGpuInstanceGetComputeInstanceProfileInfoV == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGpuInstanceGetComputeInstanceProfileInfoV is not found")
    return (<nvmlReturn_t (*)(nvmlGpuInstance_t, unsigned int, unsigned int, nvmlComputeInstanceProfileInfo_v2_t*) noexcept nogil>__nvmlGpuInstanceGetComputeInstanceProfileInfoV)(
        gpuInstance, profile, engProfile, info)


cdef nvmlReturn_t _nvmlGpuInstanceGetComputeInstanceRemainingCapacity(nvmlGpuInstance_t gpuInstance, unsigned int profileId, unsigned int* count) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGpuInstanceGetComputeInstanceRemainingCapacity
    _check_or_init_nvml()
    if __nvmlGpuInstanceGetComputeInstanceRemainingCapacity == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGpuInstanceGetComputeInstanceRemainingCapacity is not found")
    return (<nvmlReturn_t (*)(nvmlGpuInstance_t, unsigned int, unsigned int*) noexcept nogil>__nvmlGpuInstanceGetComputeInstanceRemainingCapacity)(
        gpuInstance, profileId, count)


cdef nvmlReturn_t _nvmlGpuInstanceGetComputeInstancePossiblePlacements(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstancePlacement_t* placements, unsigned int* count) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGpuInstanceGetComputeInstancePossiblePlacements
    _check_or_init_nvml()
    if __nvmlGpuInstanceGetComputeInstancePossiblePlacements == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGpuInstanceGetComputeInstancePossiblePlacements is not found")
    return (<nvmlReturn_t (*)(nvmlGpuInstance_t, unsigned int, nvmlComputeInstancePlacement_t*, unsigned int*) noexcept nogil>__nvmlGpuInstanceGetComputeInstancePossiblePlacements)(
        gpuInstance, profileId, placements, count)


cdef nvmlReturn_t _nvmlGpuInstanceCreateComputeInstance(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t* computeInstance) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGpuInstanceCreateComputeInstance
    _check_or_init_nvml()
    if __nvmlGpuInstanceCreateComputeInstance == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGpuInstanceCreateComputeInstance is not found")
    return (<nvmlReturn_t (*)(nvmlGpuInstance_t, unsigned int, nvmlComputeInstance_t*) noexcept nogil>__nvmlGpuInstanceCreateComputeInstance)(
        gpuInstance, profileId, computeInstance)


cdef nvmlReturn_t _nvmlGpuInstanceCreateComputeInstanceWithPlacement(nvmlGpuInstance_t gpuInstance, unsigned int profileId, const nvmlComputeInstancePlacement_t* placement, nvmlComputeInstance_t* computeInstance) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGpuInstanceCreateComputeInstanceWithPlacement
    _check_or_init_nvml()
    if __nvmlGpuInstanceCreateComputeInstanceWithPlacement == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGpuInstanceCreateComputeInstanceWithPlacement is not found")
    return (<nvmlReturn_t (*)(nvmlGpuInstance_t, unsigned int, const nvmlComputeInstancePlacement_t*, nvmlComputeInstance_t*) noexcept nogil>__nvmlGpuInstanceCreateComputeInstanceWithPlacement)(
        gpuInstance, profileId, placement, computeInstance)


cdef nvmlReturn_t _nvmlComputeInstanceDestroy(nvmlComputeInstance_t computeInstance) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlComputeInstanceDestroy
    _check_or_init_nvml()
    if __nvmlComputeInstanceDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlComputeInstanceDestroy is not found")
    return (<nvmlReturn_t (*)(nvmlComputeInstance_t) noexcept nogil>__nvmlComputeInstanceDestroy)(
        computeInstance)


cdef nvmlReturn_t _nvmlGpuInstanceGetComputeInstances(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t* computeInstances, unsigned int* count) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGpuInstanceGetComputeInstances
    _check_or_init_nvml()
    if __nvmlGpuInstanceGetComputeInstances == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGpuInstanceGetComputeInstances is not found")
    return (<nvmlReturn_t (*)(nvmlGpuInstance_t, unsigned int, nvmlComputeInstance_t*, unsigned int*) noexcept nogil>__nvmlGpuInstanceGetComputeInstances)(
        gpuInstance, profileId, computeInstances, count)


cdef nvmlReturn_t _nvmlGpuInstanceGetComputeInstanceById(nvmlGpuInstance_t gpuInstance, unsigned int id, nvmlComputeInstance_t* computeInstance) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGpuInstanceGetComputeInstanceById
    _check_or_init_nvml()
    if __nvmlGpuInstanceGetComputeInstanceById == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGpuInstanceGetComputeInstanceById is not found")
    return (<nvmlReturn_t (*)(nvmlGpuInstance_t, unsigned int, nvmlComputeInstance_t*) noexcept nogil>__nvmlGpuInstanceGetComputeInstanceById)(
        gpuInstance, id, computeInstance)


cdef nvmlReturn_t _nvmlComputeInstanceGetInfo_v2(nvmlComputeInstance_t computeInstance, nvmlComputeInstanceInfo_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlComputeInstanceGetInfo_v2
    _check_or_init_nvml()
    if __nvmlComputeInstanceGetInfo_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlComputeInstanceGetInfo_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlComputeInstance_t, nvmlComputeInstanceInfo_t*) noexcept nogil>__nvmlComputeInstanceGetInfo_v2)(
        computeInstance, info)


cdef nvmlReturn_t _nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device, unsigned int* isMigDevice) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceIsMigDeviceHandle
    _check_or_init_nvml()
    if __nvmlDeviceIsMigDeviceHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceIsMigDeviceHandle is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceIsMigDeviceHandle)(
        device, isMigDevice)


cdef nvmlReturn_t _nvmlDeviceGetGpuInstanceId(nvmlDevice_t device, unsigned int* id) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetGpuInstanceId
    _check_or_init_nvml()
    if __nvmlDeviceGetGpuInstanceId == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetGpuInstanceId is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetGpuInstanceId)(
        device, id)


cdef nvmlReturn_t _nvmlDeviceGetComputeInstanceId(nvmlDevice_t device, unsigned int* id) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetComputeInstanceId
    _check_or_init_nvml()
    if __nvmlDeviceGetComputeInstanceId == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetComputeInstanceId is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetComputeInstanceId)(
        device, id)


cdef nvmlReturn_t _nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device, unsigned int* count) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetMaxMigDeviceCount
    _check_or_init_nvml()
    if __nvmlDeviceGetMaxMigDeviceCount == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetMaxMigDeviceCount is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*) noexcept nogil>__nvmlDeviceGetMaxMigDeviceCount)(
        device, count)


cdef nvmlReturn_t _nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t device, unsigned int index, nvmlDevice_t* migDevice) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetMigDeviceHandleByIndex
    _check_or_init_nvml()
    if __nvmlDeviceGetMigDeviceHandleByIndex == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetMigDeviceHandleByIndex is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlDevice_t*) noexcept nogil>__nvmlDeviceGetMigDeviceHandleByIndex)(
        device, index, migDevice)


cdef nvmlReturn_t _nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t migDevice, nvmlDevice_t* device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetDeviceHandleFromMigDeviceHandle
    _check_or_init_nvml()
    if __nvmlDeviceGetDeviceHandleFromMigDeviceHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetDeviceHandleFromMigDeviceHandle is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlDevice_t*) noexcept nogil>__nvmlDeviceGetDeviceHandleFromMigDeviceHandle)(
        migDevice, device)


cdef nvmlReturn_t _nvmlDeviceGetCapabilities(nvmlDevice_t device, nvmlDeviceCapabilities_t* caps) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetCapabilities
    _check_or_init_nvml()
    if __nvmlDeviceGetCapabilities == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetCapabilities is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlDeviceCapabilities_t*) noexcept nogil>__nvmlDeviceGetCapabilities)(
        device, caps)


cdef nvmlReturn_t _nvmlDevicePowerSmoothingActivatePresetProfile(nvmlDevice_t device, nvmlPowerSmoothingProfile_t* profile) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDevicePowerSmoothingActivatePresetProfile
    _check_or_init_nvml()
    if __nvmlDevicePowerSmoothingActivatePresetProfile == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDevicePowerSmoothingActivatePresetProfile is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlPowerSmoothingProfile_t*) noexcept nogil>__nvmlDevicePowerSmoothingActivatePresetProfile)(
        device, profile)


cdef nvmlReturn_t _nvmlDevicePowerSmoothingUpdatePresetProfileParam(nvmlDevice_t device, nvmlPowerSmoothingProfile_t* profile) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDevicePowerSmoothingUpdatePresetProfileParam
    _check_or_init_nvml()
    if __nvmlDevicePowerSmoothingUpdatePresetProfileParam == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDevicePowerSmoothingUpdatePresetProfileParam is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlPowerSmoothingProfile_t*) noexcept nogil>__nvmlDevicePowerSmoothingUpdatePresetProfileParam)(
        device, profile)


cdef nvmlReturn_t _nvmlDevicePowerSmoothingSetState(nvmlDevice_t device, nvmlPowerSmoothingState_t* state) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDevicePowerSmoothingSetState
    _check_or_init_nvml()
    if __nvmlDevicePowerSmoothingSetState == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDevicePowerSmoothingSetState is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlPowerSmoothingState_t*) noexcept nogil>__nvmlDevicePowerSmoothingSetState)(
        device, state)


cdef nvmlReturn_t _nvmlDeviceGetAddressingMode(nvmlDevice_t device, nvmlDeviceAddressingMode_t* mode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetAddressingMode
    _check_or_init_nvml()
    if __nvmlDeviceGetAddressingMode == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetAddressingMode is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlDeviceAddressingMode_t*) noexcept nogil>__nvmlDeviceGetAddressingMode)(
        device, mode)


cdef nvmlReturn_t _nvmlDeviceGetRepairStatus(nvmlDevice_t device, nvmlRepairStatus_t* repairStatus) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetRepairStatus
    _check_or_init_nvml()
    if __nvmlDeviceGetRepairStatus == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetRepairStatus is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlRepairStatus_t*) noexcept nogil>__nvmlDeviceGetRepairStatus)(
        device, repairStatus)


cdef nvmlReturn_t _nvmlDeviceGetPowerMizerMode_v1(nvmlDevice_t device, nvmlDevicePowerMizerModes_v1_t* powerMizerMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetPowerMizerMode_v1
    _check_or_init_nvml()
    if __nvmlDeviceGetPowerMizerMode_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetPowerMizerMode_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlDevicePowerMizerModes_v1_t*) noexcept nogil>__nvmlDeviceGetPowerMizerMode_v1)(
        device, powerMizerMode)


cdef nvmlReturn_t _nvmlDeviceSetPowerMizerMode_v1(nvmlDevice_t device, nvmlDevicePowerMizerModes_v1_t* powerMizerMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetPowerMizerMode_v1
    _check_or_init_nvml()
    if __nvmlDeviceSetPowerMizerMode_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetPowerMizerMode_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlDevicePowerMizerModes_v1_t*) noexcept nogil>__nvmlDeviceSetPowerMizerMode_v1)(
        device, powerMizerMode)


cdef nvmlReturn_t _nvmlDeviceGetPdi(nvmlDevice_t device, nvmlPdi_t* pdi) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetPdi
    _check_or_init_nvml()
    if __nvmlDeviceGetPdi == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetPdi is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlPdi_t*) noexcept nogil>__nvmlDeviceGetPdi)(
        device, pdi)


cdef nvmlReturn_t _nvmlDeviceSetHostname_v1(nvmlDevice_t device, nvmlHostname_v1_t* hostname) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetHostname_v1
    _check_or_init_nvml()
    if __nvmlDeviceSetHostname_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetHostname_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlHostname_v1_t*) noexcept nogil>__nvmlDeviceSetHostname_v1)(
        device, hostname)


cdef nvmlReturn_t _nvmlDeviceGetHostname_v1(nvmlDevice_t device, nvmlHostname_v1_t* hostname) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetHostname_v1
    _check_or_init_nvml()
    if __nvmlDeviceGetHostname_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetHostname_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlHostname_v1_t*) noexcept nogil>__nvmlDeviceGetHostname_v1)(
        device, hostname)


cdef nvmlReturn_t _nvmlDeviceGetNvLinkInfo(nvmlDevice_t device, nvmlNvLinkInfo_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetNvLinkInfo
    _check_or_init_nvml()
    if __nvmlDeviceGetNvLinkInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetNvLinkInfo is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlNvLinkInfo_t*) noexcept nogil>__nvmlDeviceGetNvLinkInfo)(
        device, info)


cdef nvmlReturn_t _nvmlDeviceReadWritePRM_v1(nvmlDevice_t device, nvmlPRMTLV_v1_t* buffer) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceReadWritePRM_v1
    _check_or_init_nvml()
    if __nvmlDeviceReadWritePRM_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceReadWritePRM_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlPRMTLV_v1_t*) noexcept nogil>__nvmlDeviceReadWritePRM_v1)(
        device, buffer)


cdef nvmlReturn_t _nvmlDeviceGetGpuInstanceProfileInfoByIdV(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstanceProfileInfo_v2_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetGpuInstanceProfileInfoByIdV
    _check_or_init_nvml()
    if __nvmlDeviceGetGpuInstanceProfileInfoByIdV == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetGpuInstanceProfileInfoByIdV is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlGpuInstanceProfileInfo_v2_t*) noexcept nogil>__nvmlDeviceGetGpuInstanceProfileInfoByIdV)(
        device, profileId, info)


cdef nvmlReturn_t _nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts(nvmlDevice_t device, nvmlEccSramUniqueUncorrectedErrorCounts_t* errorCounts) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts
    _check_or_init_nvml()
    if __nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlEccSramUniqueUncorrectedErrorCounts_t*) noexcept nogil>__nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts)(
        device, errorCounts)


cdef nvmlReturn_t _nvmlDeviceGetUnrepairableMemoryFlag_v1(nvmlDevice_t device, nvmlUnrepairableMemoryStatus_v1_t* unrepairableMemoryStatus) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetUnrepairableMemoryFlag_v1
    _check_or_init_nvml()
    if __nvmlDeviceGetUnrepairableMemoryFlag_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetUnrepairableMemoryFlag_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlUnrepairableMemoryStatus_v1_t*) noexcept nogil>__nvmlDeviceGetUnrepairableMemoryFlag_v1)(
        device, unrepairableMemoryStatus)


cdef nvmlReturn_t _nvmlDeviceReadPRMCounters_v1(nvmlDevice_t device, nvmlPRMCounterList_v1_t* counterList) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceReadPRMCounters_v1
    _check_or_init_nvml()
    if __nvmlDeviceReadPRMCounters_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceReadPRMCounters_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlPRMCounterList_v1_t*) noexcept nogil>__nvmlDeviceReadPRMCounters_v1)(
        device, counterList)


cdef nvmlReturn_t _nvmlDeviceSetRusdSettings_v1(nvmlDevice_t device, nvmlRusdSettings_v1_t* settings) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetRusdSettings_v1
    _check_or_init_nvml()
    if __nvmlDeviceSetRusdSettings_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetRusdSettings_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlRusdSettings_v1_t*) noexcept nogil>__nvmlDeviceSetRusdSettings_v1)(
        device, settings)


cdef nvmlReturn_t _nvmlDeviceVgpuForceGspUnload(nvmlDevice_t device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceVgpuForceGspUnload
    _check_or_init_nvml()
    if __nvmlDeviceVgpuForceGspUnload == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceVgpuForceGspUnload is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t) noexcept nogil>__nvmlDeviceVgpuForceGspUnload)(
        device)


cdef nvmlReturn_t _nvmlDeviceGetVgpuSchedulerState_v2(nvmlDevice_t device, nvmlVgpuSchedulerStateInfo_v2_t* pSchedulerStateInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetVgpuSchedulerState_v2
    _check_or_init_nvml()
    if __nvmlDeviceGetVgpuSchedulerState_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetVgpuSchedulerState_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlVgpuSchedulerStateInfo_v2_t*) noexcept nogil>__nvmlDeviceGetVgpuSchedulerState_v2)(
        device, pSchedulerStateInfo)


cdef nvmlReturn_t _nvmlGpuInstanceGetVgpuSchedulerState_v2(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerStateInfo_v2_t* pSchedulerStateInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGpuInstanceGetVgpuSchedulerState_v2
    _check_or_init_nvml()
    if __nvmlGpuInstanceGetVgpuSchedulerState_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGpuInstanceGetVgpuSchedulerState_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlGpuInstance_t, nvmlVgpuSchedulerStateInfo_v2_t*) noexcept nogil>__nvmlGpuInstanceGetVgpuSchedulerState_v2)(
        gpuInstance, pSchedulerStateInfo)


cdef nvmlReturn_t _nvmlDeviceGetVgpuSchedulerLog_v2(nvmlDevice_t device, nvmlVgpuSchedulerLogInfo_v2_t* pSchedulerLogInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetVgpuSchedulerLog_v2
    _check_or_init_nvml()
    if __nvmlDeviceGetVgpuSchedulerLog_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetVgpuSchedulerLog_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlVgpuSchedulerLogInfo_v2_t*) noexcept nogil>__nvmlDeviceGetVgpuSchedulerLog_v2)(
        device, pSchedulerLogInfo)


cdef nvmlReturn_t _nvmlGpuInstanceGetVgpuSchedulerLog_v2(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerLogInfo_v2_t* pSchedulerLogInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGpuInstanceGetVgpuSchedulerLog_v2
    _check_or_init_nvml()
    if __nvmlGpuInstanceGetVgpuSchedulerLog_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGpuInstanceGetVgpuSchedulerLog_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlGpuInstance_t, nvmlVgpuSchedulerLogInfo_v2_t*) noexcept nogil>__nvmlGpuInstanceGetVgpuSchedulerLog_v2)(
        gpuInstance, pSchedulerLogInfo)


cdef nvmlReturn_t _nvmlDeviceSetVgpuSchedulerState_v2(nvmlDevice_t device, nvmlVgpuSchedulerState_v2_t* pSchedulerState) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetVgpuSchedulerState_v2
    _check_or_init_nvml()
    if __nvmlDeviceSetVgpuSchedulerState_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetVgpuSchedulerState_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlVgpuSchedulerState_v2_t*) noexcept nogil>__nvmlDeviceSetVgpuSchedulerState_v2)(
        device, pSchedulerState)


cdef nvmlReturn_t _nvmlGpuInstanceSetVgpuSchedulerState_v2(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerState_v2_t* pSchedulerState) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlGpuInstanceSetVgpuSchedulerState_v2
    _check_or_init_nvml()
    if __nvmlGpuInstanceSetVgpuSchedulerState_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlGpuInstanceSetVgpuSchedulerState_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlGpuInstance_t, nvmlVgpuSchedulerState_v2_t*) noexcept nogil>__nvmlGpuInstanceSetVgpuSchedulerState_v2)(
        gpuInstance, pSchedulerState)


cdef nvmlReturn_t _nvmlSystemGetCPER_v1(nvmlGetCPER_v1_t* cper) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemGetCPER_v1
    _check_or_init_nvml()
    if __nvmlSystemGetCPER_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemGetCPER_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlGetCPER_v1_t*) noexcept nogil>__nvmlSystemGetCPER_v1)(
        cper)


cdef nvmlReturn_t _nvmlDeviceGetBBXTimeData_v1(nvmlDevice_t device, nvmlBBXTimeData_v1_t* timeData) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetBBXTimeData_v1
    _check_or_init_nvml()
    if __nvmlDeviceGetBBXTimeData_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetBBXTimeData_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlBBXTimeData_v1_t*) noexcept nogil>__nvmlDeviceGetBBXTimeData_v1)(
        device, timeData)


cdef nvmlReturn_t _nvmlDeviceGetAccountingStats_v2(nvmlDevice_t device, nvmlAccountingStats_v2_t* stats) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetAccountingStats_v2
    _check_or_init_nvml()
    if __nvmlDeviceGetAccountingStats_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetAccountingStats_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlAccountingStats_v2_t*) noexcept nogil>__nvmlDeviceGetAccountingStats_v2)(
        device, stats)


cdef nvmlReturn_t _nvmlDeviceGetRemappedRows_v2(nvmlDevice_t device, nvmlRemappedRowsInfo_v2_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetRemappedRows_v2
    _check_or_init_nvml()
    if __nvmlDeviceGetRemappedRows_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetRemappedRows_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlRemappedRowsInfo_v2_t*) noexcept nogil>__nvmlDeviceGetRemappedRows_v2)(
        device, info)


cdef nvmlReturn_t _nvmlDeviceSetAdaptiveTgpMode_v1(nvmlDevice_t device, nvmlEnableState_t mode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetAdaptiveTgpMode_v1
    _check_or_init_nvml()
    if __nvmlDeviceSetAdaptiveTgpMode_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetAdaptiveTgpMode_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t) noexcept nogil>__nvmlDeviceSetAdaptiveTgpMode_v1)(
        device, mode)


cdef nvmlReturn_t _nvmlDeviceGetAdaptiveTgpModeInfo_v1(nvmlDevice_t device, nvmlAdaptiveTgpModeInfo_v1_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetAdaptiveTgpModeInfo_v1
    _check_or_init_nvml()
    if __nvmlDeviceGetAdaptiveTgpModeInfo_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetAdaptiveTgpModeInfo_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlAdaptiveTgpModeInfo_v1_t*) noexcept nogil>__nvmlDeviceGetAdaptiveTgpModeInfo_v1)(
        device, info)


cdef nvmlReturn_t _nvmlDeviceSetMemoryLimits_v1(nvmlDevice_t device, nvmlSetMemoryLimits_v1_t* limits) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetMemoryLimits_v1
    _check_or_init_nvml()
    if __nvmlDeviceSetMemoryLimits_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetMemoryLimits_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlSetMemoryLimits_v1_t*) noexcept nogil>__nvmlDeviceSetMemoryLimits_v1)(
        device, limits)


cdef nvmlReturn_t _nvmlDeviceGetMemoryLimits_v1(nvmlDevice_t device, nvmlGetMemoryLimits_v1_t* limits) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetMemoryLimits_v1
    _check_or_init_nvml()
    if __nvmlDeviceGetMemoryLimits_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetMemoryLimits_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlGetMemoryLimits_v1_t*) noexcept nogil>__nvmlDeviceGetMemoryLimits_v1)(
        device, limits)


cdef nvmlReturn_t _nvmlDeviceGetGpuFabricInfo_v4(nvmlDevice_t device, nvmlGpuFabricInfo_v4_t* gpuFabricInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetGpuFabricInfo_v4
    _check_or_init_nvml()
    if __nvmlDeviceGetGpuFabricInfo_v4 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetGpuFabricInfo_v4 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlGpuFabricInfo_v4_t*) noexcept nogil>__nvmlDeviceGetGpuFabricInfo_v4)(
        device, gpuFabricInfo)


cdef nvmlReturn_t _nvmlDevicePerfMetricsGetSamples_v1(nvmlDevice_t device, nvmlPerfMetricsSamples_v1_t* samples) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDevicePerfMetricsGetSamples_v1
    _check_or_init_nvml()
    if __nvmlDevicePerfMetricsGetSamples_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDevicePerfMetricsGetSamples_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlPerfMetricsSamples_v1_t*) noexcept nogil>__nvmlDevicePerfMetricsGetSamples_v1)(
        device, samples)


cdef nvmlReturn_t _nvmlDeviceSetNvlinkBwModeAsync_v1(nvmlDevice_t device, nvmlNvlinkSetBwModeAsync_v1_t* setBwModeAsync) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetNvlinkBwModeAsync_v1
    _check_or_init_nvml()
    if __nvmlDeviceSetNvlinkBwModeAsync_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetNvlinkBwModeAsync_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlNvlinkSetBwModeAsync_v1_t*) noexcept nogil>__nvmlDeviceSetNvlinkBwModeAsync_v1)(
        device, setBwModeAsync)


cdef nvmlReturn_t _nvmlDeviceGetNvLinkTelemetrySamples_v1(nvmlDevice_t device, nvmlNvlinkTelemetrySamples_v1_t* samples) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetNvLinkTelemetrySamples_v1
    _check_or_init_nvml()
    if __nvmlDeviceGetNvLinkTelemetrySamples_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetNvLinkTelemetrySamples_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlNvlinkTelemetrySamples_v1_t*) noexcept nogil>__nvmlDeviceGetNvLinkTelemetrySamples_v1)(
        device, samples)


cdef nvmlReturn_t _nvmlEventSetRegisterGpuOperationalEvents_v1(nvmlEventSet_t eventSet, const nvmlGpuOperationalEventConfig_v1_t* config) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlEventSetRegisterGpuOperationalEvents_v1
    _check_or_init_nvml()
    if __nvmlEventSetRegisterGpuOperationalEvents_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlEventSetRegisterGpuOperationalEvents_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlEventSet_t, const nvmlGpuOperationalEventConfig_v1_t*) noexcept nogil>__nvmlEventSetRegisterGpuOperationalEvents_v1)(
        eventSet, config)


cdef nvmlReturn_t _nvmlEventSetWait_v3(nvmlEventSet_t set, nvmlEventSetWait_v3_t* params) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlEventSetWait_v3
    _check_or_init_nvml()
    if __nvmlEventSetWait_v3 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlEventSetWait_v3 is not found")
    return (<nvmlReturn_t (*)(nvmlEventSet_t, nvmlEventSetWait_v3_t*) noexcept nogil>__nvmlEventSetWait_v3)(
        set, params)


cdef nvmlReturn_t _nvmlEventSetGetContextCount_v1(nvmlEventSet_t set, nvmlEventSetGetContextCount_v1_t* params) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlEventSetGetContextCount_v1
    _check_or_init_nvml()
    if __nvmlEventSetGetContextCount_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlEventSetGetContextCount_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlEventSet_t, nvmlEventSetGetContextCount_v1_t*) noexcept nogil>__nvmlEventSetGetContextCount_v1)(
        set, params)


cdef nvmlReturn_t _nvmlEventSetGetContextInfo_v1(nvmlEventSet_t set, nvmlEventSetGetContextInfo_v1_t* params) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlEventSetGetContextInfo_v1
    _check_or_init_nvml()
    if __nvmlEventSetGetContextInfo_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlEventSetGetContextInfo_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlEventSet_t, nvmlEventSetGetContextInfo_v1_t*) noexcept nogil>__nvmlEventSetGetContextInfo_v1)(
        set, params)


cdef nvmlReturn_t _nvmlEventSetGetContextData_v1(nvmlEventSet_t set, nvmlEventSetGetContextData_v1_t* params) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlEventSetGetContextData_v1
    _check_or_init_nvml()
    if __nvmlEventSetGetContextData_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlEventSetGetContextData_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlEventSet_t, nvmlEventSetGetContextData_v1_t*) noexcept nogil>__nvmlEventSetGetContextData_v1)(
        set, params)


cdef nvmlReturn_t _nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1(nvmlEventSet_t set, nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1_t* params) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1
    _check_or_init_nvml()
    if __nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlEventSet_t, nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1_t*) noexcept nogil>__nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1)(
        set, params)


cdef nvmlReturn_t _nvmlDeviceGetBankRemapperStatus_v1(nvmlDevice_t device, nvmlEccBankRemapperStatus_v1_t* pBankRemapperStatus) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetBankRemapperStatus_v1
    _check_or_init_nvml()
    if __nvmlDeviceGetBankRemapperStatus_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetBankRemapperStatus_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlEccBankRemapperStatus_v1_t*) noexcept nogil>__nvmlDeviceGetBankRemapperStatus_v1)(
        device, pBankRemapperStatus)


cdef nvmlReturn_t _nvmlDeviceGetGpuUptime_v1(nvmlDevice_t device, nvmlGpuUptime_v1_t* uptime) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetGpuUptime_v1
    _check_or_init_nvml()
    if __nvmlDeviceGetGpuUptime_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetGpuUptime_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlGpuUptime_v1_t*) noexcept nogil>__nvmlDeviceGetGpuUptime_v1)(
        device, uptime)


cdef nvmlReturn_t _nvmlSystemGetConfComputeSettings_v2(nvmlSystemConfComputeSettings_v2_t* settings) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlSystemGetConfComputeSettings_v2
    _check_or_init_nvml()
    if __nvmlSystemGetConfComputeSettings_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlSystemGetConfComputeSettings_v2 is not found")
    return (<nvmlReturn_t (*)(nvmlSystemConfComputeSettings_v2_t*) noexcept nogil>__nvmlSystemGetConfComputeSettings_v2)(
        settings)


cdef nvmlReturn_t _nvmlDeviceGetNvLinkInfo_v3(nvmlDevice_t device, nvmlNvLinkInfo_v3_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetNvLinkInfo_v3
    _check_or_init_nvml()
    if __nvmlDeviceGetNvLinkInfo_v3 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetNvLinkInfo_v3 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlNvLinkInfo_v3_t*) noexcept nogil>__nvmlDeviceGetNvLinkInfo_v3)(
        device, info)


cdef nvmlReturn_t _nvmlDeviceGetNvlinkRemapTableInfo_v1(nvmlDevice_t device, nvmlNvlinkGetRemapTableInfo_v1_t* remapTableInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetNvlinkRemapTableInfo_v1
    _check_or_init_nvml()
    if __nvmlDeviceGetNvlinkRemapTableInfo_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetNvlinkRemapTableInfo_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlNvlinkGetRemapTableInfo_v1_t*) noexcept nogil>__nvmlDeviceGetNvlinkRemapTableInfo_v1)(
        device, remapTableInfo)


cdef nvmlReturn_t _nvmlDeviceSetNvlinkNvleReady_v1(nvmlDevice_t device, nvmlNvlinkSetNvleReady_v1_t* setNvleReady) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetNvlinkNvleReady_v1
    _check_or_init_nvml()
    if __nvmlDeviceSetNvlinkNvleReady_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetNvlinkNvleReady_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlNvlinkSetNvleReady_v1_t*) noexcept nogil>__nvmlDeviceSetNvlinkNvleReady_v1)(
        device, setNvleReady)


cdef nvmlReturn_t _nvmlDeviceLockNvlinkRemapTable_v1(nvmlDevice_t device, nvmlNvlinkLockRemapTable_v1_t* remapTableLockInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceLockNvlinkRemapTable_v1
    _check_or_init_nvml()
    if __nvmlDeviceLockNvlinkRemapTable_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceLockNvlinkRemapTable_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlNvlinkLockRemapTable_v1_t*) noexcept nogil>__nvmlDeviceLockNvlinkRemapTable_v1)(
        device, remapTableLockInfo)


cdef nvmlReturn_t _nvmlDeviceSetupNvlinkNvleEncryptionKey_v1(nvmlDevice_t device, nvmlNvlinkSetupNvleEncryptionKey_v1_t* encryptionKey) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceSetupNvlinkNvleEncryptionKey_v1
    _check_or_init_nvml()
    if __nvmlDeviceSetupNvlinkNvleEncryptionKey_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceSetupNvlinkNvleEncryptionKey_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlNvlinkSetupNvleEncryptionKey_v1_t*) noexcept nogil>__nvmlDeviceSetupNvlinkNvleEncryptionKey_v1)(
        device, encryptionKey)


cdef nvmlReturn_t _nvmlDeviceGetNvlinkNvlePrivateInfo_v1(nvmlDevice_t device, nvmlNvlinkNvlePrivateInfo_v1_t* privateInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    global __nvmlDeviceGetNvlinkNvlePrivateInfo_v1
    _check_or_init_nvml()
    if __nvmlDeviceGetNvlinkNvlePrivateInfo_v1 == NULL:
        with gil:
            raise FunctionNotFoundError("function nvmlDeviceGetNvlinkNvlePrivateInfo_v1 is not found")
    return (<nvmlReturn_t (*)(nvmlDevice_t, nvmlNvlinkNvlePrivateInfo_v1_t*) noexcept nogil>__nvmlDeviceGetNvlinkNvlePrivateInfo_v1)(
        device, privateInfo)
