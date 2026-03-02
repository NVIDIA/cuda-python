# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.9.1 to 13.1.1, generator version 0.3.1.dev1322+g646ce84ec. Do not modify it directly.

from libc.stdint cimport intptr_t

import os
import threading

from .utils import FunctionNotFoundError, NotSupportedError

from libc.stddef cimport wchar_t
from libc.stdint cimport uintptr_t
from cpython cimport PyUnicode_AsWideCharString, PyMem_Free

# You must 'from .utils import NotSupportedError' before using this template

cdef extern from "windows.h" nogil:
    ctypedef void* HMODULE
    ctypedef void* HANDLE
    ctypedef void* FARPROC
    ctypedef unsigned long DWORD
    ctypedef const wchar_t *LPCWSTR
    ctypedef const char *LPCSTR

    cdef DWORD LOAD_LIBRARY_SEARCH_SYSTEM32 = 0x00000800
    cdef DWORD LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000
    cdef DWORD LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100

    HMODULE _LoadLibraryExW "LoadLibraryExW"(
        LPCWSTR lpLibFileName,
        HANDLE hFile,
        DWORD dwFlags
    )

    FARPROC _GetProcAddress "GetProcAddress"(HMODULE hModule, LPCSTR lpProcName)

cdef inline uintptr_t LoadLibraryExW(str path, HANDLE hFile, DWORD dwFlags):
    cdef uintptr_t result
    cdef wchar_t* wpath = PyUnicode_AsWideCharString(path, NULL)
    with nogil:
        result = <uintptr_t>_LoadLibraryExW(
            wpath,
            hFile,
            dwFlags
        )
    PyMem_Free(wpath)
    return result

cdef inline void *GetProcAddress(uintptr_t hModule, const char* lpProcName) nogil:
    return _GetProcAddress(<HMODULE>hModule, lpProcName)

cdef int get_cuda_version():
    cdef int err, driver_ver = 0

    # Load driver to check version
    handle = LoadLibraryExW("nvcuda.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32)
    if handle == 0:
        raise NotSupportedError('CUDA driver is not found')
    cuDriverGetVersion = GetProcAddress(handle, 'cuDriverGetVersion')
    if cuDriverGetVersion == NULL:
        raise RuntimeError('Did not find cuDriverGetVersion symbol in nvcuda.dll')
    err = (<int (*)(int*) noexcept nogil>cuDriverGetVersion)(&driver_ver)
    if err != 0:
        raise RuntimeError(f'cuDriverGetVersion returned error code {err}')

    return driver_ver



###############################################################################
# Wrapper init
###############################################################################

cdef object __symbol_lock = threading.Lock()
cdef bint __py_nvml_init = False

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


cdef uintptr_t load_library() except* with gil:
    def do_load(path):
        return LoadLibraryExW(
            path,
            <void *>0,
            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
        )

    handle = do_load(
        os.path.join(
            os.getenv("WINDIR", "C:/Windows"),
            "System32/nvml.dll"
        )
    )
    if handle:
        return handle

    handle = do_load(
        os.path.join(
            os.getenv("ProgramFiles", "C:/Program Files"),
            "NVIDIA Corporation/NVSMI/nvml.dll"
        )
    )
    if handle:
        return handle

    return 0


cdef int _init_nvml() except -1 nogil:
    global __py_nvml_init

    cdef int err, driver_ver = 0
    cdef uintptr_t handle

    with gil, __symbol_lock:
        handle = load_library()

        # Load function
        global __nvmlInit_v2
        __nvmlInit_v2 = GetProcAddress(handle, 'nvmlInit_v2')

        global __nvmlInitWithFlags
        __nvmlInitWithFlags = GetProcAddress(handle, 'nvmlInitWithFlags')

        global __nvmlShutdown
        __nvmlShutdown = GetProcAddress(handle, 'nvmlShutdown')

        global __nvmlErrorString
        __nvmlErrorString = GetProcAddress(handle, 'nvmlErrorString')

        global __nvmlSystemGetDriverVersion
        __nvmlSystemGetDriverVersion = GetProcAddress(handle, 'nvmlSystemGetDriverVersion')

        global __nvmlSystemGetNVMLVersion
        __nvmlSystemGetNVMLVersion = GetProcAddress(handle, 'nvmlSystemGetNVMLVersion')

        global __nvmlSystemGetCudaDriverVersion
        __nvmlSystemGetCudaDriverVersion = GetProcAddress(handle, 'nvmlSystemGetCudaDriverVersion')

        global __nvmlSystemGetCudaDriverVersion_v2
        __nvmlSystemGetCudaDriverVersion_v2 = GetProcAddress(handle, 'nvmlSystemGetCudaDriverVersion_v2')

        global __nvmlSystemGetProcessName
        __nvmlSystemGetProcessName = GetProcAddress(handle, 'nvmlSystemGetProcessName')

        global __nvmlSystemGetHicVersion
        __nvmlSystemGetHicVersion = GetProcAddress(handle, 'nvmlSystemGetHicVersion')

        global __nvmlSystemGetTopologyGpuSet
        __nvmlSystemGetTopologyGpuSet = GetProcAddress(handle, 'nvmlSystemGetTopologyGpuSet')

        global __nvmlSystemGetDriverBranch
        __nvmlSystemGetDriverBranch = GetProcAddress(handle, 'nvmlSystemGetDriverBranch')

        global __nvmlUnitGetCount
        __nvmlUnitGetCount = GetProcAddress(handle, 'nvmlUnitGetCount')

        global __nvmlUnitGetHandleByIndex
        __nvmlUnitGetHandleByIndex = GetProcAddress(handle, 'nvmlUnitGetHandleByIndex')

        global __nvmlUnitGetUnitInfo
        __nvmlUnitGetUnitInfo = GetProcAddress(handle, 'nvmlUnitGetUnitInfo')

        global __nvmlUnitGetLedState
        __nvmlUnitGetLedState = GetProcAddress(handle, 'nvmlUnitGetLedState')

        global __nvmlUnitGetPsuInfo
        __nvmlUnitGetPsuInfo = GetProcAddress(handle, 'nvmlUnitGetPsuInfo')

        global __nvmlUnitGetTemperature
        __nvmlUnitGetTemperature = GetProcAddress(handle, 'nvmlUnitGetTemperature')

        global __nvmlUnitGetFanSpeedInfo
        __nvmlUnitGetFanSpeedInfo = GetProcAddress(handle, 'nvmlUnitGetFanSpeedInfo')

        global __nvmlUnitGetDevices
        __nvmlUnitGetDevices = GetProcAddress(handle, 'nvmlUnitGetDevices')

        global __nvmlDeviceGetCount_v2
        __nvmlDeviceGetCount_v2 = GetProcAddress(handle, 'nvmlDeviceGetCount_v2')

        global __nvmlDeviceGetAttributes_v2
        __nvmlDeviceGetAttributes_v2 = GetProcAddress(handle, 'nvmlDeviceGetAttributes_v2')

        global __nvmlDeviceGetHandleByIndex_v2
        __nvmlDeviceGetHandleByIndex_v2 = GetProcAddress(handle, 'nvmlDeviceGetHandleByIndex_v2')

        global __nvmlDeviceGetHandleBySerial
        __nvmlDeviceGetHandleBySerial = GetProcAddress(handle, 'nvmlDeviceGetHandleBySerial')

        global __nvmlDeviceGetHandleByUUID
        __nvmlDeviceGetHandleByUUID = GetProcAddress(handle, 'nvmlDeviceGetHandleByUUID')

        global __nvmlDeviceGetHandleByUUIDV
        __nvmlDeviceGetHandleByUUIDV = GetProcAddress(handle, 'nvmlDeviceGetHandleByUUIDV')

        global __nvmlDeviceGetHandleByPciBusId_v2
        __nvmlDeviceGetHandleByPciBusId_v2 = GetProcAddress(handle, 'nvmlDeviceGetHandleByPciBusId_v2')

        global __nvmlDeviceGetName
        __nvmlDeviceGetName = GetProcAddress(handle, 'nvmlDeviceGetName')

        global __nvmlDeviceGetBrand
        __nvmlDeviceGetBrand = GetProcAddress(handle, 'nvmlDeviceGetBrand')

        global __nvmlDeviceGetIndex
        __nvmlDeviceGetIndex = GetProcAddress(handle, 'nvmlDeviceGetIndex')

        global __nvmlDeviceGetSerial
        __nvmlDeviceGetSerial = GetProcAddress(handle, 'nvmlDeviceGetSerial')

        global __nvmlDeviceGetModuleId
        __nvmlDeviceGetModuleId = GetProcAddress(handle, 'nvmlDeviceGetModuleId')

        global __nvmlDeviceGetC2cModeInfoV
        __nvmlDeviceGetC2cModeInfoV = GetProcAddress(handle, 'nvmlDeviceGetC2cModeInfoV')

        global __nvmlDeviceGetMemoryAffinity
        __nvmlDeviceGetMemoryAffinity = GetProcAddress(handle, 'nvmlDeviceGetMemoryAffinity')

        global __nvmlDeviceGetCpuAffinityWithinScope
        __nvmlDeviceGetCpuAffinityWithinScope = GetProcAddress(handle, 'nvmlDeviceGetCpuAffinityWithinScope')

        global __nvmlDeviceGetCpuAffinity
        __nvmlDeviceGetCpuAffinity = GetProcAddress(handle, 'nvmlDeviceGetCpuAffinity')

        global __nvmlDeviceSetCpuAffinity
        __nvmlDeviceSetCpuAffinity = GetProcAddress(handle, 'nvmlDeviceSetCpuAffinity')

        global __nvmlDeviceClearCpuAffinity
        __nvmlDeviceClearCpuAffinity = GetProcAddress(handle, 'nvmlDeviceClearCpuAffinity')

        global __nvmlDeviceGetNumaNodeId
        __nvmlDeviceGetNumaNodeId = GetProcAddress(handle, 'nvmlDeviceGetNumaNodeId')

        global __nvmlDeviceGetTopologyCommonAncestor
        __nvmlDeviceGetTopologyCommonAncestor = GetProcAddress(handle, 'nvmlDeviceGetTopologyCommonAncestor')

        global __nvmlDeviceGetTopologyNearestGpus
        __nvmlDeviceGetTopologyNearestGpus = GetProcAddress(handle, 'nvmlDeviceGetTopologyNearestGpus')

        global __nvmlDeviceGetP2PStatus
        __nvmlDeviceGetP2PStatus = GetProcAddress(handle, 'nvmlDeviceGetP2PStatus')

        global __nvmlDeviceGetUUID
        __nvmlDeviceGetUUID = GetProcAddress(handle, 'nvmlDeviceGetUUID')

        global __nvmlDeviceGetMinorNumber
        __nvmlDeviceGetMinorNumber = GetProcAddress(handle, 'nvmlDeviceGetMinorNumber')

        global __nvmlDeviceGetBoardPartNumber
        __nvmlDeviceGetBoardPartNumber = GetProcAddress(handle, 'nvmlDeviceGetBoardPartNumber')

        global __nvmlDeviceGetInforomVersion
        __nvmlDeviceGetInforomVersion = GetProcAddress(handle, 'nvmlDeviceGetInforomVersion')

        global __nvmlDeviceGetInforomImageVersion
        __nvmlDeviceGetInforomImageVersion = GetProcAddress(handle, 'nvmlDeviceGetInforomImageVersion')

        global __nvmlDeviceGetInforomConfigurationChecksum
        __nvmlDeviceGetInforomConfigurationChecksum = GetProcAddress(handle, 'nvmlDeviceGetInforomConfigurationChecksum')

        global __nvmlDeviceValidateInforom
        __nvmlDeviceValidateInforom = GetProcAddress(handle, 'nvmlDeviceValidateInforom')

        global __nvmlDeviceGetLastBBXFlushTime
        __nvmlDeviceGetLastBBXFlushTime = GetProcAddress(handle, 'nvmlDeviceGetLastBBXFlushTime')

        global __nvmlDeviceGetDisplayMode
        __nvmlDeviceGetDisplayMode = GetProcAddress(handle, 'nvmlDeviceGetDisplayMode')

        global __nvmlDeviceGetDisplayActive
        __nvmlDeviceGetDisplayActive = GetProcAddress(handle, 'nvmlDeviceGetDisplayActive')

        global __nvmlDeviceGetPersistenceMode
        __nvmlDeviceGetPersistenceMode = GetProcAddress(handle, 'nvmlDeviceGetPersistenceMode')

        global __nvmlDeviceGetPciInfoExt
        __nvmlDeviceGetPciInfoExt = GetProcAddress(handle, 'nvmlDeviceGetPciInfoExt')

        global __nvmlDeviceGetPciInfo_v3
        __nvmlDeviceGetPciInfo_v3 = GetProcAddress(handle, 'nvmlDeviceGetPciInfo_v3')

        global __nvmlDeviceGetMaxPcieLinkGeneration
        __nvmlDeviceGetMaxPcieLinkGeneration = GetProcAddress(handle, 'nvmlDeviceGetMaxPcieLinkGeneration')

        global __nvmlDeviceGetGpuMaxPcieLinkGeneration
        __nvmlDeviceGetGpuMaxPcieLinkGeneration = GetProcAddress(handle, 'nvmlDeviceGetGpuMaxPcieLinkGeneration')

        global __nvmlDeviceGetMaxPcieLinkWidth
        __nvmlDeviceGetMaxPcieLinkWidth = GetProcAddress(handle, 'nvmlDeviceGetMaxPcieLinkWidth')

        global __nvmlDeviceGetCurrPcieLinkGeneration
        __nvmlDeviceGetCurrPcieLinkGeneration = GetProcAddress(handle, 'nvmlDeviceGetCurrPcieLinkGeneration')

        global __nvmlDeviceGetCurrPcieLinkWidth
        __nvmlDeviceGetCurrPcieLinkWidth = GetProcAddress(handle, 'nvmlDeviceGetCurrPcieLinkWidth')

        global __nvmlDeviceGetPcieThroughput
        __nvmlDeviceGetPcieThroughput = GetProcAddress(handle, 'nvmlDeviceGetPcieThroughput')

        global __nvmlDeviceGetPcieReplayCounter
        __nvmlDeviceGetPcieReplayCounter = GetProcAddress(handle, 'nvmlDeviceGetPcieReplayCounter')

        global __nvmlDeviceGetClockInfo
        __nvmlDeviceGetClockInfo = GetProcAddress(handle, 'nvmlDeviceGetClockInfo')

        global __nvmlDeviceGetMaxClockInfo
        __nvmlDeviceGetMaxClockInfo = GetProcAddress(handle, 'nvmlDeviceGetMaxClockInfo')

        global __nvmlDeviceGetGpcClkVfOffset
        __nvmlDeviceGetGpcClkVfOffset = GetProcAddress(handle, 'nvmlDeviceGetGpcClkVfOffset')

        global __nvmlDeviceGetClock
        __nvmlDeviceGetClock = GetProcAddress(handle, 'nvmlDeviceGetClock')

        global __nvmlDeviceGetMaxCustomerBoostClock
        __nvmlDeviceGetMaxCustomerBoostClock = GetProcAddress(handle, 'nvmlDeviceGetMaxCustomerBoostClock')

        global __nvmlDeviceGetSupportedMemoryClocks
        __nvmlDeviceGetSupportedMemoryClocks = GetProcAddress(handle, 'nvmlDeviceGetSupportedMemoryClocks')

        global __nvmlDeviceGetSupportedGraphicsClocks
        __nvmlDeviceGetSupportedGraphicsClocks = GetProcAddress(handle, 'nvmlDeviceGetSupportedGraphicsClocks')

        global __nvmlDeviceGetAutoBoostedClocksEnabled
        __nvmlDeviceGetAutoBoostedClocksEnabled = GetProcAddress(handle, 'nvmlDeviceGetAutoBoostedClocksEnabled')

        global __nvmlDeviceGetFanSpeed
        __nvmlDeviceGetFanSpeed = GetProcAddress(handle, 'nvmlDeviceGetFanSpeed')

        global __nvmlDeviceGetFanSpeed_v2
        __nvmlDeviceGetFanSpeed_v2 = GetProcAddress(handle, 'nvmlDeviceGetFanSpeed_v2')

        global __nvmlDeviceGetFanSpeedRPM
        __nvmlDeviceGetFanSpeedRPM = GetProcAddress(handle, 'nvmlDeviceGetFanSpeedRPM')

        global __nvmlDeviceGetTargetFanSpeed
        __nvmlDeviceGetTargetFanSpeed = GetProcAddress(handle, 'nvmlDeviceGetTargetFanSpeed')

        global __nvmlDeviceGetMinMaxFanSpeed
        __nvmlDeviceGetMinMaxFanSpeed = GetProcAddress(handle, 'nvmlDeviceGetMinMaxFanSpeed')

        global __nvmlDeviceGetFanControlPolicy_v2
        __nvmlDeviceGetFanControlPolicy_v2 = GetProcAddress(handle, 'nvmlDeviceGetFanControlPolicy_v2')

        global __nvmlDeviceGetNumFans
        __nvmlDeviceGetNumFans = GetProcAddress(handle, 'nvmlDeviceGetNumFans')

        global __nvmlDeviceGetCoolerInfo
        __nvmlDeviceGetCoolerInfo = GetProcAddress(handle, 'nvmlDeviceGetCoolerInfo')

        global __nvmlDeviceGetTemperatureV
        __nvmlDeviceGetTemperatureV = GetProcAddress(handle, 'nvmlDeviceGetTemperatureV')

        global __nvmlDeviceGetTemperatureThreshold
        __nvmlDeviceGetTemperatureThreshold = GetProcAddress(handle, 'nvmlDeviceGetTemperatureThreshold')

        global __nvmlDeviceGetMarginTemperature
        __nvmlDeviceGetMarginTemperature = GetProcAddress(handle, 'nvmlDeviceGetMarginTemperature')

        global __nvmlDeviceGetThermalSettings
        __nvmlDeviceGetThermalSettings = GetProcAddress(handle, 'nvmlDeviceGetThermalSettings')

        global __nvmlDeviceGetPerformanceState
        __nvmlDeviceGetPerformanceState = GetProcAddress(handle, 'nvmlDeviceGetPerformanceState')

        global __nvmlDeviceGetCurrentClocksEventReasons
        __nvmlDeviceGetCurrentClocksEventReasons = GetProcAddress(handle, 'nvmlDeviceGetCurrentClocksEventReasons')

        global __nvmlDeviceGetSupportedClocksEventReasons
        __nvmlDeviceGetSupportedClocksEventReasons = GetProcAddress(handle, 'nvmlDeviceGetSupportedClocksEventReasons')

        global __nvmlDeviceGetPowerState
        __nvmlDeviceGetPowerState = GetProcAddress(handle, 'nvmlDeviceGetPowerState')

        global __nvmlDeviceGetDynamicPstatesInfo
        __nvmlDeviceGetDynamicPstatesInfo = GetProcAddress(handle, 'nvmlDeviceGetDynamicPstatesInfo')

        global __nvmlDeviceGetMemClkVfOffset
        __nvmlDeviceGetMemClkVfOffset = GetProcAddress(handle, 'nvmlDeviceGetMemClkVfOffset')

        global __nvmlDeviceGetMinMaxClockOfPState
        __nvmlDeviceGetMinMaxClockOfPState = GetProcAddress(handle, 'nvmlDeviceGetMinMaxClockOfPState')

        global __nvmlDeviceGetSupportedPerformanceStates
        __nvmlDeviceGetSupportedPerformanceStates = GetProcAddress(handle, 'nvmlDeviceGetSupportedPerformanceStates')

        global __nvmlDeviceGetGpcClkMinMaxVfOffset
        __nvmlDeviceGetGpcClkMinMaxVfOffset = GetProcAddress(handle, 'nvmlDeviceGetGpcClkMinMaxVfOffset')

        global __nvmlDeviceGetMemClkMinMaxVfOffset
        __nvmlDeviceGetMemClkMinMaxVfOffset = GetProcAddress(handle, 'nvmlDeviceGetMemClkMinMaxVfOffset')

        global __nvmlDeviceGetClockOffsets
        __nvmlDeviceGetClockOffsets = GetProcAddress(handle, 'nvmlDeviceGetClockOffsets')

        global __nvmlDeviceSetClockOffsets
        __nvmlDeviceSetClockOffsets = GetProcAddress(handle, 'nvmlDeviceSetClockOffsets')

        global __nvmlDeviceGetPerformanceModes
        __nvmlDeviceGetPerformanceModes = GetProcAddress(handle, 'nvmlDeviceGetPerformanceModes')

        global __nvmlDeviceGetCurrentClockFreqs
        __nvmlDeviceGetCurrentClockFreqs = GetProcAddress(handle, 'nvmlDeviceGetCurrentClockFreqs')

        global __nvmlDeviceGetPowerManagementLimit
        __nvmlDeviceGetPowerManagementLimit = GetProcAddress(handle, 'nvmlDeviceGetPowerManagementLimit')

        global __nvmlDeviceGetPowerManagementLimitConstraints
        __nvmlDeviceGetPowerManagementLimitConstraints = GetProcAddress(handle, 'nvmlDeviceGetPowerManagementLimitConstraints')

        global __nvmlDeviceGetPowerManagementDefaultLimit
        __nvmlDeviceGetPowerManagementDefaultLimit = GetProcAddress(handle, 'nvmlDeviceGetPowerManagementDefaultLimit')

        global __nvmlDeviceGetPowerUsage
        __nvmlDeviceGetPowerUsage = GetProcAddress(handle, 'nvmlDeviceGetPowerUsage')

        global __nvmlDeviceGetTotalEnergyConsumption
        __nvmlDeviceGetTotalEnergyConsumption = GetProcAddress(handle, 'nvmlDeviceGetTotalEnergyConsumption')

        global __nvmlDeviceGetEnforcedPowerLimit
        __nvmlDeviceGetEnforcedPowerLimit = GetProcAddress(handle, 'nvmlDeviceGetEnforcedPowerLimit')

        global __nvmlDeviceGetGpuOperationMode
        __nvmlDeviceGetGpuOperationMode = GetProcAddress(handle, 'nvmlDeviceGetGpuOperationMode')

        global __nvmlDeviceGetMemoryInfo_v2
        __nvmlDeviceGetMemoryInfo_v2 = GetProcAddress(handle, 'nvmlDeviceGetMemoryInfo_v2')

        global __nvmlDeviceGetComputeMode
        __nvmlDeviceGetComputeMode = GetProcAddress(handle, 'nvmlDeviceGetComputeMode')

        global __nvmlDeviceGetCudaComputeCapability
        __nvmlDeviceGetCudaComputeCapability = GetProcAddress(handle, 'nvmlDeviceGetCudaComputeCapability')

        global __nvmlDeviceGetDramEncryptionMode
        __nvmlDeviceGetDramEncryptionMode = GetProcAddress(handle, 'nvmlDeviceGetDramEncryptionMode')

        global __nvmlDeviceSetDramEncryptionMode
        __nvmlDeviceSetDramEncryptionMode = GetProcAddress(handle, 'nvmlDeviceSetDramEncryptionMode')

        global __nvmlDeviceGetEccMode
        __nvmlDeviceGetEccMode = GetProcAddress(handle, 'nvmlDeviceGetEccMode')

        global __nvmlDeviceGetDefaultEccMode
        __nvmlDeviceGetDefaultEccMode = GetProcAddress(handle, 'nvmlDeviceGetDefaultEccMode')

        global __nvmlDeviceGetBoardId
        __nvmlDeviceGetBoardId = GetProcAddress(handle, 'nvmlDeviceGetBoardId')

        global __nvmlDeviceGetMultiGpuBoard
        __nvmlDeviceGetMultiGpuBoard = GetProcAddress(handle, 'nvmlDeviceGetMultiGpuBoard')

        global __nvmlDeviceGetTotalEccErrors
        __nvmlDeviceGetTotalEccErrors = GetProcAddress(handle, 'nvmlDeviceGetTotalEccErrors')

        global __nvmlDeviceGetMemoryErrorCounter
        __nvmlDeviceGetMemoryErrorCounter = GetProcAddress(handle, 'nvmlDeviceGetMemoryErrorCounter')

        global __nvmlDeviceGetUtilizationRates
        __nvmlDeviceGetUtilizationRates = GetProcAddress(handle, 'nvmlDeviceGetUtilizationRates')

        global __nvmlDeviceGetEncoderUtilization
        __nvmlDeviceGetEncoderUtilization = GetProcAddress(handle, 'nvmlDeviceGetEncoderUtilization')

        global __nvmlDeviceGetEncoderCapacity
        __nvmlDeviceGetEncoderCapacity = GetProcAddress(handle, 'nvmlDeviceGetEncoderCapacity')

        global __nvmlDeviceGetEncoderStats
        __nvmlDeviceGetEncoderStats = GetProcAddress(handle, 'nvmlDeviceGetEncoderStats')

        global __nvmlDeviceGetEncoderSessions
        __nvmlDeviceGetEncoderSessions = GetProcAddress(handle, 'nvmlDeviceGetEncoderSessions')

        global __nvmlDeviceGetDecoderUtilization
        __nvmlDeviceGetDecoderUtilization = GetProcAddress(handle, 'nvmlDeviceGetDecoderUtilization')

        global __nvmlDeviceGetJpgUtilization
        __nvmlDeviceGetJpgUtilization = GetProcAddress(handle, 'nvmlDeviceGetJpgUtilization')

        global __nvmlDeviceGetOfaUtilization
        __nvmlDeviceGetOfaUtilization = GetProcAddress(handle, 'nvmlDeviceGetOfaUtilization')

        global __nvmlDeviceGetFBCStats
        __nvmlDeviceGetFBCStats = GetProcAddress(handle, 'nvmlDeviceGetFBCStats')

        global __nvmlDeviceGetFBCSessions
        __nvmlDeviceGetFBCSessions = GetProcAddress(handle, 'nvmlDeviceGetFBCSessions')

        global __nvmlDeviceGetDriverModel_v2
        __nvmlDeviceGetDriverModel_v2 = GetProcAddress(handle, 'nvmlDeviceGetDriverModel_v2')

        global __nvmlDeviceGetVbiosVersion
        __nvmlDeviceGetVbiosVersion = GetProcAddress(handle, 'nvmlDeviceGetVbiosVersion')

        global __nvmlDeviceGetBridgeChipInfo
        __nvmlDeviceGetBridgeChipInfo = GetProcAddress(handle, 'nvmlDeviceGetBridgeChipInfo')

        global __nvmlDeviceGetComputeRunningProcesses_v3
        __nvmlDeviceGetComputeRunningProcesses_v3 = GetProcAddress(handle, 'nvmlDeviceGetComputeRunningProcesses_v3')

        global __nvmlDeviceGetMPSComputeRunningProcesses_v3
        __nvmlDeviceGetMPSComputeRunningProcesses_v3 = GetProcAddress(handle, 'nvmlDeviceGetMPSComputeRunningProcesses_v3')

        global __nvmlDeviceGetRunningProcessDetailList
        __nvmlDeviceGetRunningProcessDetailList = GetProcAddress(handle, 'nvmlDeviceGetRunningProcessDetailList')

        global __nvmlDeviceOnSameBoard
        __nvmlDeviceOnSameBoard = GetProcAddress(handle, 'nvmlDeviceOnSameBoard')

        global __nvmlDeviceGetAPIRestriction
        __nvmlDeviceGetAPIRestriction = GetProcAddress(handle, 'nvmlDeviceGetAPIRestriction')

        global __nvmlDeviceGetSamples
        __nvmlDeviceGetSamples = GetProcAddress(handle, 'nvmlDeviceGetSamples')

        global __nvmlDeviceGetBAR1MemoryInfo
        __nvmlDeviceGetBAR1MemoryInfo = GetProcAddress(handle, 'nvmlDeviceGetBAR1MemoryInfo')

        global __nvmlDeviceGetIrqNum
        __nvmlDeviceGetIrqNum = GetProcAddress(handle, 'nvmlDeviceGetIrqNum')

        global __nvmlDeviceGetNumGpuCores
        __nvmlDeviceGetNumGpuCores = GetProcAddress(handle, 'nvmlDeviceGetNumGpuCores')

        global __nvmlDeviceGetPowerSource
        __nvmlDeviceGetPowerSource = GetProcAddress(handle, 'nvmlDeviceGetPowerSource')

        global __nvmlDeviceGetMemoryBusWidth
        __nvmlDeviceGetMemoryBusWidth = GetProcAddress(handle, 'nvmlDeviceGetMemoryBusWidth')

        global __nvmlDeviceGetPcieLinkMaxSpeed
        __nvmlDeviceGetPcieLinkMaxSpeed = GetProcAddress(handle, 'nvmlDeviceGetPcieLinkMaxSpeed')

        global __nvmlDeviceGetPcieSpeed
        __nvmlDeviceGetPcieSpeed = GetProcAddress(handle, 'nvmlDeviceGetPcieSpeed')

        global __nvmlDeviceGetAdaptiveClockInfoStatus
        __nvmlDeviceGetAdaptiveClockInfoStatus = GetProcAddress(handle, 'nvmlDeviceGetAdaptiveClockInfoStatus')

        global __nvmlDeviceGetBusType
        __nvmlDeviceGetBusType = GetProcAddress(handle, 'nvmlDeviceGetBusType')

        global __nvmlDeviceGetGpuFabricInfoV
        __nvmlDeviceGetGpuFabricInfoV = GetProcAddress(handle, 'nvmlDeviceGetGpuFabricInfoV')

        global __nvmlSystemGetConfComputeCapabilities
        __nvmlSystemGetConfComputeCapabilities = GetProcAddress(handle, 'nvmlSystemGetConfComputeCapabilities')

        global __nvmlSystemGetConfComputeState
        __nvmlSystemGetConfComputeState = GetProcAddress(handle, 'nvmlSystemGetConfComputeState')

        global __nvmlDeviceGetConfComputeMemSizeInfo
        __nvmlDeviceGetConfComputeMemSizeInfo = GetProcAddress(handle, 'nvmlDeviceGetConfComputeMemSizeInfo')

        global __nvmlSystemGetConfComputeGpusReadyState
        __nvmlSystemGetConfComputeGpusReadyState = GetProcAddress(handle, 'nvmlSystemGetConfComputeGpusReadyState')

        global __nvmlDeviceGetConfComputeProtectedMemoryUsage
        __nvmlDeviceGetConfComputeProtectedMemoryUsage = GetProcAddress(handle, 'nvmlDeviceGetConfComputeProtectedMemoryUsage')

        global __nvmlDeviceGetConfComputeGpuCertificate
        __nvmlDeviceGetConfComputeGpuCertificate = GetProcAddress(handle, 'nvmlDeviceGetConfComputeGpuCertificate')

        global __nvmlDeviceGetConfComputeGpuAttestationReport
        __nvmlDeviceGetConfComputeGpuAttestationReport = GetProcAddress(handle, 'nvmlDeviceGetConfComputeGpuAttestationReport')

        global __nvmlSystemGetConfComputeKeyRotationThresholdInfo
        __nvmlSystemGetConfComputeKeyRotationThresholdInfo = GetProcAddress(handle, 'nvmlSystemGetConfComputeKeyRotationThresholdInfo')

        global __nvmlDeviceSetConfComputeUnprotectedMemSize
        __nvmlDeviceSetConfComputeUnprotectedMemSize = GetProcAddress(handle, 'nvmlDeviceSetConfComputeUnprotectedMemSize')

        global __nvmlSystemSetConfComputeGpusReadyState
        __nvmlSystemSetConfComputeGpusReadyState = GetProcAddress(handle, 'nvmlSystemSetConfComputeGpusReadyState')

        global __nvmlSystemSetConfComputeKeyRotationThresholdInfo
        __nvmlSystemSetConfComputeKeyRotationThresholdInfo = GetProcAddress(handle, 'nvmlSystemSetConfComputeKeyRotationThresholdInfo')

        global __nvmlSystemGetConfComputeSettings
        __nvmlSystemGetConfComputeSettings = GetProcAddress(handle, 'nvmlSystemGetConfComputeSettings')

        global __nvmlDeviceGetGspFirmwareVersion
        __nvmlDeviceGetGspFirmwareVersion = GetProcAddress(handle, 'nvmlDeviceGetGspFirmwareVersion')

        global __nvmlDeviceGetGspFirmwareMode
        __nvmlDeviceGetGspFirmwareMode = GetProcAddress(handle, 'nvmlDeviceGetGspFirmwareMode')

        global __nvmlDeviceGetSramEccErrorStatus
        __nvmlDeviceGetSramEccErrorStatus = GetProcAddress(handle, 'nvmlDeviceGetSramEccErrorStatus')

        global __nvmlDeviceGetAccountingMode
        __nvmlDeviceGetAccountingMode = GetProcAddress(handle, 'nvmlDeviceGetAccountingMode')

        global __nvmlDeviceGetAccountingStats
        __nvmlDeviceGetAccountingStats = GetProcAddress(handle, 'nvmlDeviceGetAccountingStats')

        global __nvmlDeviceGetAccountingPids
        __nvmlDeviceGetAccountingPids = GetProcAddress(handle, 'nvmlDeviceGetAccountingPids')

        global __nvmlDeviceGetAccountingBufferSize
        __nvmlDeviceGetAccountingBufferSize = GetProcAddress(handle, 'nvmlDeviceGetAccountingBufferSize')

        global __nvmlDeviceGetRetiredPages
        __nvmlDeviceGetRetiredPages = GetProcAddress(handle, 'nvmlDeviceGetRetiredPages')

        global __nvmlDeviceGetRetiredPages_v2
        __nvmlDeviceGetRetiredPages_v2 = GetProcAddress(handle, 'nvmlDeviceGetRetiredPages_v2')

        global __nvmlDeviceGetRetiredPagesPendingStatus
        __nvmlDeviceGetRetiredPagesPendingStatus = GetProcAddress(handle, 'nvmlDeviceGetRetiredPagesPendingStatus')

        global __nvmlDeviceGetRemappedRows
        __nvmlDeviceGetRemappedRows = GetProcAddress(handle, 'nvmlDeviceGetRemappedRows')

        global __nvmlDeviceGetRowRemapperHistogram
        __nvmlDeviceGetRowRemapperHistogram = GetProcAddress(handle, 'nvmlDeviceGetRowRemapperHistogram')

        global __nvmlDeviceGetArchitecture
        __nvmlDeviceGetArchitecture = GetProcAddress(handle, 'nvmlDeviceGetArchitecture')

        global __nvmlDeviceGetClkMonStatus
        __nvmlDeviceGetClkMonStatus = GetProcAddress(handle, 'nvmlDeviceGetClkMonStatus')

        global __nvmlDeviceGetProcessUtilization
        __nvmlDeviceGetProcessUtilization = GetProcAddress(handle, 'nvmlDeviceGetProcessUtilization')

        global __nvmlDeviceGetProcessesUtilizationInfo
        __nvmlDeviceGetProcessesUtilizationInfo = GetProcAddress(handle, 'nvmlDeviceGetProcessesUtilizationInfo')

        global __nvmlDeviceGetPlatformInfo
        __nvmlDeviceGetPlatformInfo = GetProcAddress(handle, 'nvmlDeviceGetPlatformInfo')

        global __nvmlUnitSetLedState
        __nvmlUnitSetLedState = GetProcAddress(handle, 'nvmlUnitSetLedState')

        global __nvmlDeviceSetPersistenceMode
        __nvmlDeviceSetPersistenceMode = GetProcAddress(handle, 'nvmlDeviceSetPersistenceMode')

        global __nvmlDeviceSetComputeMode
        __nvmlDeviceSetComputeMode = GetProcAddress(handle, 'nvmlDeviceSetComputeMode')

        global __nvmlDeviceSetEccMode
        __nvmlDeviceSetEccMode = GetProcAddress(handle, 'nvmlDeviceSetEccMode')

        global __nvmlDeviceClearEccErrorCounts
        __nvmlDeviceClearEccErrorCounts = GetProcAddress(handle, 'nvmlDeviceClearEccErrorCounts')

        global __nvmlDeviceSetDriverModel
        __nvmlDeviceSetDriverModel = GetProcAddress(handle, 'nvmlDeviceSetDriverModel')

        global __nvmlDeviceSetGpuLockedClocks
        __nvmlDeviceSetGpuLockedClocks = GetProcAddress(handle, 'nvmlDeviceSetGpuLockedClocks')

        global __nvmlDeviceResetGpuLockedClocks
        __nvmlDeviceResetGpuLockedClocks = GetProcAddress(handle, 'nvmlDeviceResetGpuLockedClocks')

        global __nvmlDeviceSetMemoryLockedClocks
        __nvmlDeviceSetMemoryLockedClocks = GetProcAddress(handle, 'nvmlDeviceSetMemoryLockedClocks')

        global __nvmlDeviceResetMemoryLockedClocks
        __nvmlDeviceResetMemoryLockedClocks = GetProcAddress(handle, 'nvmlDeviceResetMemoryLockedClocks')

        global __nvmlDeviceSetAutoBoostedClocksEnabled
        __nvmlDeviceSetAutoBoostedClocksEnabled = GetProcAddress(handle, 'nvmlDeviceSetAutoBoostedClocksEnabled')

        global __nvmlDeviceSetDefaultAutoBoostedClocksEnabled
        __nvmlDeviceSetDefaultAutoBoostedClocksEnabled = GetProcAddress(handle, 'nvmlDeviceSetDefaultAutoBoostedClocksEnabled')

        global __nvmlDeviceSetDefaultFanSpeed_v2
        __nvmlDeviceSetDefaultFanSpeed_v2 = GetProcAddress(handle, 'nvmlDeviceSetDefaultFanSpeed_v2')

        global __nvmlDeviceSetFanControlPolicy
        __nvmlDeviceSetFanControlPolicy = GetProcAddress(handle, 'nvmlDeviceSetFanControlPolicy')

        global __nvmlDeviceSetTemperatureThreshold
        __nvmlDeviceSetTemperatureThreshold = GetProcAddress(handle, 'nvmlDeviceSetTemperatureThreshold')

        global __nvmlDeviceSetGpuOperationMode
        __nvmlDeviceSetGpuOperationMode = GetProcAddress(handle, 'nvmlDeviceSetGpuOperationMode')

        global __nvmlDeviceSetAPIRestriction
        __nvmlDeviceSetAPIRestriction = GetProcAddress(handle, 'nvmlDeviceSetAPIRestriction')

        global __nvmlDeviceSetFanSpeed_v2
        __nvmlDeviceSetFanSpeed_v2 = GetProcAddress(handle, 'nvmlDeviceSetFanSpeed_v2')

        global __nvmlDeviceSetAccountingMode
        __nvmlDeviceSetAccountingMode = GetProcAddress(handle, 'nvmlDeviceSetAccountingMode')

        global __nvmlDeviceClearAccountingPids
        __nvmlDeviceClearAccountingPids = GetProcAddress(handle, 'nvmlDeviceClearAccountingPids')

        global __nvmlDeviceSetPowerManagementLimit_v2
        __nvmlDeviceSetPowerManagementLimit_v2 = GetProcAddress(handle, 'nvmlDeviceSetPowerManagementLimit_v2')

        global __nvmlDeviceGetNvLinkState
        __nvmlDeviceGetNvLinkState = GetProcAddress(handle, 'nvmlDeviceGetNvLinkState')

        global __nvmlDeviceGetNvLinkVersion
        __nvmlDeviceGetNvLinkVersion = GetProcAddress(handle, 'nvmlDeviceGetNvLinkVersion')

        global __nvmlDeviceGetNvLinkCapability
        __nvmlDeviceGetNvLinkCapability = GetProcAddress(handle, 'nvmlDeviceGetNvLinkCapability')

        global __nvmlDeviceGetNvLinkRemotePciInfo_v2
        __nvmlDeviceGetNvLinkRemotePciInfo_v2 = GetProcAddress(handle, 'nvmlDeviceGetNvLinkRemotePciInfo_v2')

        global __nvmlDeviceGetNvLinkErrorCounter
        __nvmlDeviceGetNvLinkErrorCounter = GetProcAddress(handle, 'nvmlDeviceGetNvLinkErrorCounter')

        global __nvmlDeviceResetNvLinkErrorCounters
        __nvmlDeviceResetNvLinkErrorCounters = GetProcAddress(handle, 'nvmlDeviceResetNvLinkErrorCounters')

        global __nvmlDeviceGetNvLinkRemoteDeviceType
        __nvmlDeviceGetNvLinkRemoteDeviceType = GetProcAddress(handle, 'nvmlDeviceGetNvLinkRemoteDeviceType')

        global __nvmlDeviceSetNvLinkDeviceLowPowerThreshold
        __nvmlDeviceSetNvLinkDeviceLowPowerThreshold = GetProcAddress(handle, 'nvmlDeviceSetNvLinkDeviceLowPowerThreshold')

        global __nvmlSystemSetNvlinkBwMode
        __nvmlSystemSetNvlinkBwMode = GetProcAddress(handle, 'nvmlSystemSetNvlinkBwMode')

        global __nvmlSystemGetNvlinkBwMode
        __nvmlSystemGetNvlinkBwMode = GetProcAddress(handle, 'nvmlSystemGetNvlinkBwMode')

        global __nvmlDeviceGetNvlinkSupportedBwModes
        __nvmlDeviceGetNvlinkSupportedBwModes = GetProcAddress(handle, 'nvmlDeviceGetNvlinkSupportedBwModes')

        global __nvmlDeviceGetNvlinkBwMode
        __nvmlDeviceGetNvlinkBwMode = GetProcAddress(handle, 'nvmlDeviceGetNvlinkBwMode')

        global __nvmlDeviceSetNvlinkBwMode
        __nvmlDeviceSetNvlinkBwMode = GetProcAddress(handle, 'nvmlDeviceSetNvlinkBwMode')

        global __nvmlEventSetCreate
        __nvmlEventSetCreate = GetProcAddress(handle, 'nvmlEventSetCreate')

        global __nvmlDeviceRegisterEvents
        __nvmlDeviceRegisterEvents = GetProcAddress(handle, 'nvmlDeviceRegisterEvents')

        global __nvmlDeviceGetSupportedEventTypes
        __nvmlDeviceGetSupportedEventTypes = GetProcAddress(handle, 'nvmlDeviceGetSupportedEventTypes')

        global __nvmlEventSetWait_v2
        __nvmlEventSetWait_v2 = GetProcAddress(handle, 'nvmlEventSetWait_v2')

        global __nvmlEventSetFree
        __nvmlEventSetFree = GetProcAddress(handle, 'nvmlEventSetFree')

        global __nvmlSystemEventSetCreate
        __nvmlSystemEventSetCreate = GetProcAddress(handle, 'nvmlSystemEventSetCreate')

        global __nvmlSystemEventSetFree
        __nvmlSystemEventSetFree = GetProcAddress(handle, 'nvmlSystemEventSetFree')

        global __nvmlSystemRegisterEvents
        __nvmlSystemRegisterEvents = GetProcAddress(handle, 'nvmlSystemRegisterEvents')

        global __nvmlSystemEventSetWait
        __nvmlSystemEventSetWait = GetProcAddress(handle, 'nvmlSystemEventSetWait')

        global __nvmlDeviceModifyDrainState
        __nvmlDeviceModifyDrainState = GetProcAddress(handle, 'nvmlDeviceModifyDrainState')

        global __nvmlDeviceQueryDrainState
        __nvmlDeviceQueryDrainState = GetProcAddress(handle, 'nvmlDeviceQueryDrainState')

        global __nvmlDeviceRemoveGpu_v2
        __nvmlDeviceRemoveGpu_v2 = GetProcAddress(handle, 'nvmlDeviceRemoveGpu_v2')

        global __nvmlDeviceDiscoverGpus
        __nvmlDeviceDiscoverGpus = GetProcAddress(handle, 'nvmlDeviceDiscoverGpus')

        global __nvmlDeviceGetFieldValues
        __nvmlDeviceGetFieldValues = GetProcAddress(handle, 'nvmlDeviceGetFieldValues')

        global __nvmlDeviceClearFieldValues
        __nvmlDeviceClearFieldValues = GetProcAddress(handle, 'nvmlDeviceClearFieldValues')

        global __nvmlDeviceGetVirtualizationMode
        __nvmlDeviceGetVirtualizationMode = GetProcAddress(handle, 'nvmlDeviceGetVirtualizationMode')

        global __nvmlDeviceGetHostVgpuMode
        __nvmlDeviceGetHostVgpuMode = GetProcAddress(handle, 'nvmlDeviceGetHostVgpuMode')

        global __nvmlDeviceSetVirtualizationMode
        __nvmlDeviceSetVirtualizationMode = GetProcAddress(handle, 'nvmlDeviceSetVirtualizationMode')

        global __nvmlDeviceGetVgpuHeterogeneousMode
        __nvmlDeviceGetVgpuHeterogeneousMode = GetProcAddress(handle, 'nvmlDeviceGetVgpuHeterogeneousMode')

        global __nvmlDeviceSetVgpuHeterogeneousMode
        __nvmlDeviceSetVgpuHeterogeneousMode = GetProcAddress(handle, 'nvmlDeviceSetVgpuHeterogeneousMode')

        global __nvmlVgpuInstanceGetPlacementId
        __nvmlVgpuInstanceGetPlacementId = GetProcAddress(handle, 'nvmlVgpuInstanceGetPlacementId')

        global __nvmlDeviceGetVgpuTypeSupportedPlacements
        __nvmlDeviceGetVgpuTypeSupportedPlacements = GetProcAddress(handle, 'nvmlDeviceGetVgpuTypeSupportedPlacements')

        global __nvmlDeviceGetVgpuTypeCreatablePlacements
        __nvmlDeviceGetVgpuTypeCreatablePlacements = GetProcAddress(handle, 'nvmlDeviceGetVgpuTypeCreatablePlacements')

        global __nvmlVgpuTypeGetGspHeapSize
        __nvmlVgpuTypeGetGspHeapSize = GetProcAddress(handle, 'nvmlVgpuTypeGetGspHeapSize')

        global __nvmlVgpuTypeGetFbReservation
        __nvmlVgpuTypeGetFbReservation = GetProcAddress(handle, 'nvmlVgpuTypeGetFbReservation')

        global __nvmlVgpuInstanceGetRuntimeStateSize
        __nvmlVgpuInstanceGetRuntimeStateSize = GetProcAddress(handle, 'nvmlVgpuInstanceGetRuntimeStateSize')

        global __nvmlDeviceSetVgpuCapabilities
        __nvmlDeviceSetVgpuCapabilities = GetProcAddress(handle, 'nvmlDeviceSetVgpuCapabilities')

        global __nvmlDeviceGetGridLicensableFeatures_v4
        __nvmlDeviceGetGridLicensableFeatures_v4 = GetProcAddress(handle, 'nvmlDeviceGetGridLicensableFeatures_v4')

        global __nvmlGetVgpuDriverCapabilities
        __nvmlGetVgpuDriverCapabilities = GetProcAddress(handle, 'nvmlGetVgpuDriverCapabilities')

        global __nvmlDeviceGetVgpuCapabilities
        __nvmlDeviceGetVgpuCapabilities = GetProcAddress(handle, 'nvmlDeviceGetVgpuCapabilities')

        global __nvmlDeviceGetSupportedVgpus
        __nvmlDeviceGetSupportedVgpus = GetProcAddress(handle, 'nvmlDeviceGetSupportedVgpus')

        global __nvmlDeviceGetCreatableVgpus
        __nvmlDeviceGetCreatableVgpus = GetProcAddress(handle, 'nvmlDeviceGetCreatableVgpus')

        global __nvmlVgpuTypeGetClass
        __nvmlVgpuTypeGetClass = GetProcAddress(handle, 'nvmlVgpuTypeGetClass')

        global __nvmlVgpuTypeGetName
        __nvmlVgpuTypeGetName = GetProcAddress(handle, 'nvmlVgpuTypeGetName')

        global __nvmlVgpuTypeGetGpuInstanceProfileId
        __nvmlVgpuTypeGetGpuInstanceProfileId = GetProcAddress(handle, 'nvmlVgpuTypeGetGpuInstanceProfileId')

        global __nvmlVgpuTypeGetDeviceID
        __nvmlVgpuTypeGetDeviceID = GetProcAddress(handle, 'nvmlVgpuTypeGetDeviceID')

        global __nvmlVgpuTypeGetFramebufferSize
        __nvmlVgpuTypeGetFramebufferSize = GetProcAddress(handle, 'nvmlVgpuTypeGetFramebufferSize')

        global __nvmlVgpuTypeGetNumDisplayHeads
        __nvmlVgpuTypeGetNumDisplayHeads = GetProcAddress(handle, 'nvmlVgpuTypeGetNumDisplayHeads')

        global __nvmlVgpuTypeGetResolution
        __nvmlVgpuTypeGetResolution = GetProcAddress(handle, 'nvmlVgpuTypeGetResolution')

        global __nvmlVgpuTypeGetLicense
        __nvmlVgpuTypeGetLicense = GetProcAddress(handle, 'nvmlVgpuTypeGetLicense')

        global __nvmlVgpuTypeGetFrameRateLimit
        __nvmlVgpuTypeGetFrameRateLimit = GetProcAddress(handle, 'nvmlVgpuTypeGetFrameRateLimit')

        global __nvmlVgpuTypeGetMaxInstances
        __nvmlVgpuTypeGetMaxInstances = GetProcAddress(handle, 'nvmlVgpuTypeGetMaxInstances')

        global __nvmlVgpuTypeGetMaxInstancesPerVm
        __nvmlVgpuTypeGetMaxInstancesPerVm = GetProcAddress(handle, 'nvmlVgpuTypeGetMaxInstancesPerVm')

        global __nvmlVgpuTypeGetBAR1Info
        __nvmlVgpuTypeGetBAR1Info = GetProcAddress(handle, 'nvmlVgpuTypeGetBAR1Info')

        global __nvmlDeviceGetActiveVgpus
        __nvmlDeviceGetActiveVgpus = GetProcAddress(handle, 'nvmlDeviceGetActiveVgpus')

        global __nvmlVgpuInstanceGetVmID
        __nvmlVgpuInstanceGetVmID = GetProcAddress(handle, 'nvmlVgpuInstanceGetVmID')

        global __nvmlVgpuInstanceGetUUID
        __nvmlVgpuInstanceGetUUID = GetProcAddress(handle, 'nvmlVgpuInstanceGetUUID')

        global __nvmlVgpuInstanceGetVmDriverVersion
        __nvmlVgpuInstanceGetVmDriverVersion = GetProcAddress(handle, 'nvmlVgpuInstanceGetVmDriverVersion')

        global __nvmlVgpuInstanceGetFbUsage
        __nvmlVgpuInstanceGetFbUsage = GetProcAddress(handle, 'nvmlVgpuInstanceGetFbUsage')

        global __nvmlVgpuInstanceGetLicenseStatus
        __nvmlVgpuInstanceGetLicenseStatus = GetProcAddress(handle, 'nvmlVgpuInstanceGetLicenseStatus')

        global __nvmlVgpuInstanceGetType
        __nvmlVgpuInstanceGetType = GetProcAddress(handle, 'nvmlVgpuInstanceGetType')

        global __nvmlVgpuInstanceGetFrameRateLimit
        __nvmlVgpuInstanceGetFrameRateLimit = GetProcAddress(handle, 'nvmlVgpuInstanceGetFrameRateLimit')

        global __nvmlVgpuInstanceGetEccMode
        __nvmlVgpuInstanceGetEccMode = GetProcAddress(handle, 'nvmlVgpuInstanceGetEccMode')

        global __nvmlVgpuInstanceGetEncoderCapacity
        __nvmlVgpuInstanceGetEncoderCapacity = GetProcAddress(handle, 'nvmlVgpuInstanceGetEncoderCapacity')

        global __nvmlVgpuInstanceSetEncoderCapacity
        __nvmlVgpuInstanceSetEncoderCapacity = GetProcAddress(handle, 'nvmlVgpuInstanceSetEncoderCapacity')

        global __nvmlVgpuInstanceGetEncoderStats
        __nvmlVgpuInstanceGetEncoderStats = GetProcAddress(handle, 'nvmlVgpuInstanceGetEncoderStats')

        global __nvmlVgpuInstanceGetEncoderSessions
        __nvmlVgpuInstanceGetEncoderSessions = GetProcAddress(handle, 'nvmlVgpuInstanceGetEncoderSessions')

        global __nvmlVgpuInstanceGetFBCStats
        __nvmlVgpuInstanceGetFBCStats = GetProcAddress(handle, 'nvmlVgpuInstanceGetFBCStats')

        global __nvmlVgpuInstanceGetFBCSessions
        __nvmlVgpuInstanceGetFBCSessions = GetProcAddress(handle, 'nvmlVgpuInstanceGetFBCSessions')

        global __nvmlVgpuInstanceGetGpuInstanceId
        __nvmlVgpuInstanceGetGpuInstanceId = GetProcAddress(handle, 'nvmlVgpuInstanceGetGpuInstanceId')

        global __nvmlVgpuInstanceGetGpuPciId
        __nvmlVgpuInstanceGetGpuPciId = GetProcAddress(handle, 'nvmlVgpuInstanceGetGpuPciId')

        global __nvmlVgpuTypeGetCapabilities
        __nvmlVgpuTypeGetCapabilities = GetProcAddress(handle, 'nvmlVgpuTypeGetCapabilities')

        global __nvmlVgpuInstanceGetMdevUUID
        __nvmlVgpuInstanceGetMdevUUID = GetProcAddress(handle, 'nvmlVgpuInstanceGetMdevUUID')

        global __nvmlGpuInstanceGetCreatableVgpus
        __nvmlGpuInstanceGetCreatableVgpus = GetProcAddress(handle, 'nvmlGpuInstanceGetCreatableVgpus')

        global __nvmlVgpuTypeGetMaxInstancesPerGpuInstance
        __nvmlVgpuTypeGetMaxInstancesPerGpuInstance = GetProcAddress(handle, 'nvmlVgpuTypeGetMaxInstancesPerGpuInstance')

        global __nvmlGpuInstanceGetActiveVgpus
        __nvmlGpuInstanceGetActiveVgpus = GetProcAddress(handle, 'nvmlGpuInstanceGetActiveVgpus')

        global __nvmlGpuInstanceSetVgpuSchedulerState
        __nvmlGpuInstanceSetVgpuSchedulerState = GetProcAddress(handle, 'nvmlGpuInstanceSetVgpuSchedulerState')

        global __nvmlGpuInstanceGetVgpuSchedulerState
        __nvmlGpuInstanceGetVgpuSchedulerState = GetProcAddress(handle, 'nvmlGpuInstanceGetVgpuSchedulerState')

        global __nvmlGpuInstanceGetVgpuSchedulerLog
        __nvmlGpuInstanceGetVgpuSchedulerLog = GetProcAddress(handle, 'nvmlGpuInstanceGetVgpuSchedulerLog')

        global __nvmlGpuInstanceGetVgpuTypeCreatablePlacements
        __nvmlGpuInstanceGetVgpuTypeCreatablePlacements = GetProcAddress(handle, 'nvmlGpuInstanceGetVgpuTypeCreatablePlacements')

        global __nvmlGpuInstanceGetVgpuHeterogeneousMode
        __nvmlGpuInstanceGetVgpuHeterogeneousMode = GetProcAddress(handle, 'nvmlGpuInstanceGetVgpuHeterogeneousMode')

        global __nvmlGpuInstanceSetVgpuHeterogeneousMode
        __nvmlGpuInstanceSetVgpuHeterogeneousMode = GetProcAddress(handle, 'nvmlGpuInstanceSetVgpuHeterogeneousMode')

        global __nvmlVgpuInstanceGetMetadata
        __nvmlVgpuInstanceGetMetadata = GetProcAddress(handle, 'nvmlVgpuInstanceGetMetadata')

        global __nvmlDeviceGetVgpuMetadata
        __nvmlDeviceGetVgpuMetadata = GetProcAddress(handle, 'nvmlDeviceGetVgpuMetadata')

        global __nvmlGetVgpuCompatibility
        __nvmlGetVgpuCompatibility = GetProcAddress(handle, 'nvmlGetVgpuCompatibility')

        global __nvmlDeviceGetPgpuMetadataString
        __nvmlDeviceGetPgpuMetadataString = GetProcAddress(handle, 'nvmlDeviceGetPgpuMetadataString')

        global __nvmlDeviceGetVgpuSchedulerLog
        __nvmlDeviceGetVgpuSchedulerLog = GetProcAddress(handle, 'nvmlDeviceGetVgpuSchedulerLog')

        global __nvmlDeviceGetVgpuSchedulerState
        __nvmlDeviceGetVgpuSchedulerState = GetProcAddress(handle, 'nvmlDeviceGetVgpuSchedulerState')

        global __nvmlDeviceGetVgpuSchedulerCapabilities
        __nvmlDeviceGetVgpuSchedulerCapabilities = GetProcAddress(handle, 'nvmlDeviceGetVgpuSchedulerCapabilities')

        global __nvmlDeviceSetVgpuSchedulerState
        __nvmlDeviceSetVgpuSchedulerState = GetProcAddress(handle, 'nvmlDeviceSetVgpuSchedulerState')

        global __nvmlGetVgpuVersion
        __nvmlGetVgpuVersion = GetProcAddress(handle, 'nvmlGetVgpuVersion')

        global __nvmlSetVgpuVersion
        __nvmlSetVgpuVersion = GetProcAddress(handle, 'nvmlSetVgpuVersion')

        global __nvmlDeviceGetVgpuUtilization
        __nvmlDeviceGetVgpuUtilization = GetProcAddress(handle, 'nvmlDeviceGetVgpuUtilization')

        global __nvmlDeviceGetVgpuInstancesUtilizationInfo
        __nvmlDeviceGetVgpuInstancesUtilizationInfo = GetProcAddress(handle, 'nvmlDeviceGetVgpuInstancesUtilizationInfo')

        global __nvmlDeviceGetVgpuProcessUtilization
        __nvmlDeviceGetVgpuProcessUtilization = GetProcAddress(handle, 'nvmlDeviceGetVgpuProcessUtilization')

        global __nvmlDeviceGetVgpuProcessesUtilizationInfo
        __nvmlDeviceGetVgpuProcessesUtilizationInfo = GetProcAddress(handle, 'nvmlDeviceGetVgpuProcessesUtilizationInfo')

        global __nvmlVgpuInstanceGetAccountingMode
        __nvmlVgpuInstanceGetAccountingMode = GetProcAddress(handle, 'nvmlVgpuInstanceGetAccountingMode')

        global __nvmlVgpuInstanceGetAccountingPids
        __nvmlVgpuInstanceGetAccountingPids = GetProcAddress(handle, 'nvmlVgpuInstanceGetAccountingPids')

        global __nvmlVgpuInstanceGetAccountingStats
        __nvmlVgpuInstanceGetAccountingStats = GetProcAddress(handle, 'nvmlVgpuInstanceGetAccountingStats')

        global __nvmlVgpuInstanceClearAccountingPids
        __nvmlVgpuInstanceClearAccountingPids = GetProcAddress(handle, 'nvmlVgpuInstanceClearAccountingPids')

        global __nvmlVgpuInstanceGetLicenseInfo_v2
        __nvmlVgpuInstanceGetLicenseInfo_v2 = GetProcAddress(handle, 'nvmlVgpuInstanceGetLicenseInfo_v2')

        global __nvmlGetExcludedDeviceCount
        __nvmlGetExcludedDeviceCount = GetProcAddress(handle, 'nvmlGetExcludedDeviceCount')

        global __nvmlGetExcludedDeviceInfoByIndex
        __nvmlGetExcludedDeviceInfoByIndex = GetProcAddress(handle, 'nvmlGetExcludedDeviceInfoByIndex')

        global __nvmlDeviceSetMigMode
        __nvmlDeviceSetMigMode = GetProcAddress(handle, 'nvmlDeviceSetMigMode')

        global __nvmlDeviceGetMigMode
        __nvmlDeviceGetMigMode = GetProcAddress(handle, 'nvmlDeviceGetMigMode')

        global __nvmlDeviceGetGpuInstanceProfileInfoV
        __nvmlDeviceGetGpuInstanceProfileInfoV = GetProcAddress(handle, 'nvmlDeviceGetGpuInstanceProfileInfoV')

        global __nvmlDeviceGetGpuInstancePossiblePlacements_v2
        __nvmlDeviceGetGpuInstancePossiblePlacements_v2 = GetProcAddress(handle, 'nvmlDeviceGetGpuInstancePossiblePlacements_v2')

        global __nvmlDeviceGetGpuInstanceRemainingCapacity
        __nvmlDeviceGetGpuInstanceRemainingCapacity = GetProcAddress(handle, 'nvmlDeviceGetGpuInstanceRemainingCapacity')

        global __nvmlDeviceCreateGpuInstance
        __nvmlDeviceCreateGpuInstance = GetProcAddress(handle, 'nvmlDeviceCreateGpuInstance')

        global __nvmlDeviceCreateGpuInstanceWithPlacement
        __nvmlDeviceCreateGpuInstanceWithPlacement = GetProcAddress(handle, 'nvmlDeviceCreateGpuInstanceWithPlacement')

        global __nvmlGpuInstanceDestroy
        __nvmlGpuInstanceDestroy = GetProcAddress(handle, 'nvmlGpuInstanceDestroy')

        global __nvmlDeviceGetGpuInstances
        __nvmlDeviceGetGpuInstances = GetProcAddress(handle, 'nvmlDeviceGetGpuInstances')

        global __nvmlDeviceGetGpuInstanceById
        __nvmlDeviceGetGpuInstanceById = GetProcAddress(handle, 'nvmlDeviceGetGpuInstanceById')

        global __nvmlGpuInstanceGetInfo
        __nvmlGpuInstanceGetInfo = GetProcAddress(handle, 'nvmlGpuInstanceGetInfo')

        global __nvmlGpuInstanceGetComputeInstanceProfileInfoV
        __nvmlGpuInstanceGetComputeInstanceProfileInfoV = GetProcAddress(handle, 'nvmlGpuInstanceGetComputeInstanceProfileInfoV')

        global __nvmlGpuInstanceGetComputeInstanceRemainingCapacity
        __nvmlGpuInstanceGetComputeInstanceRemainingCapacity = GetProcAddress(handle, 'nvmlGpuInstanceGetComputeInstanceRemainingCapacity')

        global __nvmlGpuInstanceGetComputeInstancePossiblePlacements
        __nvmlGpuInstanceGetComputeInstancePossiblePlacements = GetProcAddress(handle, 'nvmlGpuInstanceGetComputeInstancePossiblePlacements')

        global __nvmlGpuInstanceCreateComputeInstance
        __nvmlGpuInstanceCreateComputeInstance = GetProcAddress(handle, 'nvmlGpuInstanceCreateComputeInstance')

        global __nvmlGpuInstanceCreateComputeInstanceWithPlacement
        __nvmlGpuInstanceCreateComputeInstanceWithPlacement = GetProcAddress(handle, 'nvmlGpuInstanceCreateComputeInstanceWithPlacement')

        global __nvmlComputeInstanceDestroy
        __nvmlComputeInstanceDestroy = GetProcAddress(handle, 'nvmlComputeInstanceDestroy')

        global __nvmlGpuInstanceGetComputeInstances
        __nvmlGpuInstanceGetComputeInstances = GetProcAddress(handle, 'nvmlGpuInstanceGetComputeInstances')

        global __nvmlGpuInstanceGetComputeInstanceById
        __nvmlGpuInstanceGetComputeInstanceById = GetProcAddress(handle, 'nvmlGpuInstanceGetComputeInstanceById')

        global __nvmlComputeInstanceGetInfo_v2
        __nvmlComputeInstanceGetInfo_v2 = GetProcAddress(handle, 'nvmlComputeInstanceGetInfo_v2')

        global __nvmlDeviceIsMigDeviceHandle
        __nvmlDeviceIsMigDeviceHandle = GetProcAddress(handle, 'nvmlDeviceIsMigDeviceHandle')

        global __nvmlDeviceGetGpuInstanceId
        __nvmlDeviceGetGpuInstanceId = GetProcAddress(handle, 'nvmlDeviceGetGpuInstanceId')

        global __nvmlDeviceGetComputeInstanceId
        __nvmlDeviceGetComputeInstanceId = GetProcAddress(handle, 'nvmlDeviceGetComputeInstanceId')

        global __nvmlDeviceGetMaxMigDeviceCount
        __nvmlDeviceGetMaxMigDeviceCount = GetProcAddress(handle, 'nvmlDeviceGetMaxMigDeviceCount')

        global __nvmlDeviceGetMigDeviceHandleByIndex
        __nvmlDeviceGetMigDeviceHandleByIndex = GetProcAddress(handle, 'nvmlDeviceGetMigDeviceHandleByIndex')

        global __nvmlDeviceGetDeviceHandleFromMigDeviceHandle
        __nvmlDeviceGetDeviceHandleFromMigDeviceHandle = GetProcAddress(handle, 'nvmlDeviceGetDeviceHandleFromMigDeviceHandle')

        global __nvmlDeviceGetCapabilities
        __nvmlDeviceGetCapabilities = GetProcAddress(handle, 'nvmlDeviceGetCapabilities')

        global __nvmlDevicePowerSmoothingActivatePresetProfile
        __nvmlDevicePowerSmoothingActivatePresetProfile = GetProcAddress(handle, 'nvmlDevicePowerSmoothingActivatePresetProfile')

        global __nvmlDevicePowerSmoothingUpdatePresetProfileParam
        __nvmlDevicePowerSmoothingUpdatePresetProfileParam = GetProcAddress(handle, 'nvmlDevicePowerSmoothingUpdatePresetProfileParam')

        global __nvmlDevicePowerSmoothingSetState
        __nvmlDevicePowerSmoothingSetState = GetProcAddress(handle, 'nvmlDevicePowerSmoothingSetState')

        global __nvmlDeviceGetAddressingMode
        __nvmlDeviceGetAddressingMode = GetProcAddress(handle, 'nvmlDeviceGetAddressingMode')

        global __nvmlDeviceGetRepairStatus
        __nvmlDeviceGetRepairStatus = GetProcAddress(handle, 'nvmlDeviceGetRepairStatus')

        global __nvmlDeviceGetPowerMizerMode_v1
        __nvmlDeviceGetPowerMizerMode_v1 = GetProcAddress(handle, 'nvmlDeviceGetPowerMizerMode_v1')

        global __nvmlDeviceSetPowerMizerMode_v1
        __nvmlDeviceSetPowerMizerMode_v1 = GetProcAddress(handle, 'nvmlDeviceSetPowerMizerMode_v1')

        global __nvmlDeviceGetPdi
        __nvmlDeviceGetPdi = GetProcAddress(handle, 'nvmlDeviceGetPdi')

        global __nvmlDeviceSetHostname_v1
        __nvmlDeviceSetHostname_v1 = GetProcAddress(handle, 'nvmlDeviceSetHostname_v1')

        global __nvmlDeviceGetHostname_v1
        __nvmlDeviceGetHostname_v1 = GetProcAddress(handle, 'nvmlDeviceGetHostname_v1')

        global __nvmlDeviceGetNvLinkInfo
        __nvmlDeviceGetNvLinkInfo = GetProcAddress(handle, 'nvmlDeviceGetNvLinkInfo')

        global __nvmlDeviceReadWritePRM_v1
        __nvmlDeviceReadWritePRM_v1 = GetProcAddress(handle, 'nvmlDeviceReadWritePRM_v1')

        global __nvmlDeviceGetGpuInstanceProfileInfoByIdV
        __nvmlDeviceGetGpuInstanceProfileInfoByIdV = GetProcAddress(handle, 'nvmlDeviceGetGpuInstanceProfileInfoByIdV')

        global __nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts
        __nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts = GetProcAddress(handle, 'nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts')

        global __nvmlDeviceGetUnrepairableMemoryFlag_v1
        __nvmlDeviceGetUnrepairableMemoryFlag_v1 = GetProcAddress(handle, 'nvmlDeviceGetUnrepairableMemoryFlag_v1')

        global __nvmlDeviceReadPRMCounters_v1
        __nvmlDeviceReadPRMCounters_v1 = GetProcAddress(handle, 'nvmlDeviceReadPRMCounters_v1')

        global __nvmlDeviceSetRusdSettings_v1
        __nvmlDeviceSetRusdSettings_v1 = GetProcAddress(handle, 'nvmlDeviceSetRusdSettings_v1')

        __py_nvml_init = True
        return 0


cdef inline int _check_or_init_nvml() except -1 nogil:
    if __py_nvml_init:
        return 0

    return _init_nvml()


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_nvml()
    cdef dict data = {}

    global __nvmlInit_v2
    data["__nvmlInit_v2"] = <intptr_t>__nvmlInit_v2

    global __nvmlInitWithFlags
    data["__nvmlInitWithFlags"] = <intptr_t>__nvmlInitWithFlags

    global __nvmlShutdown
    data["__nvmlShutdown"] = <intptr_t>__nvmlShutdown

    global __nvmlErrorString
    data["__nvmlErrorString"] = <intptr_t>__nvmlErrorString

    global __nvmlSystemGetDriverVersion
    data["__nvmlSystemGetDriverVersion"] = <intptr_t>__nvmlSystemGetDriverVersion

    global __nvmlSystemGetNVMLVersion
    data["__nvmlSystemGetNVMLVersion"] = <intptr_t>__nvmlSystemGetNVMLVersion

    global __nvmlSystemGetCudaDriverVersion
    data["__nvmlSystemGetCudaDriverVersion"] = <intptr_t>__nvmlSystemGetCudaDriverVersion

    global __nvmlSystemGetCudaDriverVersion_v2
    data["__nvmlSystemGetCudaDriverVersion_v2"] = <intptr_t>__nvmlSystemGetCudaDriverVersion_v2

    global __nvmlSystemGetProcessName
    data["__nvmlSystemGetProcessName"] = <intptr_t>__nvmlSystemGetProcessName

    global __nvmlSystemGetHicVersion
    data["__nvmlSystemGetHicVersion"] = <intptr_t>__nvmlSystemGetHicVersion

    global __nvmlSystemGetTopologyGpuSet
    data["__nvmlSystemGetTopologyGpuSet"] = <intptr_t>__nvmlSystemGetTopologyGpuSet

    global __nvmlSystemGetDriverBranch
    data["__nvmlSystemGetDriverBranch"] = <intptr_t>__nvmlSystemGetDriverBranch

    global __nvmlUnitGetCount
    data["__nvmlUnitGetCount"] = <intptr_t>__nvmlUnitGetCount

    global __nvmlUnitGetHandleByIndex
    data["__nvmlUnitGetHandleByIndex"] = <intptr_t>__nvmlUnitGetHandleByIndex

    global __nvmlUnitGetUnitInfo
    data["__nvmlUnitGetUnitInfo"] = <intptr_t>__nvmlUnitGetUnitInfo

    global __nvmlUnitGetLedState
    data["__nvmlUnitGetLedState"] = <intptr_t>__nvmlUnitGetLedState

    global __nvmlUnitGetPsuInfo
    data["__nvmlUnitGetPsuInfo"] = <intptr_t>__nvmlUnitGetPsuInfo

    global __nvmlUnitGetTemperature
    data["__nvmlUnitGetTemperature"] = <intptr_t>__nvmlUnitGetTemperature

    global __nvmlUnitGetFanSpeedInfo
    data["__nvmlUnitGetFanSpeedInfo"] = <intptr_t>__nvmlUnitGetFanSpeedInfo

    global __nvmlUnitGetDevices
    data["__nvmlUnitGetDevices"] = <intptr_t>__nvmlUnitGetDevices

    global __nvmlDeviceGetCount_v2
    data["__nvmlDeviceGetCount_v2"] = <intptr_t>__nvmlDeviceGetCount_v2

    global __nvmlDeviceGetAttributes_v2
    data["__nvmlDeviceGetAttributes_v2"] = <intptr_t>__nvmlDeviceGetAttributes_v2

    global __nvmlDeviceGetHandleByIndex_v2
    data["__nvmlDeviceGetHandleByIndex_v2"] = <intptr_t>__nvmlDeviceGetHandleByIndex_v2

    global __nvmlDeviceGetHandleBySerial
    data["__nvmlDeviceGetHandleBySerial"] = <intptr_t>__nvmlDeviceGetHandleBySerial

    global __nvmlDeviceGetHandleByUUID
    data["__nvmlDeviceGetHandleByUUID"] = <intptr_t>__nvmlDeviceGetHandleByUUID

    global __nvmlDeviceGetHandleByUUIDV
    data["__nvmlDeviceGetHandleByUUIDV"] = <intptr_t>__nvmlDeviceGetHandleByUUIDV

    global __nvmlDeviceGetHandleByPciBusId_v2
    data["__nvmlDeviceGetHandleByPciBusId_v2"] = <intptr_t>__nvmlDeviceGetHandleByPciBusId_v2

    global __nvmlDeviceGetName
    data["__nvmlDeviceGetName"] = <intptr_t>__nvmlDeviceGetName

    global __nvmlDeviceGetBrand
    data["__nvmlDeviceGetBrand"] = <intptr_t>__nvmlDeviceGetBrand

    global __nvmlDeviceGetIndex
    data["__nvmlDeviceGetIndex"] = <intptr_t>__nvmlDeviceGetIndex

    global __nvmlDeviceGetSerial
    data["__nvmlDeviceGetSerial"] = <intptr_t>__nvmlDeviceGetSerial

    global __nvmlDeviceGetModuleId
    data["__nvmlDeviceGetModuleId"] = <intptr_t>__nvmlDeviceGetModuleId

    global __nvmlDeviceGetC2cModeInfoV
    data["__nvmlDeviceGetC2cModeInfoV"] = <intptr_t>__nvmlDeviceGetC2cModeInfoV

    global __nvmlDeviceGetMemoryAffinity
    data["__nvmlDeviceGetMemoryAffinity"] = <intptr_t>__nvmlDeviceGetMemoryAffinity

    global __nvmlDeviceGetCpuAffinityWithinScope
    data["__nvmlDeviceGetCpuAffinityWithinScope"] = <intptr_t>__nvmlDeviceGetCpuAffinityWithinScope

    global __nvmlDeviceGetCpuAffinity
    data["__nvmlDeviceGetCpuAffinity"] = <intptr_t>__nvmlDeviceGetCpuAffinity

    global __nvmlDeviceSetCpuAffinity
    data["__nvmlDeviceSetCpuAffinity"] = <intptr_t>__nvmlDeviceSetCpuAffinity

    global __nvmlDeviceClearCpuAffinity
    data["__nvmlDeviceClearCpuAffinity"] = <intptr_t>__nvmlDeviceClearCpuAffinity

    global __nvmlDeviceGetNumaNodeId
    data["__nvmlDeviceGetNumaNodeId"] = <intptr_t>__nvmlDeviceGetNumaNodeId

    global __nvmlDeviceGetTopologyCommonAncestor
    data["__nvmlDeviceGetTopologyCommonAncestor"] = <intptr_t>__nvmlDeviceGetTopologyCommonAncestor

    global __nvmlDeviceGetTopologyNearestGpus
    data["__nvmlDeviceGetTopologyNearestGpus"] = <intptr_t>__nvmlDeviceGetTopologyNearestGpus

    global __nvmlDeviceGetP2PStatus
    data["__nvmlDeviceGetP2PStatus"] = <intptr_t>__nvmlDeviceGetP2PStatus

    global __nvmlDeviceGetUUID
    data["__nvmlDeviceGetUUID"] = <intptr_t>__nvmlDeviceGetUUID

    global __nvmlDeviceGetMinorNumber
    data["__nvmlDeviceGetMinorNumber"] = <intptr_t>__nvmlDeviceGetMinorNumber

    global __nvmlDeviceGetBoardPartNumber
    data["__nvmlDeviceGetBoardPartNumber"] = <intptr_t>__nvmlDeviceGetBoardPartNumber

    global __nvmlDeviceGetInforomVersion
    data["__nvmlDeviceGetInforomVersion"] = <intptr_t>__nvmlDeviceGetInforomVersion

    global __nvmlDeviceGetInforomImageVersion
    data["__nvmlDeviceGetInforomImageVersion"] = <intptr_t>__nvmlDeviceGetInforomImageVersion

    global __nvmlDeviceGetInforomConfigurationChecksum
    data["__nvmlDeviceGetInforomConfigurationChecksum"] = <intptr_t>__nvmlDeviceGetInforomConfigurationChecksum

    global __nvmlDeviceValidateInforom
    data["__nvmlDeviceValidateInforom"] = <intptr_t>__nvmlDeviceValidateInforom

    global __nvmlDeviceGetLastBBXFlushTime
    data["__nvmlDeviceGetLastBBXFlushTime"] = <intptr_t>__nvmlDeviceGetLastBBXFlushTime

    global __nvmlDeviceGetDisplayMode
    data["__nvmlDeviceGetDisplayMode"] = <intptr_t>__nvmlDeviceGetDisplayMode

    global __nvmlDeviceGetDisplayActive
    data["__nvmlDeviceGetDisplayActive"] = <intptr_t>__nvmlDeviceGetDisplayActive

    global __nvmlDeviceGetPersistenceMode
    data["__nvmlDeviceGetPersistenceMode"] = <intptr_t>__nvmlDeviceGetPersistenceMode

    global __nvmlDeviceGetPciInfoExt
    data["__nvmlDeviceGetPciInfoExt"] = <intptr_t>__nvmlDeviceGetPciInfoExt

    global __nvmlDeviceGetPciInfo_v3
    data["__nvmlDeviceGetPciInfo_v3"] = <intptr_t>__nvmlDeviceGetPciInfo_v3

    global __nvmlDeviceGetMaxPcieLinkGeneration
    data["__nvmlDeviceGetMaxPcieLinkGeneration"] = <intptr_t>__nvmlDeviceGetMaxPcieLinkGeneration

    global __nvmlDeviceGetGpuMaxPcieLinkGeneration
    data["__nvmlDeviceGetGpuMaxPcieLinkGeneration"] = <intptr_t>__nvmlDeviceGetGpuMaxPcieLinkGeneration

    global __nvmlDeviceGetMaxPcieLinkWidth
    data["__nvmlDeviceGetMaxPcieLinkWidth"] = <intptr_t>__nvmlDeviceGetMaxPcieLinkWidth

    global __nvmlDeviceGetCurrPcieLinkGeneration
    data["__nvmlDeviceGetCurrPcieLinkGeneration"] = <intptr_t>__nvmlDeviceGetCurrPcieLinkGeneration

    global __nvmlDeviceGetCurrPcieLinkWidth
    data["__nvmlDeviceGetCurrPcieLinkWidth"] = <intptr_t>__nvmlDeviceGetCurrPcieLinkWidth

    global __nvmlDeviceGetPcieThroughput
    data["__nvmlDeviceGetPcieThroughput"] = <intptr_t>__nvmlDeviceGetPcieThroughput

    global __nvmlDeviceGetPcieReplayCounter
    data["__nvmlDeviceGetPcieReplayCounter"] = <intptr_t>__nvmlDeviceGetPcieReplayCounter

    global __nvmlDeviceGetClockInfo
    data["__nvmlDeviceGetClockInfo"] = <intptr_t>__nvmlDeviceGetClockInfo

    global __nvmlDeviceGetMaxClockInfo
    data["__nvmlDeviceGetMaxClockInfo"] = <intptr_t>__nvmlDeviceGetMaxClockInfo

    global __nvmlDeviceGetGpcClkVfOffset
    data["__nvmlDeviceGetGpcClkVfOffset"] = <intptr_t>__nvmlDeviceGetGpcClkVfOffset

    global __nvmlDeviceGetClock
    data["__nvmlDeviceGetClock"] = <intptr_t>__nvmlDeviceGetClock

    global __nvmlDeviceGetMaxCustomerBoostClock
    data["__nvmlDeviceGetMaxCustomerBoostClock"] = <intptr_t>__nvmlDeviceGetMaxCustomerBoostClock

    global __nvmlDeviceGetSupportedMemoryClocks
    data["__nvmlDeviceGetSupportedMemoryClocks"] = <intptr_t>__nvmlDeviceGetSupportedMemoryClocks

    global __nvmlDeviceGetSupportedGraphicsClocks
    data["__nvmlDeviceGetSupportedGraphicsClocks"] = <intptr_t>__nvmlDeviceGetSupportedGraphicsClocks

    global __nvmlDeviceGetAutoBoostedClocksEnabled
    data["__nvmlDeviceGetAutoBoostedClocksEnabled"] = <intptr_t>__nvmlDeviceGetAutoBoostedClocksEnabled

    global __nvmlDeviceGetFanSpeed
    data["__nvmlDeviceGetFanSpeed"] = <intptr_t>__nvmlDeviceGetFanSpeed

    global __nvmlDeviceGetFanSpeed_v2
    data["__nvmlDeviceGetFanSpeed_v2"] = <intptr_t>__nvmlDeviceGetFanSpeed_v2

    global __nvmlDeviceGetFanSpeedRPM
    data["__nvmlDeviceGetFanSpeedRPM"] = <intptr_t>__nvmlDeviceGetFanSpeedRPM

    global __nvmlDeviceGetTargetFanSpeed
    data["__nvmlDeviceGetTargetFanSpeed"] = <intptr_t>__nvmlDeviceGetTargetFanSpeed

    global __nvmlDeviceGetMinMaxFanSpeed
    data["__nvmlDeviceGetMinMaxFanSpeed"] = <intptr_t>__nvmlDeviceGetMinMaxFanSpeed

    global __nvmlDeviceGetFanControlPolicy_v2
    data["__nvmlDeviceGetFanControlPolicy_v2"] = <intptr_t>__nvmlDeviceGetFanControlPolicy_v2

    global __nvmlDeviceGetNumFans
    data["__nvmlDeviceGetNumFans"] = <intptr_t>__nvmlDeviceGetNumFans

    global __nvmlDeviceGetCoolerInfo
    data["__nvmlDeviceGetCoolerInfo"] = <intptr_t>__nvmlDeviceGetCoolerInfo

    global __nvmlDeviceGetTemperatureV
    data["__nvmlDeviceGetTemperatureV"] = <intptr_t>__nvmlDeviceGetTemperatureV

    global __nvmlDeviceGetTemperatureThreshold
    data["__nvmlDeviceGetTemperatureThreshold"] = <intptr_t>__nvmlDeviceGetTemperatureThreshold

    global __nvmlDeviceGetMarginTemperature
    data["__nvmlDeviceGetMarginTemperature"] = <intptr_t>__nvmlDeviceGetMarginTemperature

    global __nvmlDeviceGetThermalSettings
    data["__nvmlDeviceGetThermalSettings"] = <intptr_t>__nvmlDeviceGetThermalSettings

    global __nvmlDeviceGetPerformanceState
    data["__nvmlDeviceGetPerformanceState"] = <intptr_t>__nvmlDeviceGetPerformanceState

    global __nvmlDeviceGetCurrentClocksEventReasons
    data["__nvmlDeviceGetCurrentClocksEventReasons"] = <intptr_t>__nvmlDeviceGetCurrentClocksEventReasons

    global __nvmlDeviceGetSupportedClocksEventReasons
    data["__nvmlDeviceGetSupportedClocksEventReasons"] = <intptr_t>__nvmlDeviceGetSupportedClocksEventReasons

    global __nvmlDeviceGetPowerState
    data["__nvmlDeviceGetPowerState"] = <intptr_t>__nvmlDeviceGetPowerState

    global __nvmlDeviceGetDynamicPstatesInfo
    data["__nvmlDeviceGetDynamicPstatesInfo"] = <intptr_t>__nvmlDeviceGetDynamicPstatesInfo

    global __nvmlDeviceGetMemClkVfOffset
    data["__nvmlDeviceGetMemClkVfOffset"] = <intptr_t>__nvmlDeviceGetMemClkVfOffset

    global __nvmlDeviceGetMinMaxClockOfPState
    data["__nvmlDeviceGetMinMaxClockOfPState"] = <intptr_t>__nvmlDeviceGetMinMaxClockOfPState

    global __nvmlDeviceGetSupportedPerformanceStates
    data["__nvmlDeviceGetSupportedPerformanceStates"] = <intptr_t>__nvmlDeviceGetSupportedPerformanceStates

    global __nvmlDeviceGetGpcClkMinMaxVfOffset
    data["__nvmlDeviceGetGpcClkMinMaxVfOffset"] = <intptr_t>__nvmlDeviceGetGpcClkMinMaxVfOffset

    global __nvmlDeviceGetMemClkMinMaxVfOffset
    data["__nvmlDeviceGetMemClkMinMaxVfOffset"] = <intptr_t>__nvmlDeviceGetMemClkMinMaxVfOffset

    global __nvmlDeviceGetClockOffsets
    data["__nvmlDeviceGetClockOffsets"] = <intptr_t>__nvmlDeviceGetClockOffsets

    global __nvmlDeviceSetClockOffsets
    data["__nvmlDeviceSetClockOffsets"] = <intptr_t>__nvmlDeviceSetClockOffsets

    global __nvmlDeviceGetPerformanceModes
    data["__nvmlDeviceGetPerformanceModes"] = <intptr_t>__nvmlDeviceGetPerformanceModes

    global __nvmlDeviceGetCurrentClockFreqs
    data["__nvmlDeviceGetCurrentClockFreqs"] = <intptr_t>__nvmlDeviceGetCurrentClockFreqs

    global __nvmlDeviceGetPowerManagementLimit
    data["__nvmlDeviceGetPowerManagementLimit"] = <intptr_t>__nvmlDeviceGetPowerManagementLimit

    global __nvmlDeviceGetPowerManagementLimitConstraints
    data["__nvmlDeviceGetPowerManagementLimitConstraints"] = <intptr_t>__nvmlDeviceGetPowerManagementLimitConstraints

    global __nvmlDeviceGetPowerManagementDefaultLimit
    data["__nvmlDeviceGetPowerManagementDefaultLimit"] = <intptr_t>__nvmlDeviceGetPowerManagementDefaultLimit

    global __nvmlDeviceGetPowerUsage
    data["__nvmlDeviceGetPowerUsage"] = <intptr_t>__nvmlDeviceGetPowerUsage

    global __nvmlDeviceGetTotalEnergyConsumption
    data["__nvmlDeviceGetTotalEnergyConsumption"] = <intptr_t>__nvmlDeviceGetTotalEnergyConsumption

    global __nvmlDeviceGetEnforcedPowerLimit
    data["__nvmlDeviceGetEnforcedPowerLimit"] = <intptr_t>__nvmlDeviceGetEnforcedPowerLimit

    global __nvmlDeviceGetGpuOperationMode
    data["__nvmlDeviceGetGpuOperationMode"] = <intptr_t>__nvmlDeviceGetGpuOperationMode

    global __nvmlDeviceGetMemoryInfo_v2
    data["__nvmlDeviceGetMemoryInfo_v2"] = <intptr_t>__nvmlDeviceGetMemoryInfo_v2

    global __nvmlDeviceGetComputeMode
    data["__nvmlDeviceGetComputeMode"] = <intptr_t>__nvmlDeviceGetComputeMode

    global __nvmlDeviceGetCudaComputeCapability
    data["__nvmlDeviceGetCudaComputeCapability"] = <intptr_t>__nvmlDeviceGetCudaComputeCapability

    global __nvmlDeviceGetDramEncryptionMode
    data["__nvmlDeviceGetDramEncryptionMode"] = <intptr_t>__nvmlDeviceGetDramEncryptionMode

    global __nvmlDeviceSetDramEncryptionMode
    data["__nvmlDeviceSetDramEncryptionMode"] = <intptr_t>__nvmlDeviceSetDramEncryptionMode

    global __nvmlDeviceGetEccMode
    data["__nvmlDeviceGetEccMode"] = <intptr_t>__nvmlDeviceGetEccMode

    global __nvmlDeviceGetDefaultEccMode
    data["__nvmlDeviceGetDefaultEccMode"] = <intptr_t>__nvmlDeviceGetDefaultEccMode

    global __nvmlDeviceGetBoardId
    data["__nvmlDeviceGetBoardId"] = <intptr_t>__nvmlDeviceGetBoardId

    global __nvmlDeviceGetMultiGpuBoard
    data["__nvmlDeviceGetMultiGpuBoard"] = <intptr_t>__nvmlDeviceGetMultiGpuBoard

    global __nvmlDeviceGetTotalEccErrors
    data["__nvmlDeviceGetTotalEccErrors"] = <intptr_t>__nvmlDeviceGetTotalEccErrors

    global __nvmlDeviceGetMemoryErrorCounter
    data["__nvmlDeviceGetMemoryErrorCounter"] = <intptr_t>__nvmlDeviceGetMemoryErrorCounter

    global __nvmlDeviceGetUtilizationRates
    data["__nvmlDeviceGetUtilizationRates"] = <intptr_t>__nvmlDeviceGetUtilizationRates

    global __nvmlDeviceGetEncoderUtilization
    data["__nvmlDeviceGetEncoderUtilization"] = <intptr_t>__nvmlDeviceGetEncoderUtilization

    global __nvmlDeviceGetEncoderCapacity
    data["__nvmlDeviceGetEncoderCapacity"] = <intptr_t>__nvmlDeviceGetEncoderCapacity

    global __nvmlDeviceGetEncoderStats
    data["__nvmlDeviceGetEncoderStats"] = <intptr_t>__nvmlDeviceGetEncoderStats

    global __nvmlDeviceGetEncoderSessions
    data["__nvmlDeviceGetEncoderSessions"] = <intptr_t>__nvmlDeviceGetEncoderSessions

    global __nvmlDeviceGetDecoderUtilization
    data["__nvmlDeviceGetDecoderUtilization"] = <intptr_t>__nvmlDeviceGetDecoderUtilization

    global __nvmlDeviceGetJpgUtilization
    data["__nvmlDeviceGetJpgUtilization"] = <intptr_t>__nvmlDeviceGetJpgUtilization

    global __nvmlDeviceGetOfaUtilization
    data["__nvmlDeviceGetOfaUtilization"] = <intptr_t>__nvmlDeviceGetOfaUtilization

    global __nvmlDeviceGetFBCStats
    data["__nvmlDeviceGetFBCStats"] = <intptr_t>__nvmlDeviceGetFBCStats

    global __nvmlDeviceGetFBCSessions
    data["__nvmlDeviceGetFBCSessions"] = <intptr_t>__nvmlDeviceGetFBCSessions

    global __nvmlDeviceGetDriverModel_v2
    data["__nvmlDeviceGetDriverModel_v2"] = <intptr_t>__nvmlDeviceGetDriverModel_v2

    global __nvmlDeviceGetVbiosVersion
    data["__nvmlDeviceGetVbiosVersion"] = <intptr_t>__nvmlDeviceGetVbiosVersion

    global __nvmlDeviceGetBridgeChipInfo
    data["__nvmlDeviceGetBridgeChipInfo"] = <intptr_t>__nvmlDeviceGetBridgeChipInfo

    global __nvmlDeviceGetComputeRunningProcesses_v3
    data["__nvmlDeviceGetComputeRunningProcesses_v3"] = <intptr_t>__nvmlDeviceGetComputeRunningProcesses_v3

    global __nvmlDeviceGetMPSComputeRunningProcesses_v3
    data["__nvmlDeviceGetMPSComputeRunningProcesses_v3"] = <intptr_t>__nvmlDeviceGetMPSComputeRunningProcesses_v3

    global __nvmlDeviceGetRunningProcessDetailList
    data["__nvmlDeviceGetRunningProcessDetailList"] = <intptr_t>__nvmlDeviceGetRunningProcessDetailList

    global __nvmlDeviceOnSameBoard
    data["__nvmlDeviceOnSameBoard"] = <intptr_t>__nvmlDeviceOnSameBoard

    global __nvmlDeviceGetAPIRestriction
    data["__nvmlDeviceGetAPIRestriction"] = <intptr_t>__nvmlDeviceGetAPIRestriction

    global __nvmlDeviceGetSamples
    data["__nvmlDeviceGetSamples"] = <intptr_t>__nvmlDeviceGetSamples

    global __nvmlDeviceGetBAR1MemoryInfo
    data["__nvmlDeviceGetBAR1MemoryInfo"] = <intptr_t>__nvmlDeviceGetBAR1MemoryInfo

    global __nvmlDeviceGetIrqNum
    data["__nvmlDeviceGetIrqNum"] = <intptr_t>__nvmlDeviceGetIrqNum

    global __nvmlDeviceGetNumGpuCores
    data["__nvmlDeviceGetNumGpuCores"] = <intptr_t>__nvmlDeviceGetNumGpuCores

    global __nvmlDeviceGetPowerSource
    data["__nvmlDeviceGetPowerSource"] = <intptr_t>__nvmlDeviceGetPowerSource

    global __nvmlDeviceGetMemoryBusWidth
    data["__nvmlDeviceGetMemoryBusWidth"] = <intptr_t>__nvmlDeviceGetMemoryBusWidth

    global __nvmlDeviceGetPcieLinkMaxSpeed
    data["__nvmlDeviceGetPcieLinkMaxSpeed"] = <intptr_t>__nvmlDeviceGetPcieLinkMaxSpeed

    global __nvmlDeviceGetPcieSpeed
    data["__nvmlDeviceGetPcieSpeed"] = <intptr_t>__nvmlDeviceGetPcieSpeed

    global __nvmlDeviceGetAdaptiveClockInfoStatus
    data["__nvmlDeviceGetAdaptiveClockInfoStatus"] = <intptr_t>__nvmlDeviceGetAdaptiveClockInfoStatus

    global __nvmlDeviceGetBusType
    data["__nvmlDeviceGetBusType"] = <intptr_t>__nvmlDeviceGetBusType

    global __nvmlDeviceGetGpuFabricInfoV
    data["__nvmlDeviceGetGpuFabricInfoV"] = <intptr_t>__nvmlDeviceGetGpuFabricInfoV

    global __nvmlSystemGetConfComputeCapabilities
    data["__nvmlSystemGetConfComputeCapabilities"] = <intptr_t>__nvmlSystemGetConfComputeCapabilities

    global __nvmlSystemGetConfComputeState
    data["__nvmlSystemGetConfComputeState"] = <intptr_t>__nvmlSystemGetConfComputeState

    global __nvmlDeviceGetConfComputeMemSizeInfo
    data["__nvmlDeviceGetConfComputeMemSizeInfo"] = <intptr_t>__nvmlDeviceGetConfComputeMemSizeInfo

    global __nvmlSystemGetConfComputeGpusReadyState
    data["__nvmlSystemGetConfComputeGpusReadyState"] = <intptr_t>__nvmlSystemGetConfComputeGpusReadyState

    global __nvmlDeviceGetConfComputeProtectedMemoryUsage
    data["__nvmlDeviceGetConfComputeProtectedMemoryUsage"] = <intptr_t>__nvmlDeviceGetConfComputeProtectedMemoryUsage

    global __nvmlDeviceGetConfComputeGpuCertificate
    data["__nvmlDeviceGetConfComputeGpuCertificate"] = <intptr_t>__nvmlDeviceGetConfComputeGpuCertificate

    global __nvmlDeviceGetConfComputeGpuAttestationReport
    data["__nvmlDeviceGetConfComputeGpuAttestationReport"] = <intptr_t>__nvmlDeviceGetConfComputeGpuAttestationReport

    global __nvmlSystemGetConfComputeKeyRotationThresholdInfo
    data["__nvmlSystemGetConfComputeKeyRotationThresholdInfo"] = <intptr_t>__nvmlSystemGetConfComputeKeyRotationThresholdInfo

    global __nvmlDeviceSetConfComputeUnprotectedMemSize
    data["__nvmlDeviceSetConfComputeUnprotectedMemSize"] = <intptr_t>__nvmlDeviceSetConfComputeUnprotectedMemSize

    global __nvmlSystemSetConfComputeGpusReadyState
    data["__nvmlSystemSetConfComputeGpusReadyState"] = <intptr_t>__nvmlSystemSetConfComputeGpusReadyState

    global __nvmlSystemSetConfComputeKeyRotationThresholdInfo
    data["__nvmlSystemSetConfComputeKeyRotationThresholdInfo"] = <intptr_t>__nvmlSystemSetConfComputeKeyRotationThresholdInfo

    global __nvmlSystemGetConfComputeSettings
    data["__nvmlSystemGetConfComputeSettings"] = <intptr_t>__nvmlSystemGetConfComputeSettings

    global __nvmlDeviceGetGspFirmwareVersion
    data["__nvmlDeviceGetGspFirmwareVersion"] = <intptr_t>__nvmlDeviceGetGspFirmwareVersion

    global __nvmlDeviceGetGspFirmwareMode
    data["__nvmlDeviceGetGspFirmwareMode"] = <intptr_t>__nvmlDeviceGetGspFirmwareMode

    global __nvmlDeviceGetSramEccErrorStatus
    data["__nvmlDeviceGetSramEccErrorStatus"] = <intptr_t>__nvmlDeviceGetSramEccErrorStatus

    global __nvmlDeviceGetAccountingMode
    data["__nvmlDeviceGetAccountingMode"] = <intptr_t>__nvmlDeviceGetAccountingMode

    global __nvmlDeviceGetAccountingStats
    data["__nvmlDeviceGetAccountingStats"] = <intptr_t>__nvmlDeviceGetAccountingStats

    global __nvmlDeviceGetAccountingPids
    data["__nvmlDeviceGetAccountingPids"] = <intptr_t>__nvmlDeviceGetAccountingPids

    global __nvmlDeviceGetAccountingBufferSize
    data["__nvmlDeviceGetAccountingBufferSize"] = <intptr_t>__nvmlDeviceGetAccountingBufferSize

    global __nvmlDeviceGetRetiredPages
    data["__nvmlDeviceGetRetiredPages"] = <intptr_t>__nvmlDeviceGetRetiredPages

    global __nvmlDeviceGetRetiredPages_v2
    data["__nvmlDeviceGetRetiredPages_v2"] = <intptr_t>__nvmlDeviceGetRetiredPages_v2

    global __nvmlDeviceGetRetiredPagesPendingStatus
    data["__nvmlDeviceGetRetiredPagesPendingStatus"] = <intptr_t>__nvmlDeviceGetRetiredPagesPendingStatus

    global __nvmlDeviceGetRemappedRows
    data["__nvmlDeviceGetRemappedRows"] = <intptr_t>__nvmlDeviceGetRemappedRows

    global __nvmlDeviceGetRowRemapperHistogram
    data["__nvmlDeviceGetRowRemapperHistogram"] = <intptr_t>__nvmlDeviceGetRowRemapperHistogram

    global __nvmlDeviceGetArchitecture
    data["__nvmlDeviceGetArchitecture"] = <intptr_t>__nvmlDeviceGetArchitecture

    global __nvmlDeviceGetClkMonStatus
    data["__nvmlDeviceGetClkMonStatus"] = <intptr_t>__nvmlDeviceGetClkMonStatus

    global __nvmlDeviceGetProcessUtilization
    data["__nvmlDeviceGetProcessUtilization"] = <intptr_t>__nvmlDeviceGetProcessUtilization

    global __nvmlDeviceGetProcessesUtilizationInfo
    data["__nvmlDeviceGetProcessesUtilizationInfo"] = <intptr_t>__nvmlDeviceGetProcessesUtilizationInfo

    global __nvmlDeviceGetPlatformInfo
    data["__nvmlDeviceGetPlatformInfo"] = <intptr_t>__nvmlDeviceGetPlatformInfo

    global __nvmlUnitSetLedState
    data["__nvmlUnitSetLedState"] = <intptr_t>__nvmlUnitSetLedState

    global __nvmlDeviceSetPersistenceMode
    data["__nvmlDeviceSetPersistenceMode"] = <intptr_t>__nvmlDeviceSetPersistenceMode

    global __nvmlDeviceSetComputeMode
    data["__nvmlDeviceSetComputeMode"] = <intptr_t>__nvmlDeviceSetComputeMode

    global __nvmlDeviceSetEccMode
    data["__nvmlDeviceSetEccMode"] = <intptr_t>__nvmlDeviceSetEccMode

    global __nvmlDeviceClearEccErrorCounts
    data["__nvmlDeviceClearEccErrorCounts"] = <intptr_t>__nvmlDeviceClearEccErrorCounts

    global __nvmlDeviceSetDriverModel
    data["__nvmlDeviceSetDriverModel"] = <intptr_t>__nvmlDeviceSetDriverModel

    global __nvmlDeviceSetGpuLockedClocks
    data["__nvmlDeviceSetGpuLockedClocks"] = <intptr_t>__nvmlDeviceSetGpuLockedClocks

    global __nvmlDeviceResetGpuLockedClocks
    data["__nvmlDeviceResetGpuLockedClocks"] = <intptr_t>__nvmlDeviceResetGpuLockedClocks

    global __nvmlDeviceSetMemoryLockedClocks
    data["__nvmlDeviceSetMemoryLockedClocks"] = <intptr_t>__nvmlDeviceSetMemoryLockedClocks

    global __nvmlDeviceResetMemoryLockedClocks
    data["__nvmlDeviceResetMemoryLockedClocks"] = <intptr_t>__nvmlDeviceResetMemoryLockedClocks

    global __nvmlDeviceSetAutoBoostedClocksEnabled
    data["__nvmlDeviceSetAutoBoostedClocksEnabled"] = <intptr_t>__nvmlDeviceSetAutoBoostedClocksEnabled

    global __nvmlDeviceSetDefaultAutoBoostedClocksEnabled
    data["__nvmlDeviceSetDefaultAutoBoostedClocksEnabled"] = <intptr_t>__nvmlDeviceSetDefaultAutoBoostedClocksEnabled

    global __nvmlDeviceSetDefaultFanSpeed_v2
    data["__nvmlDeviceSetDefaultFanSpeed_v2"] = <intptr_t>__nvmlDeviceSetDefaultFanSpeed_v2

    global __nvmlDeviceSetFanControlPolicy
    data["__nvmlDeviceSetFanControlPolicy"] = <intptr_t>__nvmlDeviceSetFanControlPolicy

    global __nvmlDeviceSetTemperatureThreshold
    data["__nvmlDeviceSetTemperatureThreshold"] = <intptr_t>__nvmlDeviceSetTemperatureThreshold

    global __nvmlDeviceSetGpuOperationMode
    data["__nvmlDeviceSetGpuOperationMode"] = <intptr_t>__nvmlDeviceSetGpuOperationMode

    global __nvmlDeviceSetAPIRestriction
    data["__nvmlDeviceSetAPIRestriction"] = <intptr_t>__nvmlDeviceSetAPIRestriction

    global __nvmlDeviceSetFanSpeed_v2
    data["__nvmlDeviceSetFanSpeed_v2"] = <intptr_t>__nvmlDeviceSetFanSpeed_v2

    global __nvmlDeviceSetAccountingMode
    data["__nvmlDeviceSetAccountingMode"] = <intptr_t>__nvmlDeviceSetAccountingMode

    global __nvmlDeviceClearAccountingPids
    data["__nvmlDeviceClearAccountingPids"] = <intptr_t>__nvmlDeviceClearAccountingPids

    global __nvmlDeviceSetPowerManagementLimit_v2
    data["__nvmlDeviceSetPowerManagementLimit_v2"] = <intptr_t>__nvmlDeviceSetPowerManagementLimit_v2

    global __nvmlDeviceGetNvLinkState
    data["__nvmlDeviceGetNvLinkState"] = <intptr_t>__nvmlDeviceGetNvLinkState

    global __nvmlDeviceGetNvLinkVersion
    data["__nvmlDeviceGetNvLinkVersion"] = <intptr_t>__nvmlDeviceGetNvLinkVersion

    global __nvmlDeviceGetNvLinkCapability
    data["__nvmlDeviceGetNvLinkCapability"] = <intptr_t>__nvmlDeviceGetNvLinkCapability

    global __nvmlDeviceGetNvLinkRemotePciInfo_v2
    data["__nvmlDeviceGetNvLinkRemotePciInfo_v2"] = <intptr_t>__nvmlDeviceGetNvLinkRemotePciInfo_v2

    global __nvmlDeviceGetNvLinkErrorCounter
    data["__nvmlDeviceGetNvLinkErrorCounter"] = <intptr_t>__nvmlDeviceGetNvLinkErrorCounter

    global __nvmlDeviceResetNvLinkErrorCounters
    data["__nvmlDeviceResetNvLinkErrorCounters"] = <intptr_t>__nvmlDeviceResetNvLinkErrorCounters

    global __nvmlDeviceGetNvLinkRemoteDeviceType
    data["__nvmlDeviceGetNvLinkRemoteDeviceType"] = <intptr_t>__nvmlDeviceGetNvLinkRemoteDeviceType

    global __nvmlDeviceSetNvLinkDeviceLowPowerThreshold
    data["__nvmlDeviceSetNvLinkDeviceLowPowerThreshold"] = <intptr_t>__nvmlDeviceSetNvLinkDeviceLowPowerThreshold

    global __nvmlSystemSetNvlinkBwMode
    data["__nvmlSystemSetNvlinkBwMode"] = <intptr_t>__nvmlSystemSetNvlinkBwMode

    global __nvmlSystemGetNvlinkBwMode
    data["__nvmlSystemGetNvlinkBwMode"] = <intptr_t>__nvmlSystemGetNvlinkBwMode

    global __nvmlDeviceGetNvlinkSupportedBwModes
    data["__nvmlDeviceGetNvlinkSupportedBwModes"] = <intptr_t>__nvmlDeviceGetNvlinkSupportedBwModes

    global __nvmlDeviceGetNvlinkBwMode
    data["__nvmlDeviceGetNvlinkBwMode"] = <intptr_t>__nvmlDeviceGetNvlinkBwMode

    global __nvmlDeviceSetNvlinkBwMode
    data["__nvmlDeviceSetNvlinkBwMode"] = <intptr_t>__nvmlDeviceSetNvlinkBwMode

    global __nvmlEventSetCreate
    data["__nvmlEventSetCreate"] = <intptr_t>__nvmlEventSetCreate

    global __nvmlDeviceRegisterEvents
    data["__nvmlDeviceRegisterEvents"] = <intptr_t>__nvmlDeviceRegisterEvents

    global __nvmlDeviceGetSupportedEventTypes
    data["__nvmlDeviceGetSupportedEventTypes"] = <intptr_t>__nvmlDeviceGetSupportedEventTypes

    global __nvmlEventSetWait_v2
    data["__nvmlEventSetWait_v2"] = <intptr_t>__nvmlEventSetWait_v2

    global __nvmlEventSetFree
    data["__nvmlEventSetFree"] = <intptr_t>__nvmlEventSetFree

    global __nvmlSystemEventSetCreate
    data["__nvmlSystemEventSetCreate"] = <intptr_t>__nvmlSystemEventSetCreate

    global __nvmlSystemEventSetFree
    data["__nvmlSystemEventSetFree"] = <intptr_t>__nvmlSystemEventSetFree

    global __nvmlSystemRegisterEvents
    data["__nvmlSystemRegisterEvents"] = <intptr_t>__nvmlSystemRegisterEvents

    global __nvmlSystemEventSetWait
    data["__nvmlSystemEventSetWait"] = <intptr_t>__nvmlSystemEventSetWait

    global __nvmlDeviceModifyDrainState
    data["__nvmlDeviceModifyDrainState"] = <intptr_t>__nvmlDeviceModifyDrainState

    global __nvmlDeviceQueryDrainState
    data["__nvmlDeviceQueryDrainState"] = <intptr_t>__nvmlDeviceQueryDrainState

    global __nvmlDeviceRemoveGpu_v2
    data["__nvmlDeviceRemoveGpu_v2"] = <intptr_t>__nvmlDeviceRemoveGpu_v2

    global __nvmlDeviceDiscoverGpus
    data["__nvmlDeviceDiscoverGpus"] = <intptr_t>__nvmlDeviceDiscoverGpus

    global __nvmlDeviceGetFieldValues
    data["__nvmlDeviceGetFieldValues"] = <intptr_t>__nvmlDeviceGetFieldValues

    global __nvmlDeviceClearFieldValues
    data["__nvmlDeviceClearFieldValues"] = <intptr_t>__nvmlDeviceClearFieldValues

    global __nvmlDeviceGetVirtualizationMode
    data["__nvmlDeviceGetVirtualizationMode"] = <intptr_t>__nvmlDeviceGetVirtualizationMode

    global __nvmlDeviceGetHostVgpuMode
    data["__nvmlDeviceGetHostVgpuMode"] = <intptr_t>__nvmlDeviceGetHostVgpuMode

    global __nvmlDeviceSetVirtualizationMode
    data["__nvmlDeviceSetVirtualizationMode"] = <intptr_t>__nvmlDeviceSetVirtualizationMode

    global __nvmlDeviceGetVgpuHeterogeneousMode
    data["__nvmlDeviceGetVgpuHeterogeneousMode"] = <intptr_t>__nvmlDeviceGetVgpuHeterogeneousMode

    global __nvmlDeviceSetVgpuHeterogeneousMode
    data["__nvmlDeviceSetVgpuHeterogeneousMode"] = <intptr_t>__nvmlDeviceSetVgpuHeterogeneousMode

    global __nvmlVgpuInstanceGetPlacementId
    data["__nvmlVgpuInstanceGetPlacementId"] = <intptr_t>__nvmlVgpuInstanceGetPlacementId

    global __nvmlDeviceGetVgpuTypeSupportedPlacements
    data["__nvmlDeviceGetVgpuTypeSupportedPlacements"] = <intptr_t>__nvmlDeviceGetVgpuTypeSupportedPlacements

    global __nvmlDeviceGetVgpuTypeCreatablePlacements
    data["__nvmlDeviceGetVgpuTypeCreatablePlacements"] = <intptr_t>__nvmlDeviceGetVgpuTypeCreatablePlacements

    global __nvmlVgpuTypeGetGspHeapSize
    data["__nvmlVgpuTypeGetGspHeapSize"] = <intptr_t>__nvmlVgpuTypeGetGspHeapSize

    global __nvmlVgpuTypeGetFbReservation
    data["__nvmlVgpuTypeGetFbReservation"] = <intptr_t>__nvmlVgpuTypeGetFbReservation

    global __nvmlVgpuInstanceGetRuntimeStateSize
    data["__nvmlVgpuInstanceGetRuntimeStateSize"] = <intptr_t>__nvmlVgpuInstanceGetRuntimeStateSize

    global __nvmlDeviceSetVgpuCapabilities
    data["__nvmlDeviceSetVgpuCapabilities"] = <intptr_t>__nvmlDeviceSetVgpuCapabilities

    global __nvmlDeviceGetGridLicensableFeatures_v4
    data["__nvmlDeviceGetGridLicensableFeatures_v4"] = <intptr_t>__nvmlDeviceGetGridLicensableFeatures_v4

    global __nvmlGetVgpuDriverCapabilities
    data["__nvmlGetVgpuDriverCapabilities"] = <intptr_t>__nvmlGetVgpuDriverCapabilities

    global __nvmlDeviceGetVgpuCapabilities
    data["__nvmlDeviceGetVgpuCapabilities"] = <intptr_t>__nvmlDeviceGetVgpuCapabilities

    global __nvmlDeviceGetSupportedVgpus
    data["__nvmlDeviceGetSupportedVgpus"] = <intptr_t>__nvmlDeviceGetSupportedVgpus

    global __nvmlDeviceGetCreatableVgpus
    data["__nvmlDeviceGetCreatableVgpus"] = <intptr_t>__nvmlDeviceGetCreatableVgpus

    global __nvmlVgpuTypeGetClass
    data["__nvmlVgpuTypeGetClass"] = <intptr_t>__nvmlVgpuTypeGetClass

    global __nvmlVgpuTypeGetName
    data["__nvmlVgpuTypeGetName"] = <intptr_t>__nvmlVgpuTypeGetName

    global __nvmlVgpuTypeGetGpuInstanceProfileId
    data["__nvmlVgpuTypeGetGpuInstanceProfileId"] = <intptr_t>__nvmlVgpuTypeGetGpuInstanceProfileId

    global __nvmlVgpuTypeGetDeviceID
    data["__nvmlVgpuTypeGetDeviceID"] = <intptr_t>__nvmlVgpuTypeGetDeviceID

    global __nvmlVgpuTypeGetFramebufferSize
    data["__nvmlVgpuTypeGetFramebufferSize"] = <intptr_t>__nvmlVgpuTypeGetFramebufferSize

    global __nvmlVgpuTypeGetNumDisplayHeads
    data["__nvmlVgpuTypeGetNumDisplayHeads"] = <intptr_t>__nvmlVgpuTypeGetNumDisplayHeads

    global __nvmlVgpuTypeGetResolution
    data["__nvmlVgpuTypeGetResolution"] = <intptr_t>__nvmlVgpuTypeGetResolution

    global __nvmlVgpuTypeGetLicense
    data["__nvmlVgpuTypeGetLicense"] = <intptr_t>__nvmlVgpuTypeGetLicense

    global __nvmlVgpuTypeGetFrameRateLimit
    data["__nvmlVgpuTypeGetFrameRateLimit"] = <intptr_t>__nvmlVgpuTypeGetFrameRateLimit

    global __nvmlVgpuTypeGetMaxInstances
    data["__nvmlVgpuTypeGetMaxInstances"] = <intptr_t>__nvmlVgpuTypeGetMaxInstances

    global __nvmlVgpuTypeGetMaxInstancesPerVm
    data["__nvmlVgpuTypeGetMaxInstancesPerVm"] = <intptr_t>__nvmlVgpuTypeGetMaxInstancesPerVm

    global __nvmlVgpuTypeGetBAR1Info
    data["__nvmlVgpuTypeGetBAR1Info"] = <intptr_t>__nvmlVgpuTypeGetBAR1Info

    global __nvmlDeviceGetActiveVgpus
    data["__nvmlDeviceGetActiveVgpus"] = <intptr_t>__nvmlDeviceGetActiveVgpus

    global __nvmlVgpuInstanceGetVmID
    data["__nvmlVgpuInstanceGetVmID"] = <intptr_t>__nvmlVgpuInstanceGetVmID

    global __nvmlVgpuInstanceGetUUID
    data["__nvmlVgpuInstanceGetUUID"] = <intptr_t>__nvmlVgpuInstanceGetUUID

    global __nvmlVgpuInstanceGetVmDriverVersion
    data["__nvmlVgpuInstanceGetVmDriverVersion"] = <intptr_t>__nvmlVgpuInstanceGetVmDriverVersion

    global __nvmlVgpuInstanceGetFbUsage
    data["__nvmlVgpuInstanceGetFbUsage"] = <intptr_t>__nvmlVgpuInstanceGetFbUsage

    global __nvmlVgpuInstanceGetLicenseStatus
    data["__nvmlVgpuInstanceGetLicenseStatus"] = <intptr_t>__nvmlVgpuInstanceGetLicenseStatus

    global __nvmlVgpuInstanceGetType
    data["__nvmlVgpuInstanceGetType"] = <intptr_t>__nvmlVgpuInstanceGetType

    global __nvmlVgpuInstanceGetFrameRateLimit
    data["__nvmlVgpuInstanceGetFrameRateLimit"] = <intptr_t>__nvmlVgpuInstanceGetFrameRateLimit

    global __nvmlVgpuInstanceGetEccMode
    data["__nvmlVgpuInstanceGetEccMode"] = <intptr_t>__nvmlVgpuInstanceGetEccMode

    global __nvmlVgpuInstanceGetEncoderCapacity
    data["__nvmlVgpuInstanceGetEncoderCapacity"] = <intptr_t>__nvmlVgpuInstanceGetEncoderCapacity

    global __nvmlVgpuInstanceSetEncoderCapacity
    data["__nvmlVgpuInstanceSetEncoderCapacity"] = <intptr_t>__nvmlVgpuInstanceSetEncoderCapacity

    global __nvmlVgpuInstanceGetEncoderStats
    data["__nvmlVgpuInstanceGetEncoderStats"] = <intptr_t>__nvmlVgpuInstanceGetEncoderStats

    global __nvmlVgpuInstanceGetEncoderSessions
    data["__nvmlVgpuInstanceGetEncoderSessions"] = <intptr_t>__nvmlVgpuInstanceGetEncoderSessions

    global __nvmlVgpuInstanceGetFBCStats
    data["__nvmlVgpuInstanceGetFBCStats"] = <intptr_t>__nvmlVgpuInstanceGetFBCStats

    global __nvmlVgpuInstanceGetFBCSessions
    data["__nvmlVgpuInstanceGetFBCSessions"] = <intptr_t>__nvmlVgpuInstanceGetFBCSessions

    global __nvmlVgpuInstanceGetGpuInstanceId
    data["__nvmlVgpuInstanceGetGpuInstanceId"] = <intptr_t>__nvmlVgpuInstanceGetGpuInstanceId

    global __nvmlVgpuInstanceGetGpuPciId
    data["__nvmlVgpuInstanceGetGpuPciId"] = <intptr_t>__nvmlVgpuInstanceGetGpuPciId

    global __nvmlVgpuTypeGetCapabilities
    data["__nvmlVgpuTypeGetCapabilities"] = <intptr_t>__nvmlVgpuTypeGetCapabilities

    global __nvmlVgpuInstanceGetMdevUUID
    data["__nvmlVgpuInstanceGetMdevUUID"] = <intptr_t>__nvmlVgpuInstanceGetMdevUUID

    global __nvmlGpuInstanceGetCreatableVgpus
    data["__nvmlGpuInstanceGetCreatableVgpus"] = <intptr_t>__nvmlGpuInstanceGetCreatableVgpus

    global __nvmlVgpuTypeGetMaxInstancesPerGpuInstance
    data["__nvmlVgpuTypeGetMaxInstancesPerGpuInstance"] = <intptr_t>__nvmlVgpuTypeGetMaxInstancesPerGpuInstance

    global __nvmlGpuInstanceGetActiveVgpus
    data["__nvmlGpuInstanceGetActiveVgpus"] = <intptr_t>__nvmlGpuInstanceGetActiveVgpus

    global __nvmlGpuInstanceSetVgpuSchedulerState
    data["__nvmlGpuInstanceSetVgpuSchedulerState"] = <intptr_t>__nvmlGpuInstanceSetVgpuSchedulerState

    global __nvmlGpuInstanceGetVgpuSchedulerState
    data["__nvmlGpuInstanceGetVgpuSchedulerState"] = <intptr_t>__nvmlGpuInstanceGetVgpuSchedulerState

    global __nvmlGpuInstanceGetVgpuSchedulerLog
    data["__nvmlGpuInstanceGetVgpuSchedulerLog"] = <intptr_t>__nvmlGpuInstanceGetVgpuSchedulerLog

    global __nvmlGpuInstanceGetVgpuTypeCreatablePlacements
    data["__nvmlGpuInstanceGetVgpuTypeCreatablePlacements"] = <intptr_t>__nvmlGpuInstanceGetVgpuTypeCreatablePlacements

    global __nvmlGpuInstanceGetVgpuHeterogeneousMode
    data["__nvmlGpuInstanceGetVgpuHeterogeneousMode"] = <intptr_t>__nvmlGpuInstanceGetVgpuHeterogeneousMode

    global __nvmlGpuInstanceSetVgpuHeterogeneousMode
    data["__nvmlGpuInstanceSetVgpuHeterogeneousMode"] = <intptr_t>__nvmlGpuInstanceSetVgpuHeterogeneousMode

    global __nvmlVgpuInstanceGetMetadata
    data["__nvmlVgpuInstanceGetMetadata"] = <intptr_t>__nvmlVgpuInstanceGetMetadata

    global __nvmlDeviceGetVgpuMetadata
    data["__nvmlDeviceGetVgpuMetadata"] = <intptr_t>__nvmlDeviceGetVgpuMetadata

    global __nvmlGetVgpuCompatibility
    data["__nvmlGetVgpuCompatibility"] = <intptr_t>__nvmlGetVgpuCompatibility

    global __nvmlDeviceGetPgpuMetadataString
    data["__nvmlDeviceGetPgpuMetadataString"] = <intptr_t>__nvmlDeviceGetPgpuMetadataString

    global __nvmlDeviceGetVgpuSchedulerLog
    data["__nvmlDeviceGetVgpuSchedulerLog"] = <intptr_t>__nvmlDeviceGetVgpuSchedulerLog

    global __nvmlDeviceGetVgpuSchedulerState
    data["__nvmlDeviceGetVgpuSchedulerState"] = <intptr_t>__nvmlDeviceGetVgpuSchedulerState

    global __nvmlDeviceGetVgpuSchedulerCapabilities
    data["__nvmlDeviceGetVgpuSchedulerCapabilities"] = <intptr_t>__nvmlDeviceGetVgpuSchedulerCapabilities

    global __nvmlDeviceSetVgpuSchedulerState
    data["__nvmlDeviceSetVgpuSchedulerState"] = <intptr_t>__nvmlDeviceSetVgpuSchedulerState

    global __nvmlGetVgpuVersion
    data["__nvmlGetVgpuVersion"] = <intptr_t>__nvmlGetVgpuVersion

    global __nvmlSetVgpuVersion
    data["__nvmlSetVgpuVersion"] = <intptr_t>__nvmlSetVgpuVersion

    global __nvmlDeviceGetVgpuUtilization
    data["__nvmlDeviceGetVgpuUtilization"] = <intptr_t>__nvmlDeviceGetVgpuUtilization

    global __nvmlDeviceGetVgpuInstancesUtilizationInfo
    data["__nvmlDeviceGetVgpuInstancesUtilizationInfo"] = <intptr_t>__nvmlDeviceGetVgpuInstancesUtilizationInfo

    global __nvmlDeviceGetVgpuProcessUtilization
    data["__nvmlDeviceGetVgpuProcessUtilization"] = <intptr_t>__nvmlDeviceGetVgpuProcessUtilization

    global __nvmlDeviceGetVgpuProcessesUtilizationInfo
    data["__nvmlDeviceGetVgpuProcessesUtilizationInfo"] = <intptr_t>__nvmlDeviceGetVgpuProcessesUtilizationInfo

    global __nvmlVgpuInstanceGetAccountingMode
    data["__nvmlVgpuInstanceGetAccountingMode"] = <intptr_t>__nvmlVgpuInstanceGetAccountingMode

    global __nvmlVgpuInstanceGetAccountingPids
    data["__nvmlVgpuInstanceGetAccountingPids"] = <intptr_t>__nvmlVgpuInstanceGetAccountingPids

    global __nvmlVgpuInstanceGetAccountingStats
    data["__nvmlVgpuInstanceGetAccountingStats"] = <intptr_t>__nvmlVgpuInstanceGetAccountingStats

    global __nvmlVgpuInstanceClearAccountingPids
    data["__nvmlVgpuInstanceClearAccountingPids"] = <intptr_t>__nvmlVgpuInstanceClearAccountingPids

    global __nvmlVgpuInstanceGetLicenseInfo_v2
    data["__nvmlVgpuInstanceGetLicenseInfo_v2"] = <intptr_t>__nvmlVgpuInstanceGetLicenseInfo_v2

    global __nvmlGetExcludedDeviceCount
    data["__nvmlGetExcludedDeviceCount"] = <intptr_t>__nvmlGetExcludedDeviceCount

    global __nvmlGetExcludedDeviceInfoByIndex
    data["__nvmlGetExcludedDeviceInfoByIndex"] = <intptr_t>__nvmlGetExcludedDeviceInfoByIndex

    global __nvmlDeviceSetMigMode
    data["__nvmlDeviceSetMigMode"] = <intptr_t>__nvmlDeviceSetMigMode

    global __nvmlDeviceGetMigMode
    data["__nvmlDeviceGetMigMode"] = <intptr_t>__nvmlDeviceGetMigMode

    global __nvmlDeviceGetGpuInstanceProfileInfoV
    data["__nvmlDeviceGetGpuInstanceProfileInfoV"] = <intptr_t>__nvmlDeviceGetGpuInstanceProfileInfoV

    global __nvmlDeviceGetGpuInstancePossiblePlacements_v2
    data["__nvmlDeviceGetGpuInstancePossiblePlacements_v2"] = <intptr_t>__nvmlDeviceGetGpuInstancePossiblePlacements_v2

    global __nvmlDeviceGetGpuInstanceRemainingCapacity
    data["__nvmlDeviceGetGpuInstanceRemainingCapacity"] = <intptr_t>__nvmlDeviceGetGpuInstanceRemainingCapacity

    global __nvmlDeviceCreateGpuInstance
    data["__nvmlDeviceCreateGpuInstance"] = <intptr_t>__nvmlDeviceCreateGpuInstance

    global __nvmlDeviceCreateGpuInstanceWithPlacement
    data["__nvmlDeviceCreateGpuInstanceWithPlacement"] = <intptr_t>__nvmlDeviceCreateGpuInstanceWithPlacement

    global __nvmlGpuInstanceDestroy
    data["__nvmlGpuInstanceDestroy"] = <intptr_t>__nvmlGpuInstanceDestroy

    global __nvmlDeviceGetGpuInstances
    data["__nvmlDeviceGetGpuInstances"] = <intptr_t>__nvmlDeviceGetGpuInstances

    global __nvmlDeviceGetGpuInstanceById
    data["__nvmlDeviceGetGpuInstanceById"] = <intptr_t>__nvmlDeviceGetGpuInstanceById

    global __nvmlGpuInstanceGetInfo
    data["__nvmlGpuInstanceGetInfo"] = <intptr_t>__nvmlGpuInstanceGetInfo

    global __nvmlGpuInstanceGetComputeInstanceProfileInfoV
    data["__nvmlGpuInstanceGetComputeInstanceProfileInfoV"] = <intptr_t>__nvmlGpuInstanceGetComputeInstanceProfileInfoV

    global __nvmlGpuInstanceGetComputeInstanceRemainingCapacity
    data["__nvmlGpuInstanceGetComputeInstanceRemainingCapacity"] = <intptr_t>__nvmlGpuInstanceGetComputeInstanceRemainingCapacity

    global __nvmlGpuInstanceGetComputeInstancePossiblePlacements
    data["__nvmlGpuInstanceGetComputeInstancePossiblePlacements"] = <intptr_t>__nvmlGpuInstanceGetComputeInstancePossiblePlacements

    global __nvmlGpuInstanceCreateComputeInstance
    data["__nvmlGpuInstanceCreateComputeInstance"] = <intptr_t>__nvmlGpuInstanceCreateComputeInstance

    global __nvmlGpuInstanceCreateComputeInstanceWithPlacement
    data["__nvmlGpuInstanceCreateComputeInstanceWithPlacement"] = <intptr_t>__nvmlGpuInstanceCreateComputeInstanceWithPlacement

    global __nvmlComputeInstanceDestroy
    data["__nvmlComputeInstanceDestroy"] = <intptr_t>__nvmlComputeInstanceDestroy

    global __nvmlGpuInstanceGetComputeInstances
    data["__nvmlGpuInstanceGetComputeInstances"] = <intptr_t>__nvmlGpuInstanceGetComputeInstances

    global __nvmlGpuInstanceGetComputeInstanceById
    data["__nvmlGpuInstanceGetComputeInstanceById"] = <intptr_t>__nvmlGpuInstanceGetComputeInstanceById

    global __nvmlComputeInstanceGetInfo_v2
    data["__nvmlComputeInstanceGetInfo_v2"] = <intptr_t>__nvmlComputeInstanceGetInfo_v2

    global __nvmlDeviceIsMigDeviceHandle
    data["__nvmlDeviceIsMigDeviceHandle"] = <intptr_t>__nvmlDeviceIsMigDeviceHandle

    global __nvmlDeviceGetGpuInstanceId
    data["__nvmlDeviceGetGpuInstanceId"] = <intptr_t>__nvmlDeviceGetGpuInstanceId

    global __nvmlDeviceGetComputeInstanceId
    data["__nvmlDeviceGetComputeInstanceId"] = <intptr_t>__nvmlDeviceGetComputeInstanceId

    global __nvmlDeviceGetMaxMigDeviceCount
    data["__nvmlDeviceGetMaxMigDeviceCount"] = <intptr_t>__nvmlDeviceGetMaxMigDeviceCount

    global __nvmlDeviceGetMigDeviceHandleByIndex
    data["__nvmlDeviceGetMigDeviceHandleByIndex"] = <intptr_t>__nvmlDeviceGetMigDeviceHandleByIndex

    global __nvmlDeviceGetDeviceHandleFromMigDeviceHandle
    data["__nvmlDeviceGetDeviceHandleFromMigDeviceHandle"] = <intptr_t>__nvmlDeviceGetDeviceHandleFromMigDeviceHandle

    global __nvmlDeviceGetCapabilities
    data["__nvmlDeviceGetCapabilities"] = <intptr_t>__nvmlDeviceGetCapabilities

    global __nvmlDevicePowerSmoothingActivatePresetProfile
    data["__nvmlDevicePowerSmoothingActivatePresetProfile"] = <intptr_t>__nvmlDevicePowerSmoothingActivatePresetProfile

    global __nvmlDevicePowerSmoothingUpdatePresetProfileParam
    data["__nvmlDevicePowerSmoothingUpdatePresetProfileParam"] = <intptr_t>__nvmlDevicePowerSmoothingUpdatePresetProfileParam

    global __nvmlDevicePowerSmoothingSetState
    data["__nvmlDevicePowerSmoothingSetState"] = <intptr_t>__nvmlDevicePowerSmoothingSetState

    global __nvmlDeviceGetAddressingMode
    data["__nvmlDeviceGetAddressingMode"] = <intptr_t>__nvmlDeviceGetAddressingMode

    global __nvmlDeviceGetRepairStatus
    data["__nvmlDeviceGetRepairStatus"] = <intptr_t>__nvmlDeviceGetRepairStatus

    global __nvmlDeviceGetPowerMizerMode_v1
    data["__nvmlDeviceGetPowerMizerMode_v1"] = <intptr_t>__nvmlDeviceGetPowerMizerMode_v1

    global __nvmlDeviceSetPowerMizerMode_v1
    data["__nvmlDeviceSetPowerMizerMode_v1"] = <intptr_t>__nvmlDeviceSetPowerMizerMode_v1

    global __nvmlDeviceGetPdi
    data["__nvmlDeviceGetPdi"] = <intptr_t>__nvmlDeviceGetPdi

    global __nvmlDeviceSetHostname_v1
    data["__nvmlDeviceSetHostname_v1"] = <intptr_t>__nvmlDeviceSetHostname_v1

    global __nvmlDeviceGetHostname_v1
    data["__nvmlDeviceGetHostname_v1"] = <intptr_t>__nvmlDeviceGetHostname_v1

    global __nvmlDeviceGetNvLinkInfo
    data["__nvmlDeviceGetNvLinkInfo"] = <intptr_t>__nvmlDeviceGetNvLinkInfo

    global __nvmlDeviceReadWritePRM_v1
    data["__nvmlDeviceReadWritePRM_v1"] = <intptr_t>__nvmlDeviceReadWritePRM_v1

    global __nvmlDeviceGetGpuInstanceProfileInfoByIdV
    data["__nvmlDeviceGetGpuInstanceProfileInfoByIdV"] = <intptr_t>__nvmlDeviceGetGpuInstanceProfileInfoByIdV

    global __nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts
    data["__nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts"] = <intptr_t>__nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts

    global __nvmlDeviceGetUnrepairableMemoryFlag_v1
    data["__nvmlDeviceGetUnrepairableMemoryFlag_v1"] = <intptr_t>__nvmlDeviceGetUnrepairableMemoryFlag_v1

    global __nvmlDeviceReadPRMCounters_v1
    data["__nvmlDeviceReadPRMCounters_v1"] = <intptr_t>__nvmlDeviceReadPRMCounters_v1

    global __nvmlDeviceSetRusdSettings_v1
    data["__nvmlDeviceSetRusdSettings_v1"] = <intptr_t>__nvmlDeviceSetRusdSettings_v1

    func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global func_ptrs
    if func_ptrs is None:
        func_ptrs = _inspect_function_pointers()
    return func_ptrs[name]


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
