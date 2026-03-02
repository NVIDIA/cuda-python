# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.9.1 to 13.1.1, generator version 0.3.1.dev1322+g646ce84ec. Do not modify it directly.

from ._internal cimport nvml as _nvml


###############################################################################
# Wrapper functions
###############################################################################

cdef nvmlReturn_t nvmlInit_v2() except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlInit_v2()


cdef nvmlReturn_t nvmlInitWithFlags(unsigned int flags) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlInitWithFlags(flags)


cdef nvmlReturn_t nvmlShutdown() except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlShutdown()


cdef const char* nvmlErrorString(nvmlReturn_t result) except?NULL nogil:
    return _nvml._nvmlErrorString(result)


cdef nvmlReturn_t nvmlSystemGetDriverVersion(char* version, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSystemGetDriverVersion(version, length)


cdef nvmlReturn_t nvmlSystemGetNVMLVersion(char* version, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSystemGetNVMLVersion(version, length)


cdef nvmlReturn_t nvmlSystemGetCudaDriverVersion(int* cudaDriverVersion) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSystemGetCudaDriverVersion(cudaDriverVersion)


cdef nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int* cudaDriverVersion) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSystemGetCudaDriverVersion_v2(cudaDriverVersion)


cdef nvmlReturn_t nvmlSystemGetProcessName(unsigned int pid, char* name, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSystemGetProcessName(pid, name, length)


cdef nvmlReturn_t nvmlSystemGetHicVersion(unsigned int* hwbcCount, nvmlHwbcEntry_t* hwbcEntries) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSystemGetHicVersion(hwbcCount, hwbcEntries)


cdef nvmlReturn_t nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber, unsigned int* count, nvmlDevice_t* deviceArray) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSystemGetTopologyGpuSet(cpuNumber, count, deviceArray)


cdef nvmlReturn_t nvmlSystemGetDriverBranch(nvmlSystemDriverBranchInfo_t* branchInfo, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSystemGetDriverBranch(branchInfo, length)


cdef nvmlReturn_t nvmlUnitGetCount(unsigned int* unitCount) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlUnitGetCount(unitCount)


cdef nvmlReturn_t nvmlUnitGetHandleByIndex(unsigned int index, nvmlUnit_t* unit) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlUnitGetHandleByIndex(index, unit)


cdef nvmlReturn_t nvmlUnitGetUnitInfo(nvmlUnit_t unit, nvmlUnitInfo_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlUnitGetUnitInfo(unit, info)


cdef nvmlReturn_t nvmlUnitGetLedState(nvmlUnit_t unit, nvmlLedState_t* state) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlUnitGetLedState(unit, state)


cdef nvmlReturn_t nvmlUnitGetPsuInfo(nvmlUnit_t unit, nvmlPSUInfo_t* psu) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlUnitGetPsuInfo(unit, psu)


cdef nvmlReturn_t nvmlUnitGetTemperature(nvmlUnit_t unit, unsigned int type, unsigned int* temp) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlUnitGetTemperature(unit, type, temp)


cdef nvmlReturn_t nvmlUnitGetFanSpeedInfo(nvmlUnit_t unit, nvmlUnitFanSpeeds_t* fanSpeeds) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlUnitGetFanSpeedInfo(unit, fanSpeeds)


cdef nvmlReturn_t nvmlUnitGetDevices(nvmlUnit_t unit, unsigned int* deviceCount, nvmlDevice_t* devices) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlUnitGetDevices(unit, deviceCount, devices)


cdef nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* deviceCount) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetCount_v2(deviceCount)


cdef nvmlReturn_t nvmlDeviceGetAttributes_v2(nvmlDevice_t device, nvmlDeviceAttributes_t* attributes) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetAttributes_v2(device, attributes)


cdef nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t* device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetHandleByIndex_v2(index, device)


cdef nvmlReturn_t nvmlDeviceGetHandleBySerial(const char* serial, nvmlDevice_t* device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetHandleBySerial(serial, device)


cdef nvmlReturn_t nvmlDeviceGetHandleByUUID(const char* uuid, nvmlDevice_t* device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetHandleByUUID(uuid, device)


cdef nvmlReturn_t nvmlDeviceGetHandleByUUIDV(const nvmlUUID_t* uuid, nvmlDevice_t* device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetHandleByUUIDV(uuid, device)


cdef nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char* pciBusId, nvmlDevice_t* device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetHandleByPciBusId_v2(pciBusId, device)


cdef nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetName(device, name, length)


cdef nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device, nvmlBrandType_t* type) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetBrand(device, type)


cdef nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetIndex(device, index)


cdef nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t device, char* serial, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetSerial(device, serial, length)


cdef nvmlReturn_t nvmlDeviceGetModuleId(nvmlDevice_t device, unsigned int* moduleId) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetModuleId(device, moduleId)


cdef nvmlReturn_t nvmlDeviceGetC2cModeInfoV(nvmlDevice_t device, nvmlC2cModeInfo_v1_t* c2cModeInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetC2cModeInfoV(device, c2cModeInfo)


cdef nvmlReturn_t nvmlDeviceGetMemoryAffinity(nvmlDevice_t device, unsigned int nodeSetSize, unsigned long* nodeSet, nvmlAffinityScope_t scope) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetMemoryAffinity(device, nodeSetSize, nodeSet, scope)


cdef nvmlReturn_t nvmlDeviceGetCpuAffinityWithinScope(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long* cpuSet, nvmlAffinityScope_t scope) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetCpuAffinityWithinScope(device, cpuSetSize, cpuSet, scope)


cdef nvmlReturn_t nvmlDeviceGetCpuAffinity(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long* cpuSet) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetCpuAffinity(device, cpuSetSize, cpuSet)


cdef nvmlReturn_t nvmlDeviceSetCpuAffinity(nvmlDevice_t device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetCpuAffinity(device)


cdef nvmlReturn_t nvmlDeviceClearCpuAffinity(nvmlDevice_t device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceClearCpuAffinity(device)


cdef nvmlReturn_t nvmlDeviceGetNumaNodeId(nvmlDevice_t device, unsigned int* node) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetNumaNodeId(device, node)


cdef nvmlReturn_t nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuTopologyLevel_t* pathInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetTopologyCommonAncestor(device1, device2, pathInfo)


cdef nvmlReturn_t nvmlDeviceGetTopologyNearestGpus(nvmlDevice_t device, nvmlGpuTopologyLevel_t level, unsigned int* count, nvmlDevice_t* deviceArray) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetTopologyNearestGpus(device, level, count, deviceArray)


cdef nvmlReturn_t nvmlDeviceGetP2PStatus(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex, nvmlGpuP2PStatus_t* p2pStatus) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetP2PStatus(device1, device2, p2pIndex, p2pStatus)


cdef nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char* uuid, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetUUID(device, uuid, length)


cdef nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int* minorNumber) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetMinorNumber(device, minorNumber)


cdef nvmlReturn_t nvmlDeviceGetBoardPartNumber(nvmlDevice_t device, char* partNumber, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetBoardPartNumber(device, partNumber, length)


cdef nvmlReturn_t nvmlDeviceGetInforomVersion(nvmlDevice_t device, nvmlInforomObject_t object, char* version, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetInforomVersion(device, object, version, length)


cdef nvmlReturn_t nvmlDeviceGetInforomImageVersion(nvmlDevice_t device, char* version, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetInforomImageVersion(device, version, length)


cdef nvmlReturn_t nvmlDeviceGetInforomConfigurationChecksum(nvmlDevice_t device, unsigned int* checksum) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetInforomConfigurationChecksum(device, checksum)


cdef nvmlReturn_t nvmlDeviceValidateInforom(nvmlDevice_t device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceValidateInforom(device)


cdef nvmlReturn_t nvmlDeviceGetLastBBXFlushTime(nvmlDevice_t device, unsigned long long* timestamp, unsigned long* durationUs) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetLastBBXFlushTime(device, timestamp, durationUs)


cdef nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device, nvmlEnableState_t* display) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetDisplayMode(device, display)


cdef nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, nvmlEnableState_t* isActive) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetDisplayActive(device, isActive)


cdef nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t* mode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetPersistenceMode(device, mode)


cdef nvmlReturn_t nvmlDeviceGetPciInfoExt(nvmlDevice_t device, nvmlPciInfoExt_t* pci) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetPciInfoExt(device, pci)


cdef nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t* pci) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetPciInfo_v3(device, pci)


cdef nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int* maxLinkGen) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetMaxPcieLinkGeneration(device, maxLinkGen)


cdef nvmlReturn_t nvmlDeviceGetGpuMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int* maxLinkGenDevice) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetGpuMaxPcieLinkGeneration(device, maxLinkGenDevice)


cdef nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device, unsigned int* maxLinkWidth) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetMaxPcieLinkWidth(device, maxLinkWidth)


cdef nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int* currLinkGen) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetCurrPcieLinkGeneration(device, currLinkGen)


cdef nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int* currLinkWidth) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetCurrPcieLinkWidth(device, currLinkWidth)


cdef nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int* value) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetPcieThroughput(device, counter, value)


cdef nvmlReturn_t nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device, unsigned int* value) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetPcieReplayCounter(device, value)


cdef nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetClockInfo(device, type, clock)


cdef nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetMaxClockInfo(device, type, clock)


cdef nvmlReturn_t nvmlDeviceGetGpcClkVfOffset(nvmlDevice_t device, int* offset) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetGpcClkVfOffset(device, offset)


cdef nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int* clockMHz) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetClock(device, clockType, clockId, clockMHz)


cdef nvmlReturn_t nvmlDeviceGetMaxCustomerBoostClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int* clockMHz) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetMaxCustomerBoostClock(device, clockType, clockMHz)


cdef nvmlReturn_t nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device, unsigned int* count, unsigned int* clocksMHz) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetSupportedMemoryClocks(device, count, clocksMHz)


cdef nvmlReturn_t nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t device, unsigned int memoryClockMHz, unsigned int* count, unsigned int* clocksMHz) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetSupportedGraphicsClocks(device, memoryClockMHz, count, clocksMHz)


cdef nvmlReturn_t nvmlDeviceGetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t* isEnabled, nvmlEnableState_t* defaultIsEnabled) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetAutoBoostedClocksEnabled(device, isEnabled, defaultIsEnabled)


cdef nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int* speed) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetFanSpeed(device, speed)


cdef nvmlReturn_t nvmlDeviceGetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int* speed) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetFanSpeed_v2(device, fan, speed)


cdef nvmlReturn_t nvmlDeviceGetFanSpeedRPM(nvmlDevice_t device, nvmlFanSpeedInfo_t* fanSpeed) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetFanSpeedRPM(device, fanSpeed)


cdef nvmlReturn_t nvmlDeviceGetTargetFanSpeed(nvmlDevice_t device, unsigned int fan, unsigned int* targetSpeed) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetTargetFanSpeed(device, fan, targetSpeed)


cdef nvmlReturn_t nvmlDeviceGetMinMaxFanSpeed(nvmlDevice_t device, unsigned int* minSpeed, unsigned int* maxSpeed) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetMinMaxFanSpeed(device, minSpeed, maxSpeed)


cdef nvmlReturn_t nvmlDeviceGetFanControlPolicy_v2(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t* policy) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetFanControlPolicy_v2(device, fan, policy)


cdef nvmlReturn_t nvmlDeviceGetNumFans(nvmlDevice_t device, unsigned int* numFans) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetNumFans(device, numFans)


cdef nvmlReturn_t nvmlDeviceGetCoolerInfo(nvmlDevice_t device, nvmlCoolerInfo_t* coolerInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetCoolerInfo(device, coolerInfo)


cdef nvmlReturn_t nvmlDeviceGetTemperatureV(nvmlDevice_t device, nvmlTemperature_t* temperature) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetTemperatureV(device, temperature)


cdef nvmlReturn_t nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, unsigned int* temp) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetTemperatureThreshold(device, thresholdType, temp)


cdef nvmlReturn_t nvmlDeviceGetMarginTemperature(nvmlDevice_t device, nvmlMarginTemperature_t* marginTempInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetMarginTemperature(device, marginTempInfo)


cdef nvmlReturn_t nvmlDeviceGetThermalSettings(nvmlDevice_t device, unsigned int sensorIndex, nvmlGpuThermalSettings_t* pThermalSettings) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetThermalSettings(device, sensorIndex, pThermalSettings)


cdef nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t* pState) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetPerformanceState(device, pState)


cdef nvmlReturn_t nvmlDeviceGetCurrentClocksEventReasons(nvmlDevice_t device, unsigned long long* clocksEventReasons) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetCurrentClocksEventReasons(device, clocksEventReasons)


cdef nvmlReturn_t nvmlDeviceGetSupportedClocksEventReasons(nvmlDevice_t device, unsigned long long* supportedClocksEventReasons) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetSupportedClocksEventReasons(device, supportedClocksEventReasons)


cdef nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t device, nvmlPstates_t* pState) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetPowerState(device, pState)


cdef nvmlReturn_t nvmlDeviceGetDynamicPstatesInfo(nvmlDevice_t device, nvmlGpuDynamicPstatesInfo_t* pDynamicPstatesInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetDynamicPstatesInfo(device, pDynamicPstatesInfo)


cdef nvmlReturn_t nvmlDeviceGetMemClkVfOffset(nvmlDevice_t device, int* offset) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetMemClkVfOffset(device, offset)


cdef nvmlReturn_t nvmlDeviceGetMinMaxClockOfPState(nvmlDevice_t device, nvmlClockType_t type, nvmlPstates_t pstate, unsigned int* minClockMHz, unsigned int* maxClockMHz) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetMinMaxClockOfPState(device, type, pstate, minClockMHz, maxClockMHz)


cdef nvmlReturn_t nvmlDeviceGetSupportedPerformanceStates(nvmlDevice_t device, nvmlPstates_t* pstates, unsigned int size) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetSupportedPerformanceStates(device, pstates, size)


cdef nvmlReturn_t nvmlDeviceGetGpcClkMinMaxVfOffset(nvmlDevice_t device, int* minOffset, int* maxOffset) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetGpcClkMinMaxVfOffset(device, minOffset, maxOffset)


cdef nvmlReturn_t nvmlDeviceGetMemClkMinMaxVfOffset(nvmlDevice_t device, int* minOffset, int* maxOffset) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetMemClkMinMaxVfOffset(device, minOffset, maxOffset)


cdef nvmlReturn_t nvmlDeviceGetClockOffsets(nvmlDevice_t device, nvmlClockOffset_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetClockOffsets(device, info)


cdef nvmlReturn_t nvmlDeviceSetClockOffsets(nvmlDevice_t device, nvmlClockOffset_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetClockOffsets(device, info)


cdef nvmlReturn_t nvmlDeviceGetPerformanceModes(nvmlDevice_t device, nvmlDevicePerfModes_t* perfModes) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetPerformanceModes(device, perfModes)


cdef nvmlReturn_t nvmlDeviceGetCurrentClockFreqs(nvmlDevice_t device, nvmlDeviceCurrentClockFreqs_t* currentClockFreqs) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetCurrentClockFreqs(device, currentClockFreqs)


cdef nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int* limit) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetPowerManagementLimit(device, limit)


cdef nvmlReturn_t nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device, unsigned int* minLimit, unsigned int* maxLimit) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetPowerManagementLimitConstraints(device, minLimit, maxLimit)


cdef nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device, unsigned int* defaultLimit) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetPowerManagementDefaultLimit(device, defaultLimit)


cdef nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int* power) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetPowerUsage(device, power)


cdef nvmlReturn_t nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t device, unsigned long long* energy) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetTotalEnergyConsumption(device, energy)


cdef nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int* limit) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetEnforcedPowerLimit(device, limit)


cdef nvmlReturn_t nvmlDeviceGetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t* current, nvmlGpuOperationMode_t* pending) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetGpuOperationMode(device, current, pending)


cdef nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t* memory) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetMemoryInfo_v2(device, memory)


cdef nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t* mode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetComputeMode(device, mode)


cdef nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major, int* minor) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetCudaComputeCapability(device, major, minor)


cdef nvmlReturn_t nvmlDeviceGetDramEncryptionMode(nvmlDevice_t device, nvmlDramEncryptionInfo_t* current, nvmlDramEncryptionInfo_t* pending) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetDramEncryptionMode(device, current, pending)


cdef nvmlReturn_t nvmlDeviceSetDramEncryptionMode(nvmlDevice_t device, const nvmlDramEncryptionInfo_t* dramEncryption) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetDramEncryptionMode(device, dramEncryption)


cdef nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, nvmlEnableState_t* current, nvmlEnableState_t* pending) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetEccMode(device, current, pending)


cdef nvmlReturn_t nvmlDeviceGetDefaultEccMode(nvmlDevice_t device, nvmlEnableState_t* defaultMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetDefaultEccMode(device, defaultMode)


cdef nvmlReturn_t nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int* boardId) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetBoardId(device, boardId)


cdef nvmlReturn_t nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device, unsigned int* multiGpuBool) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetMultiGpuBoard(device, multiGpuBool)


cdef nvmlReturn_t nvmlDeviceGetTotalEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long* eccCounts) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetTotalEccErrors(device, errorType, counterType, eccCounts)


cdef nvmlReturn_t nvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlMemoryLocation_t locationType, unsigned long long* count) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetMemoryErrorCounter(device, errorType, counterType, locationType, count)


cdef nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t* utilization) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetUtilizationRates(device, utilization)


cdef nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetEncoderUtilization(device, utilization, samplingPeriodUs)


cdef nvmlReturn_t nvmlDeviceGetEncoderCapacity(nvmlDevice_t device, nvmlEncoderType_t encoderQueryType, unsigned int* encoderCapacity) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetEncoderCapacity(device, encoderQueryType, encoderCapacity)


cdef nvmlReturn_t nvmlDeviceGetEncoderStats(nvmlDevice_t device, unsigned int* sessionCount, unsigned int* averageFps, unsigned int* averageLatency) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetEncoderStats(device, sessionCount, averageFps, averageLatency)


cdef nvmlReturn_t nvmlDeviceGetEncoderSessions(nvmlDevice_t device, unsigned int* sessionCount, nvmlEncoderSessionInfo_t* sessionInfos) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetEncoderSessions(device, sessionCount, sessionInfos)


cdef nvmlReturn_t nvmlDeviceGetDecoderUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetDecoderUtilization(device, utilization, samplingPeriodUs)


cdef nvmlReturn_t nvmlDeviceGetJpgUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetJpgUtilization(device, utilization, samplingPeriodUs)


cdef nvmlReturn_t nvmlDeviceGetOfaUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetOfaUtilization(device, utilization, samplingPeriodUs)


cdef nvmlReturn_t nvmlDeviceGetFBCStats(nvmlDevice_t device, nvmlFBCStats_t* fbcStats) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetFBCStats(device, fbcStats)


cdef nvmlReturn_t nvmlDeviceGetFBCSessions(nvmlDevice_t device, unsigned int* sessionCount, nvmlFBCSessionInfo_t* sessionInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetFBCSessions(device, sessionCount, sessionInfo)


cdef nvmlReturn_t nvmlDeviceGetDriverModel_v2(nvmlDevice_t device, nvmlDriverModel_t* current, nvmlDriverModel_t* pending) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetDriverModel_v2(device, current, pending)


cdef nvmlReturn_t nvmlDeviceGetVbiosVersion(nvmlDevice_t device, char* version, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetVbiosVersion(device, version, length)


cdef nvmlReturn_t nvmlDeviceGetBridgeChipInfo(nvmlDevice_t device, nvmlBridgeChipHierarchy_t* bridgeHierarchy) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetBridgeChipInfo(device, bridgeHierarchy)


cdef nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetComputeRunningProcesses_v3(device, infoCount, infos)


cdef nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetMPSComputeRunningProcesses_v3(device, infoCount, infos)


cdef nvmlReturn_t nvmlDeviceGetRunningProcessDetailList(nvmlDevice_t device, nvmlProcessDetailList_t* plist) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetRunningProcessDetailList(device, plist)


cdef nvmlReturn_t nvmlDeviceOnSameBoard(nvmlDevice_t device1, nvmlDevice_t device2, int* onSameBoard) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceOnSameBoard(device1, device2, onSameBoard)


cdef nvmlReturn_t nvmlDeviceGetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t* isRestricted) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetAPIRestriction(device, apiType, isRestricted)


cdef nvmlReturn_t nvmlDeviceGetSamples(nvmlDevice_t device, nvmlSamplingType_t type, unsigned long long lastSeenTimeStamp, nvmlValueType_t* sampleValType, unsigned int* sampleCount, nvmlSample_t* samples) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetSamples(device, type, lastSeenTimeStamp, sampleValType, sampleCount, samples)


cdef nvmlReturn_t nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t device, nvmlBAR1Memory_t* bar1Memory) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetBAR1MemoryInfo(device, bar1Memory)


cdef nvmlReturn_t nvmlDeviceGetIrqNum(nvmlDevice_t device, unsigned int* irqNum) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetIrqNum(device, irqNum)


cdef nvmlReturn_t nvmlDeviceGetNumGpuCores(nvmlDevice_t device, unsigned int* numCores) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetNumGpuCores(device, numCores)


cdef nvmlReturn_t nvmlDeviceGetPowerSource(nvmlDevice_t device, nvmlPowerSource_t* powerSource) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetPowerSource(device, powerSource)


cdef nvmlReturn_t nvmlDeviceGetMemoryBusWidth(nvmlDevice_t device, unsigned int* busWidth) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetMemoryBusWidth(device, busWidth)


cdef nvmlReturn_t nvmlDeviceGetPcieLinkMaxSpeed(nvmlDevice_t device, unsigned int* maxSpeed) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetPcieLinkMaxSpeed(device, maxSpeed)


cdef nvmlReturn_t nvmlDeviceGetPcieSpeed(nvmlDevice_t device, unsigned int* pcieSpeed) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetPcieSpeed(device, pcieSpeed)


cdef nvmlReturn_t nvmlDeviceGetAdaptiveClockInfoStatus(nvmlDevice_t device, unsigned int* adaptiveClockStatus) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetAdaptiveClockInfoStatus(device, adaptiveClockStatus)


cdef nvmlReturn_t nvmlDeviceGetBusType(nvmlDevice_t device, nvmlBusType_t* type) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetBusType(device, type)


cdef nvmlReturn_t nvmlDeviceGetGpuFabricInfoV(nvmlDevice_t device, nvmlGpuFabricInfoV_t* gpuFabricInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetGpuFabricInfoV(device, gpuFabricInfo)


cdef nvmlReturn_t nvmlSystemGetConfComputeCapabilities(nvmlConfComputeSystemCaps_t* capabilities) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSystemGetConfComputeCapabilities(capabilities)


cdef nvmlReturn_t nvmlSystemGetConfComputeState(nvmlConfComputeSystemState_t* state) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSystemGetConfComputeState(state)


cdef nvmlReturn_t nvmlDeviceGetConfComputeMemSizeInfo(nvmlDevice_t device, nvmlConfComputeMemSizeInfo_t* memInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetConfComputeMemSizeInfo(device, memInfo)


cdef nvmlReturn_t nvmlSystemGetConfComputeGpusReadyState(unsigned int* isAcceptingWork) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSystemGetConfComputeGpusReadyState(isAcceptingWork)


cdef nvmlReturn_t nvmlDeviceGetConfComputeProtectedMemoryUsage(nvmlDevice_t device, nvmlMemory_t* memory) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetConfComputeProtectedMemoryUsage(device, memory)


cdef nvmlReturn_t nvmlDeviceGetConfComputeGpuCertificate(nvmlDevice_t device, nvmlConfComputeGpuCertificate_t* gpuCert) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetConfComputeGpuCertificate(device, gpuCert)


cdef nvmlReturn_t nvmlDeviceGetConfComputeGpuAttestationReport(nvmlDevice_t device, nvmlConfComputeGpuAttestationReport_t* gpuAtstReport) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetConfComputeGpuAttestationReport(device, gpuAtstReport)


cdef nvmlReturn_t nvmlSystemGetConfComputeKeyRotationThresholdInfo(nvmlConfComputeGetKeyRotationThresholdInfo_t* pKeyRotationThrInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSystemGetConfComputeKeyRotationThresholdInfo(pKeyRotationThrInfo)


cdef nvmlReturn_t nvmlDeviceSetConfComputeUnprotectedMemSize(nvmlDevice_t device, unsigned long long sizeKiB) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetConfComputeUnprotectedMemSize(device, sizeKiB)


cdef nvmlReturn_t nvmlSystemSetConfComputeGpusReadyState(unsigned int isAcceptingWork) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSystemSetConfComputeGpusReadyState(isAcceptingWork)


cdef nvmlReturn_t nvmlSystemSetConfComputeKeyRotationThresholdInfo(nvmlConfComputeSetKeyRotationThresholdInfo_t* pKeyRotationThrInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSystemSetConfComputeKeyRotationThresholdInfo(pKeyRotationThrInfo)


cdef nvmlReturn_t nvmlSystemGetConfComputeSettings(nvmlSystemConfComputeSettings_t* settings) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSystemGetConfComputeSettings(settings)


cdef nvmlReturn_t nvmlDeviceGetGspFirmwareVersion(nvmlDevice_t device, char* version) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetGspFirmwareVersion(device, version)


cdef nvmlReturn_t nvmlDeviceGetGspFirmwareMode(nvmlDevice_t device, unsigned int* isEnabled, unsigned int* defaultMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetGspFirmwareMode(device, isEnabled, defaultMode)


cdef nvmlReturn_t nvmlDeviceGetSramEccErrorStatus(nvmlDevice_t device, nvmlEccSramErrorStatus_t* status) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetSramEccErrorStatus(device, status)


cdef nvmlReturn_t nvmlDeviceGetAccountingMode(nvmlDevice_t device, nvmlEnableState_t* mode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetAccountingMode(device, mode)


cdef nvmlReturn_t nvmlDeviceGetAccountingStats(nvmlDevice_t device, unsigned int pid, nvmlAccountingStats_t* stats) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetAccountingStats(device, pid, stats)


cdef nvmlReturn_t nvmlDeviceGetAccountingPids(nvmlDevice_t device, unsigned int* count, unsigned int* pids) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetAccountingPids(device, count, pids)


cdef nvmlReturn_t nvmlDeviceGetAccountingBufferSize(nvmlDevice_t device, unsigned int* bufferSize) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetAccountingBufferSize(device, bufferSize)


cdef nvmlReturn_t nvmlDeviceGetRetiredPages(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int* pageCount, unsigned long long* addresses) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetRetiredPages(device, cause, pageCount, addresses)


cdef nvmlReturn_t nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int* pageCount, unsigned long long* addresses, unsigned long long* timestamps) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetRetiredPages_v2(device, cause, pageCount, addresses, timestamps)


cdef nvmlReturn_t nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device, nvmlEnableState_t* isPending) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetRetiredPagesPendingStatus(device, isPending)


cdef nvmlReturn_t nvmlDeviceGetRemappedRows(nvmlDevice_t device, unsigned int* corrRows, unsigned int* uncRows, unsigned int* isPending, unsigned int* failureOccurred) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetRemappedRows(device, corrRows, uncRows, isPending, failureOccurred)


cdef nvmlReturn_t nvmlDeviceGetRowRemapperHistogram(nvmlDevice_t device, nvmlRowRemapperHistogramValues_t* values) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetRowRemapperHistogram(device, values)


cdef nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t device, nvmlDeviceArchitecture_t* arch) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetArchitecture(device, arch)


cdef nvmlReturn_t nvmlDeviceGetClkMonStatus(nvmlDevice_t device, nvmlClkMonStatus_t* status) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetClkMonStatus(device, status)


cdef nvmlReturn_t nvmlDeviceGetProcessUtilization(nvmlDevice_t device, nvmlProcessUtilizationSample_t* utilization, unsigned int* processSamplesCount, unsigned long long lastSeenTimeStamp) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetProcessUtilization(device, utilization, processSamplesCount, lastSeenTimeStamp)


cdef nvmlReturn_t nvmlDeviceGetProcessesUtilizationInfo(nvmlDevice_t device, nvmlProcessesUtilizationInfo_t* procesesUtilInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetProcessesUtilizationInfo(device, procesesUtilInfo)


cdef nvmlReturn_t nvmlDeviceGetPlatformInfo(nvmlDevice_t device, nvmlPlatformInfo_t* platformInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetPlatformInfo(device, platformInfo)


cdef nvmlReturn_t nvmlUnitSetLedState(nvmlUnit_t unit, nvmlLedColor_t color) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlUnitSetLedState(unit, color)


cdef nvmlReturn_t nvmlDeviceSetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t mode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetPersistenceMode(device, mode)


cdef nvmlReturn_t nvmlDeviceSetComputeMode(nvmlDevice_t device, nvmlComputeMode_t mode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetComputeMode(device, mode)


cdef nvmlReturn_t nvmlDeviceSetEccMode(nvmlDevice_t device, nvmlEnableState_t ecc) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetEccMode(device, ecc)


cdef nvmlReturn_t nvmlDeviceClearEccErrorCounts(nvmlDevice_t device, nvmlEccCounterType_t counterType) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceClearEccErrorCounts(device, counterType)


cdef nvmlReturn_t nvmlDeviceSetDriverModel(nvmlDevice_t device, nvmlDriverModel_t driverModel, unsigned int flags) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetDriverModel(device, driverModel, flags)


cdef nvmlReturn_t nvmlDeviceSetGpuLockedClocks(nvmlDevice_t device, unsigned int minGpuClockMHz, unsigned int maxGpuClockMHz) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetGpuLockedClocks(device, minGpuClockMHz, maxGpuClockMHz)


cdef nvmlReturn_t nvmlDeviceResetGpuLockedClocks(nvmlDevice_t device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceResetGpuLockedClocks(device)


cdef nvmlReturn_t nvmlDeviceSetMemoryLockedClocks(nvmlDevice_t device, unsigned int minMemClockMHz, unsigned int maxMemClockMHz) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetMemoryLockedClocks(device, minMemClockMHz, maxMemClockMHz)


cdef nvmlReturn_t nvmlDeviceResetMemoryLockedClocks(nvmlDevice_t device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceResetMemoryLockedClocks(device)


cdef nvmlReturn_t nvmlDeviceSetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetAutoBoostedClocksEnabled(device, enabled)


cdef nvmlReturn_t nvmlDeviceSetDefaultAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled, unsigned int flags) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetDefaultAutoBoostedClocksEnabled(device, enabled, flags)


cdef nvmlReturn_t nvmlDeviceSetDefaultFanSpeed_v2(nvmlDevice_t device, unsigned int fan) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetDefaultFanSpeed_v2(device, fan)


cdef nvmlReturn_t nvmlDeviceSetFanControlPolicy(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t policy) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetFanControlPolicy(device, fan, policy)


cdef nvmlReturn_t nvmlDeviceSetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, int* temp) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetTemperatureThreshold(device, thresholdType, temp)


cdef nvmlReturn_t nvmlDeviceSetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t mode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetGpuOperationMode(device, mode)


cdef nvmlReturn_t nvmlDeviceSetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t isRestricted) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetAPIRestriction(device, apiType, isRestricted)


cdef nvmlReturn_t nvmlDeviceSetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int speed) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetFanSpeed_v2(device, fan, speed)


cdef nvmlReturn_t nvmlDeviceSetAccountingMode(nvmlDevice_t device, nvmlEnableState_t mode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetAccountingMode(device, mode)


cdef nvmlReturn_t nvmlDeviceClearAccountingPids(nvmlDevice_t device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceClearAccountingPids(device)


cdef nvmlReturn_t nvmlDeviceSetPowerManagementLimit_v2(nvmlDevice_t device, nvmlPowerValue_v2_t* powerValue) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetPowerManagementLimit_v2(device, powerValue)


cdef nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t* isActive) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetNvLinkState(device, link, isActive)


cdef nvmlReturn_t nvmlDeviceGetNvLinkVersion(nvmlDevice_t device, unsigned int link, unsigned int* version) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetNvLinkVersion(device, link, version)


cdef nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int* capResult) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetNvLinkCapability(device, link, capability, capResult)


cdef nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t* pci) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetNvLinkRemotePciInfo_v2(device, link, pci)


cdef nvmlReturn_t nvmlDeviceGetNvLinkErrorCounter(nvmlDevice_t device, unsigned int link, nvmlNvLinkErrorCounter_t counter, unsigned long long* counterValue) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetNvLinkErrorCounter(device, link, counter, counterValue)


cdef nvmlReturn_t nvmlDeviceResetNvLinkErrorCounters(nvmlDevice_t device, unsigned int link) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceResetNvLinkErrorCounters(device, link)


cdef nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link, nvmlIntNvLinkDeviceType_t* pNvLinkDeviceType) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetNvLinkRemoteDeviceType(device, link, pNvLinkDeviceType)


cdef nvmlReturn_t nvmlDeviceSetNvLinkDeviceLowPowerThreshold(nvmlDevice_t device, nvmlNvLinkPowerThres_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetNvLinkDeviceLowPowerThreshold(device, info)


cdef nvmlReturn_t nvmlSystemSetNvlinkBwMode(unsigned int nvlinkBwMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSystemSetNvlinkBwMode(nvlinkBwMode)


cdef nvmlReturn_t nvmlSystemGetNvlinkBwMode(unsigned int* nvlinkBwMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSystemGetNvlinkBwMode(nvlinkBwMode)


cdef nvmlReturn_t nvmlDeviceGetNvlinkSupportedBwModes(nvmlDevice_t device, nvmlNvlinkSupportedBwModes_t* supportedBwMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetNvlinkSupportedBwModes(device, supportedBwMode)


cdef nvmlReturn_t nvmlDeviceGetNvlinkBwMode(nvmlDevice_t device, nvmlNvlinkGetBwMode_t* getBwMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetNvlinkBwMode(device, getBwMode)


cdef nvmlReturn_t nvmlDeviceSetNvlinkBwMode(nvmlDevice_t device, nvmlNvlinkSetBwMode_t* setBwMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetNvlinkBwMode(device, setBwMode)


cdef nvmlReturn_t nvmlEventSetCreate(nvmlEventSet_t* set) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlEventSetCreate(set)


cdef nvmlReturn_t nvmlDeviceRegisterEvents(nvmlDevice_t device, unsigned long long eventTypes, nvmlEventSet_t set) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceRegisterEvents(device, eventTypes, set)


cdef nvmlReturn_t nvmlDeviceGetSupportedEventTypes(nvmlDevice_t device, unsigned long long* eventTypes) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetSupportedEventTypes(device, eventTypes)


cdef nvmlReturn_t nvmlEventSetWait_v2(nvmlEventSet_t set, nvmlEventData_t* data, unsigned int timeoutms) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlEventSetWait_v2(set, data, timeoutms)


cdef nvmlReturn_t nvmlEventSetFree(nvmlEventSet_t set) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlEventSetFree(set)


cdef nvmlReturn_t nvmlSystemEventSetCreate(nvmlSystemEventSetCreateRequest_t* request) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSystemEventSetCreate(request)


cdef nvmlReturn_t nvmlSystemEventSetFree(nvmlSystemEventSetFreeRequest_t* request) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSystemEventSetFree(request)


cdef nvmlReturn_t nvmlSystemRegisterEvents(nvmlSystemRegisterEventRequest_t* request) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSystemRegisterEvents(request)


cdef nvmlReturn_t nvmlSystemEventSetWait(nvmlSystemEventSetWaitRequest_t* request) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSystemEventSetWait(request)


cdef nvmlReturn_t nvmlDeviceModifyDrainState(nvmlPciInfo_t* pciInfo, nvmlEnableState_t newState) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceModifyDrainState(pciInfo, newState)


cdef nvmlReturn_t nvmlDeviceQueryDrainState(nvmlPciInfo_t* pciInfo, nvmlEnableState_t* currentState) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceQueryDrainState(pciInfo, currentState)


cdef nvmlReturn_t nvmlDeviceRemoveGpu_v2(nvmlPciInfo_t* pciInfo, nvmlDetachGpuState_t gpuState, nvmlPcieLinkState_t linkState) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceRemoveGpu_v2(pciInfo, gpuState, linkState)


cdef nvmlReturn_t nvmlDeviceDiscoverGpus(nvmlPciInfo_t* pciInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceDiscoverGpus(pciInfo)


cdef nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetFieldValues(device, valuesCount, values)


cdef nvmlReturn_t nvmlDeviceClearFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceClearFieldValues(device, valuesCount, values)


cdef nvmlReturn_t nvmlDeviceGetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t* pVirtualMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetVirtualizationMode(device, pVirtualMode)


cdef nvmlReturn_t nvmlDeviceGetHostVgpuMode(nvmlDevice_t device, nvmlHostVgpuMode_t* pHostVgpuMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetHostVgpuMode(device, pHostVgpuMode)


cdef nvmlReturn_t nvmlDeviceSetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t virtualMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetVirtualizationMode(device, virtualMode)


cdef nvmlReturn_t nvmlDeviceGetVgpuHeterogeneousMode(nvmlDevice_t device, nvmlVgpuHeterogeneousMode_t* pHeterogeneousMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetVgpuHeterogeneousMode(device, pHeterogeneousMode)


cdef nvmlReturn_t nvmlDeviceSetVgpuHeterogeneousMode(nvmlDevice_t device, const nvmlVgpuHeterogeneousMode_t* pHeterogeneousMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetVgpuHeterogeneousMode(device, pHeterogeneousMode)


cdef nvmlReturn_t nvmlVgpuInstanceGetPlacementId(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuPlacementId_t* pPlacement) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetPlacementId(vgpuInstance, pPlacement)


cdef nvmlReturn_t nvmlDeviceGetVgpuTypeSupportedPlacements(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuPlacementList_t* pPlacementList) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetVgpuTypeSupportedPlacements(device, vgpuTypeId, pPlacementList)


cdef nvmlReturn_t nvmlDeviceGetVgpuTypeCreatablePlacements(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuPlacementList_t* pPlacementList) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetVgpuTypeCreatablePlacements(device, vgpuTypeId, pPlacementList)


cdef nvmlReturn_t nvmlVgpuTypeGetGspHeapSize(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* gspHeapSize) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuTypeGetGspHeapSize(vgpuTypeId, gspHeapSize)


cdef nvmlReturn_t nvmlVgpuTypeGetFbReservation(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* fbReservation) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuTypeGetFbReservation(vgpuTypeId, fbReservation)


cdef nvmlReturn_t nvmlVgpuInstanceGetRuntimeStateSize(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuRuntimeState_t* pState) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetRuntimeStateSize(vgpuInstance, pState)


cdef nvmlReturn_t nvmlDeviceSetVgpuCapabilities(nvmlDevice_t device, nvmlDeviceVgpuCapability_t capability, nvmlEnableState_t state) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetVgpuCapabilities(device, capability, state)


cdef nvmlReturn_t nvmlDeviceGetGridLicensableFeatures_v4(nvmlDevice_t device, nvmlGridLicensableFeatures_t* pGridLicensableFeatures) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetGridLicensableFeatures_v4(device, pGridLicensableFeatures)


cdef nvmlReturn_t nvmlGetVgpuDriverCapabilities(nvmlVgpuDriverCapability_t capability, unsigned int* capResult) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGetVgpuDriverCapabilities(capability, capResult)


cdef nvmlReturn_t nvmlDeviceGetVgpuCapabilities(nvmlDevice_t device, nvmlDeviceVgpuCapability_t capability, unsigned int* capResult) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetVgpuCapabilities(device, capability, capResult)


cdef nvmlReturn_t nvmlDeviceGetSupportedVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuTypeId_t* vgpuTypeIds) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetSupportedVgpus(device, vgpuCount, vgpuTypeIds)


cdef nvmlReturn_t nvmlDeviceGetCreatableVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuTypeId_t* vgpuTypeIds) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetCreatableVgpus(device, vgpuCount, vgpuTypeIds)


cdef nvmlReturn_t nvmlVgpuTypeGetClass(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeClass, unsigned int* size) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuTypeGetClass(vgpuTypeId, vgpuTypeClass, size)


cdef nvmlReturn_t nvmlVgpuTypeGetName(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeName, unsigned int* size) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuTypeGetName(vgpuTypeId, vgpuTypeName, size)


cdef nvmlReturn_t nvmlVgpuTypeGetGpuInstanceProfileId(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* gpuInstanceProfileId) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuTypeGetGpuInstanceProfileId(vgpuTypeId, gpuInstanceProfileId)


cdef nvmlReturn_t nvmlVgpuTypeGetDeviceID(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* deviceID, unsigned long long* subsystemID) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuTypeGetDeviceID(vgpuTypeId, deviceID, subsystemID)


cdef nvmlReturn_t nvmlVgpuTypeGetFramebufferSize(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* fbSize) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuTypeGetFramebufferSize(vgpuTypeId, fbSize)


cdef nvmlReturn_t nvmlVgpuTypeGetNumDisplayHeads(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* numDisplayHeads) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuTypeGetNumDisplayHeads(vgpuTypeId, numDisplayHeads)


cdef nvmlReturn_t nvmlVgpuTypeGetResolution(nvmlVgpuTypeId_t vgpuTypeId, unsigned int displayIndex, unsigned int* xdim, unsigned int* ydim) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuTypeGetResolution(vgpuTypeId, displayIndex, xdim, ydim)


cdef nvmlReturn_t nvmlVgpuTypeGetLicense(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeLicenseString, unsigned int size) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuTypeGetLicense(vgpuTypeId, vgpuTypeLicenseString, size)


cdef nvmlReturn_t nvmlVgpuTypeGetFrameRateLimit(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* frameRateLimit) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuTypeGetFrameRateLimit(vgpuTypeId, frameRateLimit)


cdef nvmlReturn_t nvmlVgpuTypeGetMaxInstances(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, unsigned int* vgpuInstanceCount) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuTypeGetMaxInstances(device, vgpuTypeId, vgpuInstanceCount)


cdef nvmlReturn_t nvmlVgpuTypeGetMaxInstancesPerVm(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* vgpuInstanceCountPerVm) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuTypeGetMaxInstancesPerVm(vgpuTypeId, vgpuInstanceCountPerVm)


cdef nvmlReturn_t nvmlVgpuTypeGetBAR1Info(nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuTypeBar1Info_t* bar1Info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuTypeGetBAR1Info(vgpuTypeId, bar1Info)


cdef nvmlReturn_t nvmlDeviceGetActiveVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuInstance_t* vgpuInstances) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetActiveVgpus(device, vgpuCount, vgpuInstances)


cdef nvmlReturn_t nvmlVgpuInstanceGetVmID(nvmlVgpuInstance_t vgpuInstance, char* vmId, unsigned int size, nvmlVgpuVmIdType_t* vmIdType) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetVmID(vgpuInstance, vmId, size, vmIdType)


cdef nvmlReturn_t nvmlVgpuInstanceGetUUID(nvmlVgpuInstance_t vgpuInstance, char* uuid, unsigned int size) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetUUID(vgpuInstance, uuid, size)


cdef nvmlReturn_t nvmlVgpuInstanceGetVmDriverVersion(nvmlVgpuInstance_t vgpuInstance, char* version, unsigned int length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetVmDriverVersion(vgpuInstance, version, length)


cdef nvmlReturn_t nvmlVgpuInstanceGetFbUsage(nvmlVgpuInstance_t vgpuInstance, unsigned long long* fbUsage) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetFbUsage(vgpuInstance, fbUsage)


cdef nvmlReturn_t nvmlVgpuInstanceGetLicenseStatus(nvmlVgpuInstance_t vgpuInstance, unsigned int* licensed) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetLicenseStatus(vgpuInstance, licensed)


cdef nvmlReturn_t nvmlVgpuInstanceGetType(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuTypeId_t* vgpuTypeId) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetType(vgpuInstance, vgpuTypeId)


cdef nvmlReturn_t nvmlVgpuInstanceGetFrameRateLimit(nvmlVgpuInstance_t vgpuInstance, unsigned int* frameRateLimit) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetFrameRateLimit(vgpuInstance, frameRateLimit)


cdef nvmlReturn_t nvmlVgpuInstanceGetEccMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t* eccMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetEccMode(vgpuInstance, eccMode)


cdef nvmlReturn_t nvmlVgpuInstanceGetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int* encoderCapacity) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetEncoderCapacity(vgpuInstance, encoderCapacity)


cdef nvmlReturn_t nvmlVgpuInstanceSetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int encoderCapacity) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceSetEncoderCapacity(vgpuInstance, encoderCapacity)


cdef nvmlReturn_t nvmlVgpuInstanceGetEncoderStats(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, unsigned int* averageFps, unsigned int* averageLatency) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetEncoderStats(vgpuInstance, sessionCount, averageFps, averageLatency)


cdef nvmlReturn_t nvmlVgpuInstanceGetEncoderSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, nvmlEncoderSessionInfo_t* sessionInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetEncoderSessions(vgpuInstance, sessionCount, sessionInfo)


cdef nvmlReturn_t nvmlVgpuInstanceGetFBCStats(nvmlVgpuInstance_t vgpuInstance, nvmlFBCStats_t* fbcStats) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetFBCStats(vgpuInstance, fbcStats)


cdef nvmlReturn_t nvmlVgpuInstanceGetFBCSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, nvmlFBCSessionInfo_t* sessionInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetFBCSessions(vgpuInstance, sessionCount, sessionInfo)


cdef nvmlReturn_t nvmlVgpuInstanceGetGpuInstanceId(nvmlVgpuInstance_t vgpuInstance, unsigned int* gpuInstanceId) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetGpuInstanceId(vgpuInstance, gpuInstanceId)


cdef nvmlReturn_t nvmlVgpuInstanceGetGpuPciId(nvmlVgpuInstance_t vgpuInstance, char* vgpuPciId, unsigned int* length) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetGpuPciId(vgpuInstance, vgpuPciId, length)


cdef nvmlReturn_t nvmlVgpuTypeGetCapabilities(nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuCapability_t capability, unsigned int* capResult) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuTypeGetCapabilities(vgpuTypeId, capability, capResult)


cdef nvmlReturn_t nvmlVgpuInstanceGetMdevUUID(nvmlVgpuInstance_t vgpuInstance, char* mdevUuid, unsigned int size) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetMdevUUID(vgpuInstance, mdevUuid, size)


cdef nvmlReturn_t nvmlGpuInstanceGetCreatableVgpus(nvmlGpuInstance_t gpuInstance, nvmlVgpuTypeIdInfo_t* pVgpus) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGpuInstanceGetCreatableVgpus(gpuInstance, pVgpus)


cdef nvmlReturn_t nvmlVgpuTypeGetMaxInstancesPerGpuInstance(nvmlVgpuTypeMaxInstance_t* pMaxInstance) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuTypeGetMaxInstancesPerGpuInstance(pMaxInstance)


cdef nvmlReturn_t nvmlGpuInstanceGetActiveVgpus(nvmlGpuInstance_t gpuInstance, nvmlActiveVgpuInstanceInfo_t* pVgpuInstanceInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGpuInstanceGetActiveVgpus(gpuInstance, pVgpuInstanceInfo)


cdef nvmlReturn_t nvmlGpuInstanceSetVgpuSchedulerState(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerState_t* pScheduler) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGpuInstanceSetVgpuSchedulerState(gpuInstance, pScheduler)


cdef nvmlReturn_t nvmlGpuInstanceGetVgpuSchedulerState(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerStateInfo_t* pSchedulerStateInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGpuInstanceGetVgpuSchedulerState(gpuInstance, pSchedulerStateInfo)


cdef nvmlReturn_t nvmlGpuInstanceGetVgpuSchedulerLog(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerLogInfo_t* pSchedulerLogInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGpuInstanceGetVgpuSchedulerLog(gpuInstance, pSchedulerLogInfo)


cdef nvmlReturn_t nvmlGpuInstanceGetVgpuTypeCreatablePlacements(nvmlGpuInstance_t gpuInstance, nvmlVgpuCreatablePlacementInfo_t* pCreatablePlacementInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGpuInstanceGetVgpuTypeCreatablePlacements(gpuInstance, pCreatablePlacementInfo)


cdef nvmlReturn_t nvmlGpuInstanceGetVgpuHeterogeneousMode(nvmlGpuInstance_t gpuInstance, nvmlVgpuHeterogeneousMode_t* pHeterogeneousMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGpuInstanceGetVgpuHeterogeneousMode(gpuInstance, pHeterogeneousMode)


cdef nvmlReturn_t nvmlGpuInstanceSetVgpuHeterogeneousMode(nvmlGpuInstance_t gpuInstance, const nvmlVgpuHeterogeneousMode_t* pHeterogeneousMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGpuInstanceSetVgpuHeterogeneousMode(gpuInstance, pHeterogeneousMode)


cdef nvmlReturn_t nvmlVgpuInstanceGetMetadata(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuMetadata_t* vgpuMetadata, unsigned int* bufferSize) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetMetadata(vgpuInstance, vgpuMetadata, bufferSize)


cdef nvmlReturn_t nvmlDeviceGetVgpuMetadata(nvmlDevice_t device, nvmlVgpuPgpuMetadata_t* pgpuMetadata, unsigned int* bufferSize) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetVgpuMetadata(device, pgpuMetadata, bufferSize)


cdef nvmlReturn_t nvmlGetVgpuCompatibility(nvmlVgpuMetadata_t* vgpuMetadata, nvmlVgpuPgpuMetadata_t* pgpuMetadata, nvmlVgpuPgpuCompatibility_t* compatibilityInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGetVgpuCompatibility(vgpuMetadata, pgpuMetadata, compatibilityInfo)


cdef nvmlReturn_t nvmlDeviceGetPgpuMetadataString(nvmlDevice_t device, char* pgpuMetadata, unsigned int* bufferSize) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetPgpuMetadataString(device, pgpuMetadata, bufferSize)


cdef nvmlReturn_t nvmlDeviceGetVgpuSchedulerLog(nvmlDevice_t device, nvmlVgpuSchedulerLog_t* pSchedulerLog) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetVgpuSchedulerLog(device, pSchedulerLog)


cdef nvmlReturn_t nvmlDeviceGetVgpuSchedulerState(nvmlDevice_t device, nvmlVgpuSchedulerGetState_t* pSchedulerState) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetVgpuSchedulerState(device, pSchedulerState)


cdef nvmlReturn_t nvmlDeviceGetVgpuSchedulerCapabilities(nvmlDevice_t device, nvmlVgpuSchedulerCapabilities_t* pCapabilities) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetVgpuSchedulerCapabilities(device, pCapabilities)


cdef nvmlReturn_t nvmlDeviceSetVgpuSchedulerState(nvmlDevice_t device, nvmlVgpuSchedulerSetState_t* pSchedulerState) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetVgpuSchedulerState(device, pSchedulerState)


cdef nvmlReturn_t nvmlGetVgpuVersion(nvmlVgpuVersion_t* supported, nvmlVgpuVersion_t* current) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGetVgpuVersion(supported, current)


cdef nvmlReturn_t nvmlSetVgpuVersion(nvmlVgpuVersion_t* vgpuVersion) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlSetVgpuVersion(vgpuVersion)


cdef nvmlReturn_t nvmlDeviceGetVgpuUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, nvmlValueType_t* sampleValType, unsigned int* vgpuInstanceSamplesCount, nvmlVgpuInstanceUtilizationSample_t* utilizationSamples) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetVgpuUtilization(device, lastSeenTimeStamp, sampleValType, vgpuInstanceSamplesCount, utilizationSamples)


cdef nvmlReturn_t nvmlDeviceGetVgpuInstancesUtilizationInfo(nvmlDevice_t device, nvmlVgpuInstancesUtilizationInfo_t* vgpuUtilInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetVgpuInstancesUtilizationInfo(device, vgpuUtilInfo)


cdef nvmlReturn_t nvmlDeviceGetVgpuProcessUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, unsigned int* vgpuProcessSamplesCount, nvmlVgpuProcessUtilizationSample_t* utilizationSamples) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetVgpuProcessUtilization(device, lastSeenTimeStamp, vgpuProcessSamplesCount, utilizationSamples)


cdef nvmlReturn_t nvmlDeviceGetVgpuProcessesUtilizationInfo(nvmlDevice_t device, nvmlVgpuProcessesUtilizationInfo_t* vgpuProcUtilInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetVgpuProcessesUtilizationInfo(device, vgpuProcUtilInfo)


cdef nvmlReturn_t nvmlVgpuInstanceGetAccountingMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t* mode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetAccountingMode(vgpuInstance, mode)


cdef nvmlReturn_t nvmlVgpuInstanceGetAccountingPids(nvmlVgpuInstance_t vgpuInstance, unsigned int* count, unsigned int* pids) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetAccountingPids(vgpuInstance, count, pids)


cdef nvmlReturn_t nvmlVgpuInstanceGetAccountingStats(nvmlVgpuInstance_t vgpuInstance, unsigned int pid, nvmlAccountingStats_t* stats) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetAccountingStats(vgpuInstance, pid, stats)


cdef nvmlReturn_t nvmlVgpuInstanceClearAccountingPids(nvmlVgpuInstance_t vgpuInstance) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceClearAccountingPids(vgpuInstance)


cdef nvmlReturn_t nvmlVgpuInstanceGetLicenseInfo_v2(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuLicenseInfo_t* licenseInfo) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlVgpuInstanceGetLicenseInfo_v2(vgpuInstance, licenseInfo)


cdef nvmlReturn_t nvmlGetExcludedDeviceCount(unsigned int* deviceCount) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGetExcludedDeviceCount(deviceCount)


cdef nvmlReturn_t nvmlGetExcludedDeviceInfoByIndex(unsigned int index, nvmlExcludedDeviceInfo_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGetExcludedDeviceInfoByIndex(index, info)


cdef nvmlReturn_t nvmlDeviceSetMigMode(nvmlDevice_t device, unsigned int mode, nvmlReturn_t* activationStatus) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetMigMode(device, mode, activationStatus)


cdef nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int* currentMode, unsigned int* pendingMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetMigMode(device, currentMode, pendingMode)


cdef nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfoV(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_v2_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetGpuInstanceProfileInfoV(device, profile, info)


cdef nvmlReturn_t nvmlDeviceGetGpuInstancePossiblePlacements_v2(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstancePlacement_t* placements, unsigned int* count) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetGpuInstancePossiblePlacements_v2(device, profileId, placements, count)


cdef nvmlReturn_t nvmlDeviceGetGpuInstanceRemainingCapacity(nvmlDevice_t device, unsigned int profileId, unsigned int* count) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetGpuInstanceRemainingCapacity(device, profileId, count)


cdef nvmlReturn_t nvmlDeviceCreateGpuInstance(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t* gpuInstance) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceCreateGpuInstance(device, profileId, gpuInstance)


cdef nvmlReturn_t nvmlDeviceCreateGpuInstanceWithPlacement(nvmlDevice_t device, unsigned int profileId, const nvmlGpuInstancePlacement_t* placement, nvmlGpuInstance_t* gpuInstance) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceCreateGpuInstanceWithPlacement(device, profileId, placement, gpuInstance)


cdef nvmlReturn_t nvmlGpuInstanceDestroy(nvmlGpuInstance_t gpuInstance) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGpuInstanceDestroy(gpuInstance)


cdef nvmlReturn_t nvmlDeviceGetGpuInstances(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t* gpuInstances, unsigned int* count) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetGpuInstances(device, profileId, gpuInstances, count)


cdef nvmlReturn_t nvmlDeviceGetGpuInstanceById(nvmlDevice_t device, unsigned int id, nvmlGpuInstance_t* gpuInstance) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetGpuInstanceById(device, id, gpuInstance)


cdef nvmlReturn_t nvmlGpuInstanceGetInfo(nvmlGpuInstance_t gpuInstance, nvmlGpuInstanceInfo_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGpuInstanceGetInfo(gpuInstance, info)


cdef nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfoV(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_v2_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGpuInstanceGetComputeInstanceProfileInfoV(gpuInstance, profile, engProfile, info)


cdef nvmlReturn_t nvmlGpuInstanceGetComputeInstanceRemainingCapacity(nvmlGpuInstance_t gpuInstance, unsigned int profileId, unsigned int* count) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGpuInstanceGetComputeInstanceRemainingCapacity(gpuInstance, profileId, count)


cdef nvmlReturn_t nvmlGpuInstanceGetComputeInstancePossiblePlacements(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstancePlacement_t* placements, unsigned int* count) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGpuInstanceGetComputeInstancePossiblePlacements(gpuInstance, profileId, placements, count)


cdef nvmlReturn_t nvmlGpuInstanceCreateComputeInstance(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t* computeInstance) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGpuInstanceCreateComputeInstance(gpuInstance, profileId, computeInstance)


cdef nvmlReturn_t nvmlGpuInstanceCreateComputeInstanceWithPlacement(nvmlGpuInstance_t gpuInstance, unsigned int profileId, const nvmlComputeInstancePlacement_t* placement, nvmlComputeInstance_t* computeInstance) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGpuInstanceCreateComputeInstanceWithPlacement(gpuInstance, profileId, placement, computeInstance)


cdef nvmlReturn_t nvmlComputeInstanceDestroy(nvmlComputeInstance_t computeInstance) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlComputeInstanceDestroy(computeInstance)


cdef nvmlReturn_t nvmlGpuInstanceGetComputeInstances(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t* computeInstances, unsigned int* count) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGpuInstanceGetComputeInstances(gpuInstance, profileId, computeInstances, count)


cdef nvmlReturn_t nvmlGpuInstanceGetComputeInstanceById(nvmlGpuInstance_t gpuInstance, unsigned int id, nvmlComputeInstance_t* computeInstance) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlGpuInstanceGetComputeInstanceById(gpuInstance, id, computeInstance)


cdef nvmlReturn_t nvmlComputeInstanceGetInfo_v2(nvmlComputeInstance_t computeInstance, nvmlComputeInstanceInfo_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlComputeInstanceGetInfo_v2(computeInstance, info)


cdef nvmlReturn_t nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device, unsigned int* isMigDevice) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceIsMigDeviceHandle(device, isMigDevice)


cdef nvmlReturn_t nvmlDeviceGetGpuInstanceId(nvmlDevice_t device, unsigned int* id) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetGpuInstanceId(device, id)


cdef nvmlReturn_t nvmlDeviceGetComputeInstanceId(nvmlDevice_t device, unsigned int* id) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetComputeInstanceId(device, id)


cdef nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device, unsigned int* count) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetMaxMigDeviceCount(device, count)


cdef nvmlReturn_t nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t device, unsigned int index, nvmlDevice_t* migDevice) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetMigDeviceHandleByIndex(device, index, migDevice)


cdef nvmlReturn_t nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t migDevice, nvmlDevice_t* device) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetDeviceHandleFromMigDeviceHandle(migDevice, device)


cdef nvmlReturn_t nvmlDeviceGetCapabilities(nvmlDevice_t device, nvmlDeviceCapabilities_t* caps) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetCapabilities(device, caps)


cdef nvmlReturn_t nvmlDevicePowerSmoothingActivatePresetProfile(nvmlDevice_t device, nvmlPowerSmoothingProfile_t* profile) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDevicePowerSmoothingActivatePresetProfile(device, profile)


cdef nvmlReturn_t nvmlDevicePowerSmoothingUpdatePresetProfileParam(nvmlDevice_t device, nvmlPowerSmoothingProfile_t* profile) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDevicePowerSmoothingUpdatePresetProfileParam(device, profile)


cdef nvmlReturn_t nvmlDevicePowerSmoothingSetState(nvmlDevice_t device, nvmlPowerSmoothingState_t* state) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDevicePowerSmoothingSetState(device, state)


cdef nvmlReturn_t nvmlDeviceGetAddressingMode(nvmlDevice_t device, nvmlDeviceAddressingMode_t* mode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetAddressingMode(device, mode)


cdef nvmlReturn_t nvmlDeviceGetRepairStatus(nvmlDevice_t device, nvmlRepairStatus_t* repairStatus) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetRepairStatus(device, repairStatus)


cdef nvmlReturn_t nvmlDeviceGetPowerMizerMode_v1(nvmlDevice_t device, nvmlDevicePowerMizerModes_v1_t* powerMizerMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetPowerMizerMode_v1(device, powerMizerMode)


cdef nvmlReturn_t nvmlDeviceSetPowerMizerMode_v1(nvmlDevice_t device, nvmlDevicePowerMizerModes_v1_t* powerMizerMode) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetPowerMizerMode_v1(device, powerMizerMode)


cdef nvmlReturn_t nvmlDeviceGetPdi(nvmlDevice_t device, nvmlPdi_t* pdi) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetPdi(device, pdi)


cdef nvmlReturn_t nvmlDeviceSetHostname_v1(nvmlDevice_t device, nvmlHostname_v1_t* hostname) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetHostname_v1(device, hostname)


cdef nvmlReturn_t nvmlDeviceGetHostname_v1(nvmlDevice_t device, nvmlHostname_v1_t* hostname) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetHostname_v1(device, hostname)


cdef nvmlReturn_t nvmlDeviceGetNvLinkInfo(nvmlDevice_t device, nvmlNvLinkInfo_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetNvLinkInfo(device, info)


cdef nvmlReturn_t nvmlDeviceReadWritePRM_v1(nvmlDevice_t device, nvmlPRMTLV_v1_t* buffer) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceReadWritePRM_v1(device, buffer)


cdef nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfoByIdV(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstanceProfileInfo_v2_t* info) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetGpuInstanceProfileInfoByIdV(device, profileId, info)


cdef nvmlReturn_t nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts(nvmlDevice_t device, nvmlEccSramUniqueUncorrectedErrorCounts_t* errorCounts) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts(device, errorCounts)


cdef nvmlReturn_t nvmlDeviceGetUnrepairableMemoryFlag_v1(nvmlDevice_t device, nvmlUnrepairableMemoryStatus_v1_t* unrepairableMemoryStatus) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceGetUnrepairableMemoryFlag_v1(device, unrepairableMemoryStatus)


cdef nvmlReturn_t nvmlDeviceReadPRMCounters_v1(nvmlDevice_t device, nvmlPRMCounterList_v1_t* counterList) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceReadPRMCounters_v1(device, counterList)


cdef nvmlReturn_t nvmlDeviceSetRusdSettings_v1(nvmlDevice_t device, nvmlRusdSettings_v1_t* settings) except?_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil:
    return _nvml._nvmlDeviceSetRusdSettings_v1(device, settings)
