# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.9.1 to 13.3.0, generator version 0.3.1.dev1779+ga8cc71818.d20260626. Do not modify it directly.

from ..cynvml cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef nvmlReturn_t _nvmlInit_v2() except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlInitWithFlags(unsigned int flags) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlShutdown() except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef const char* _nvmlErrorString(nvmlReturn_t result) except?NULL nogil
cdef nvmlReturn_t _nvmlSystemGetDriverVersion(char* version, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlSystemGetNVMLVersion(char* version, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlSystemGetCudaDriverVersion(int* cudaDriverVersion) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlSystemGetCudaDriverVersion_v2(int* cudaDriverVersion) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlSystemGetProcessName(unsigned int pid, char* name, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlSystemGetHicVersion(unsigned int* hwbcCount, nvmlHwbcEntry_t* hwbcEntries) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber, unsigned int* count, nvmlDevice_t* deviceArray) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlSystemGetDriverBranch(nvmlSystemDriverBranchInfo_t* branchInfo, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlUnitGetCount(unsigned int* unitCount) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlUnitGetHandleByIndex(unsigned int index, nvmlUnit_t* unit) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlUnitGetUnitInfo(nvmlUnit_t unit, nvmlUnitInfo_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlUnitGetLedState(nvmlUnit_t unit, nvmlLedState_t* state) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlUnitGetPsuInfo(nvmlUnit_t unit, nvmlPSUInfo_t* psu) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlUnitGetTemperature(nvmlUnit_t unit, unsigned int type, unsigned int* temp) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlUnitGetFanSpeedInfo(nvmlUnit_t unit, nvmlUnitFanSpeeds_t* fanSpeeds) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlUnitGetDevices(nvmlUnit_t unit, unsigned int* deviceCount, nvmlDevice_t* devices) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetCount_v2(unsigned int* deviceCount) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetAttributes_v2(nvmlDevice_t device, nvmlDeviceAttributes_t* attributes) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t* device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetHandleBySerial(const char* serial, nvmlDevice_t* device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetHandleByUUID(const char* uuid, nvmlDevice_t* device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetHandleByUUIDV(const nvmlUUID_t* uuid, nvmlDevice_t* device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetHandleByPciBusId_v2(const char* pciBusId, nvmlDevice_t* device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetBrand(nvmlDevice_t device, nvmlBrandType_t* type) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetSerial(nvmlDevice_t device, char* serial, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetModuleId(nvmlDevice_t device, unsigned int* moduleId) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetC2cModeInfoV(nvmlDevice_t device, nvmlC2cModeInfo_v1_t* c2cModeInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetMemoryAffinity(nvmlDevice_t device, unsigned int nodeSetSize, unsigned long* nodeSet, nvmlAffinityScope_t scope) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetCpuAffinityWithinScope(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long* cpuSet, nvmlAffinityScope_t scope) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetCpuAffinity(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long* cpuSet) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetCpuAffinity(nvmlDevice_t device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceClearCpuAffinity(nvmlDevice_t device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetNumaNodeId(nvmlDevice_t device, unsigned int* node) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuTopologyLevel_t* pathInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetTopologyNearestGpus(nvmlDevice_t device, nvmlGpuTopologyLevel_t level, unsigned int* count, nvmlDevice_t* deviceArray) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetP2PStatus(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex, nvmlGpuP2PStatus_t* p2pStatus) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetUUID(nvmlDevice_t device, char* uuid, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int* minorNumber) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetBoardPartNumber(nvmlDevice_t device, char* partNumber, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetInforomVersion(nvmlDevice_t device, nvmlInforomObject_t object, char* version, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetInforomImageVersion(nvmlDevice_t device, char* version, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetInforomConfigurationChecksum(nvmlDevice_t device, unsigned int* checksum) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceValidateInforom(nvmlDevice_t device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetLastBBXFlushTime(nvmlDevice_t device, unsigned long long* timestamp, unsigned long* durationUs) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetDisplayMode(nvmlDevice_t device, nvmlEnableState_t* display) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetDisplayActive(nvmlDevice_t device, nvmlEnableState_t* isActive) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t* mode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetPciInfoExt(nvmlDevice_t device, nvmlPciInfoExt_t* pci) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t* pci) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int* maxLinkGen) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetGpuMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int* maxLinkGenDevice) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device, unsigned int* maxLinkWidth) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int* currLinkGen) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int* currLinkWidth) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetPcieThroughput(nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int* value) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device, unsigned int* value) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetGpcClkVfOffset(nvmlDevice_t device, int* offset) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int* clockMHz) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetMaxCustomerBoostClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int* clockMHz) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device, unsigned int* count, unsigned int* clocksMHz) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t device, unsigned int memoryClockMHz, unsigned int* count, unsigned int* clocksMHz) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t* isEnabled, nvmlEnableState_t* defaultIsEnabled) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int* speed) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int* speed) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetFanSpeedRPM(nvmlDevice_t device, nvmlFanSpeedInfo_t* fanSpeed) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetTargetFanSpeed(nvmlDevice_t device, unsigned int fan, unsigned int* targetSpeed) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetMinMaxFanSpeed(nvmlDevice_t device, unsigned int* minSpeed, unsigned int* maxSpeed) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetFanControlPolicy_v2(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t* policy) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetNumFans(nvmlDevice_t device, unsigned int* numFans) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetCoolerInfo(nvmlDevice_t device, nvmlCoolerInfo_t* coolerInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetTemperatureV(nvmlDevice_t device, nvmlTemperature_t* temperature) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, unsigned int* temp) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetMarginTemperature(nvmlDevice_t device, nvmlMarginTemperature_t* marginTempInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetThermalSettings(nvmlDevice_t device, unsigned int sensorIndex, nvmlGpuThermalSettings_t* pThermalSettings) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t* pState) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetCurrentClocksEventReasons(nvmlDevice_t device, unsigned long long* clocksEventReasons) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetSupportedClocksEventReasons(nvmlDevice_t device, unsigned long long* supportedClocksEventReasons) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetPowerState(nvmlDevice_t device, nvmlPstates_t* pState) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetDynamicPstatesInfo(nvmlDevice_t device, nvmlGpuDynamicPstatesInfo_t* pDynamicPstatesInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetMemClkVfOffset(nvmlDevice_t device, int* offset) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetMinMaxClockOfPState(nvmlDevice_t device, nvmlClockType_t type, nvmlPstates_t pstate, unsigned int* minClockMHz, unsigned int* maxClockMHz) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetSupportedPerformanceStates(nvmlDevice_t device, nvmlPstates_t* pstates, unsigned int size) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetGpcClkMinMaxVfOffset(nvmlDevice_t device, int* minOffset, int* maxOffset) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetMemClkMinMaxVfOffset(nvmlDevice_t device, int* minOffset, int* maxOffset) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetClockOffsets(nvmlDevice_t device, nvmlClockOffset_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetClockOffsets(nvmlDevice_t device, nvmlClockOffset_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetPerformanceModes(nvmlDevice_t device, nvmlDevicePerfModes_t* perfModes) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetCurrentClockFreqs(nvmlDevice_t device, nvmlDeviceCurrentClockFreqs_t* currentClockFreqs) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int* limit) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device, unsigned int* minLimit, unsigned int* maxLimit) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device, unsigned int* defaultLimit) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int* power) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t device, unsigned long long* energy) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int* limit) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t* current, nvmlGpuOperationMode_t* pending) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t* memory) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t* mode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major, int* minor) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetDramEncryptionMode(nvmlDevice_t device, nvmlDramEncryptionInfo_t* current, nvmlDramEncryptionInfo_t* pending) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetDramEncryptionMode(nvmlDevice_t device, const nvmlDramEncryptionInfo_t* dramEncryption) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetEccMode(nvmlDevice_t device, nvmlEnableState_t* current, nvmlEnableState_t* pending) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetDefaultEccMode(nvmlDevice_t device, nvmlEnableState_t* defaultMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int* boardId) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device, unsigned int* multiGpuBool) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetTotalEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long* eccCounts) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlMemoryLocation_t locationType, unsigned long long* count) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t* utilization) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetEncoderUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetEncoderCapacity(nvmlDevice_t device, nvmlEncoderType_t encoderQueryType, unsigned int* encoderCapacity) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetEncoderStats(nvmlDevice_t device, unsigned int* sessionCount, unsigned int* averageFps, unsigned int* averageLatency) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetEncoderSessions(nvmlDevice_t device, unsigned int* sessionCount, nvmlEncoderSessionInfo_t* sessionInfos) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetDecoderUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetJpgUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetOfaUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetFBCStats(nvmlDevice_t device, nvmlFBCStats_t* fbcStats) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetFBCSessions(nvmlDevice_t device, unsigned int* sessionCount, nvmlFBCSessionInfo_t* sessionInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetDriverModel_v2(nvmlDevice_t device, nvmlDriverModel_t* current, nvmlDriverModel_t* pending) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetVbiosVersion(nvmlDevice_t device, char* version, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetBridgeChipInfo(nvmlDevice_t device, nvmlBridgeChipHierarchy_t* bridgeHierarchy) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetGraphicsRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetMPSComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetRunningProcessDetailList(nvmlDevice_t device, nvmlProcessDetailList_t* plist) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceOnSameBoard(nvmlDevice_t device1, nvmlDevice_t device2, int* onSameBoard) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t* isRestricted) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetSamples(nvmlDevice_t device, nvmlSamplingType_t type, unsigned long long lastSeenTimeStamp, nvmlValueType_t* sampleValType, unsigned int* sampleCount, nvmlSample_t* samples) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t device, nvmlBAR1Memory_t* bar1Memory) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetIrqNum(nvmlDevice_t device, unsigned int* irqNum) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetNumGpuCores(nvmlDevice_t device, unsigned int* numCores) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetPowerSource(nvmlDevice_t device, nvmlPowerSource_t* powerSource) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetMemoryBusWidth(nvmlDevice_t device, unsigned int* busWidth) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetPcieLinkMaxSpeed(nvmlDevice_t device, unsigned int* maxSpeed) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetPcieSpeed(nvmlDevice_t device, unsigned int* pcieSpeed) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetAdaptiveClockInfoStatus(nvmlDevice_t device, unsigned int* adaptiveClockStatus) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetBusType(nvmlDevice_t device, nvmlBusType_t* type) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetGpuFabricInfoV(nvmlDevice_t device, nvmlGpuFabricInfoV_t* gpuFabricInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlSystemGetConfComputeCapabilities(nvmlConfComputeSystemCaps_t* capabilities) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlSystemGetConfComputeState(nvmlConfComputeSystemState_t* state) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetConfComputeMemSizeInfo(nvmlDevice_t device, nvmlConfComputeMemSizeInfo_t* memInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlSystemGetConfComputeGpusReadyState(unsigned int* isAcceptingWork) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetConfComputeProtectedMemoryUsage(nvmlDevice_t device, nvmlMemory_t* memory) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetConfComputeGpuCertificate(nvmlDevice_t device, nvmlConfComputeGpuCertificate_t* gpuCert) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetConfComputeGpuAttestationReport(nvmlDevice_t device, nvmlConfComputeGpuAttestationReport_t* gpuAtstReport) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlSystemGetConfComputeKeyRotationThresholdInfo(nvmlConfComputeGetKeyRotationThresholdInfo_t* pKeyRotationThrInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetConfComputeUnprotectedMemSize(nvmlDevice_t device, unsigned long long sizeKiB) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlSystemSetConfComputeGpusReadyState(unsigned int isAcceptingWork) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlSystemSetConfComputeKeyRotationThresholdInfo(nvmlConfComputeSetKeyRotationThresholdInfo_t* pKeyRotationThrInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlSystemGetConfComputeSettings(nvmlSystemConfComputeSettings_t* settings) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetGspFirmwareVersion(nvmlDevice_t device, char* version) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetGspFirmwareMode(nvmlDevice_t device, unsigned int* isEnabled, unsigned int* defaultMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetSramEccErrorStatus(nvmlDevice_t device, nvmlEccSramErrorStatus_t* status) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetAccountingMode(nvmlDevice_t device, nvmlEnableState_t* mode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetAccountingStats(nvmlDevice_t device, unsigned int pid, nvmlAccountingStats_t* stats) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetAccountingPids(nvmlDevice_t device, unsigned int* count, unsigned int* pids) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetAccountingBufferSize(nvmlDevice_t device, unsigned int* bufferSize) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetRetiredPages(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int* pageCount, unsigned long long* addresses) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int* pageCount, unsigned long long* addresses, unsigned long long* timestamps) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device, nvmlEnableState_t* isPending) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetRemappedRows(nvmlDevice_t device, unsigned int* corrRows, unsigned int* uncRows, unsigned int* isPending, unsigned int* failureOccurred) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetRowRemapperHistogram(nvmlDevice_t device, nvmlRowRemapperHistogramValues_t* values) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetArchitecture(nvmlDevice_t device, nvmlDeviceArchitecture_t* arch) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetClkMonStatus(nvmlDevice_t device, nvmlClkMonStatus_t* status) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetProcessUtilization(nvmlDevice_t device, nvmlProcessUtilizationSample_t* utilization, unsigned int* processSamplesCount, unsigned long long lastSeenTimeStamp) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetProcessesUtilizationInfo(nvmlDevice_t device, nvmlProcessesUtilizationInfo_t* procesesUtilInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetPlatformInfo(nvmlDevice_t device, nvmlPlatformInfo_t* platformInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlUnitSetLedState(nvmlUnit_t unit, nvmlLedColor_t color) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t mode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetComputeMode(nvmlDevice_t device, nvmlComputeMode_t mode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetEccMode(nvmlDevice_t device, nvmlEnableState_t ecc) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceClearEccErrorCounts(nvmlDevice_t device, nvmlEccCounterType_t counterType) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetDriverModel(nvmlDevice_t device, nvmlDriverModel_t driverModel, unsigned int flags) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetGpuLockedClocks(nvmlDevice_t device, unsigned int minGpuClockMHz, unsigned int maxGpuClockMHz) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceResetGpuLockedClocks(nvmlDevice_t device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetMemoryLockedClocks(nvmlDevice_t device, unsigned int minMemClockMHz, unsigned int maxMemClockMHz) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceResetMemoryLockedClocks(nvmlDevice_t device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetDefaultAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled, unsigned int flags) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetDefaultFanSpeed_v2(nvmlDevice_t device, unsigned int fan) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetFanControlPolicy(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t policy) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, int* temp) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t mode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t isRestricted) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int speed) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetAccountingMode(nvmlDevice_t device, nvmlEnableState_t mode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceClearAccountingPids(nvmlDevice_t device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetPowerManagementLimit_v2(nvmlDevice_t device, nvmlPowerValue_v2_t* powerValue) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t* isActive) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetNvLinkVersion(nvmlDevice_t device, unsigned int link, unsigned int* version) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int* capResult) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t* pci) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetNvLinkErrorCounter(nvmlDevice_t device, unsigned int link, nvmlNvLinkErrorCounter_t counter, unsigned long long* counterValue) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceResetNvLinkErrorCounters(nvmlDevice_t device, unsigned int link) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link, nvmlIntNvLinkDeviceType_t* pNvLinkDeviceType) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetNvLinkDeviceLowPowerThreshold(nvmlDevice_t device, nvmlNvLinkPowerThres_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlSystemSetNvlinkBwMode(unsigned int nvlinkBwMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlSystemGetNvlinkBwMode(unsigned int* nvlinkBwMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetNvlinkSupportedBwModes(nvmlDevice_t device, nvmlNvlinkSupportedBwModes_t* supportedBwMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetNvlinkBwMode(nvmlDevice_t device, nvmlNvlinkGetBwMode_t* getBwMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetNvlinkBwMode(nvmlDevice_t device, nvmlNvlinkSetBwMode_t* setBwMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlEventSetCreate(nvmlEventSet_t* set) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceRegisterEvents(nvmlDevice_t device, unsigned long long eventTypes, nvmlEventSet_t set) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetSupportedEventTypes(nvmlDevice_t device, unsigned long long* eventTypes) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlEventSetWait_v2(nvmlEventSet_t set, nvmlEventData_t* data, unsigned int timeoutms) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlEventSetFree(nvmlEventSet_t set) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlSystemEventSetCreate(nvmlSystemEventSetCreateRequest_t* request) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlSystemEventSetFree(nvmlSystemEventSetFreeRequest_t* request) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlSystemRegisterEvents(nvmlSystemRegisterEventRequest_t* request) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlSystemEventSetWait(nvmlSystemEventSetWaitRequest_t* request) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceModifyDrainState(nvmlPciInfo_t* pciInfo, nvmlEnableState_t newState) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceQueryDrainState(nvmlPciInfo_t* pciInfo, nvmlEnableState_t* currentState) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceRemoveGpu_v2(nvmlPciInfo_t* pciInfo, nvmlDetachGpuState_t gpuState, nvmlPcieLinkState_t linkState) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceDiscoverGpus(nvmlPciInfo_t* pciInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceClearFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t* pVirtualMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetHostVgpuMode(nvmlDevice_t device, nvmlHostVgpuMode_t* pHostVgpuMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t virtualMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetVgpuHeterogeneousMode(nvmlDevice_t device, nvmlVgpuHeterogeneousMode_t* pHeterogeneousMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetVgpuHeterogeneousMode(nvmlDevice_t device, const nvmlVgpuHeterogeneousMode_t* pHeterogeneousMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetPlacementId(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuPlacementId_t* pPlacement) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetVgpuTypeSupportedPlacements(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuPlacementList_t* pPlacementList) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetVgpuTypeCreatablePlacements(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuPlacementList_t* pPlacementList) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuTypeGetGspHeapSize(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* gspHeapSize) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuTypeGetFbReservation(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* fbReservation) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetRuntimeStateSize(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuRuntimeState_t* pState) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetVgpuCapabilities(nvmlDevice_t device, nvmlDeviceVgpuCapability_t capability, nvmlEnableState_t state) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetGridLicensableFeatures_v4(nvmlDevice_t device, nvmlGridLicensableFeatures_t* pGridLicensableFeatures) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGetVgpuDriverCapabilities(nvmlVgpuDriverCapability_t capability, unsigned int* capResult) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetVgpuCapabilities(nvmlDevice_t device, nvmlDeviceVgpuCapability_t capability, unsigned int* capResult) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetSupportedVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuTypeId_t* vgpuTypeIds) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetCreatableVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuTypeId_t* vgpuTypeIds) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuTypeGetClass(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeClass, unsigned int* size) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuTypeGetName(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeName, unsigned int* size) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuTypeGetGpuInstanceProfileId(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* gpuInstanceProfileId) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuTypeGetDeviceID(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* deviceID, unsigned long long* subsystemID) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuTypeGetFramebufferSize(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* fbSize) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuTypeGetNumDisplayHeads(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* numDisplayHeads) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuTypeGetResolution(nvmlVgpuTypeId_t vgpuTypeId, unsigned int displayIndex, unsigned int* xdim, unsigned int* ydim) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuTypeGetLicense(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeLicenseString, unsigned int size) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuTypeGetFrameRateLimit(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* frameRateLimit) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuTypeGetMaxInstances(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, unsigned int* vgpuInstanceCount) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuTypeGetMaxInstancesPerVm(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* vgpuInstanceCountPerVm) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuTypeGetBAR1Info(nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuTypeBar1Info_t* bar1Info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetActiveVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuInstance_t* vgpuInstances) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetVmID(nvmlVgpuInstance_t vgpuInstance, char* vmId, unsigned int size, nvmlVgpuVmIdType_t* vmIdType) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetUUID(nvmlVgpuInstance_t vgpuInstance, char* uuid, unsigned int size) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetVmDriverVersion(nvmlVgpuInstance_t vgpuInstance, char* version, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetFbUsage(nvmlVgpuInstance_t vgpuInstance, unsigned long long* fbUsage) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetLicenseStatus(nvmlVgpuInstance_t vgpuInstance, unsigned int* licensed) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetType(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuTypeId_t* vgpuTypeId) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetFrameRateLimit(nvmlVgpuInstance_t vgpuInstance, unsigned int* frameRateLimit) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetEccMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t* eccMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int* encoderCapacity) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceSetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int encoderCapacity) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetEncoderStats(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, unsigned int* averageFps, unsigned int* averageLatency) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetEncoderSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, nvmlEncoderSessionInfo_t* sessionInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetFBCStats(nvmlVgpuInstance_t vgpuInstance, nvmlFBCStats_t* fbcStats) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetFBCSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, nvmlFBCSessionInfo_t* sessionInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetGpuInstanceId(nvmlVgpuInstance_t vgpuInstance, unsigned int* gpuInstanceId) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetGpuPciId(nvmlVgpuInstance_t vgpuInstance, char* vgpuPciId, unsigned int* length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuTypeGetCapabilities(nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuCapability_t capability, unsigned int* capResult) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetMdevUUID(nvmlVgpuInstance_t vgpuInstance, char* mdevUuid, unsigned int size) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGpuInstanceGetCreatableVgpus(nvmlGpuInstance_t gpuInstance, nvmlVgpuTypeIdInfo_t* pVgpus) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuTypeGetMaxInstancesPerGpuInstance(nvmlVgpuTypeMaxInstance_t* pMaxInstance) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGpuInstanceGetActiveVgpus(nvmlGpuInstance_t gpuInstance, nvmlActiveVgpuInstanceInfo_t* pVgpuInstanceInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGpuInstanceSetVgpuSchedulerState(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerState_t* pScheduler) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGpuInstanceGetVgpuSchedulerState(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerStateInfo_t* pSchedulerStateInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGpuInstanceGetVgpuSchedulerLog(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerLogInfo_t* pSchedulerLogInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGpuInstanceGetVgpuTypeCreatablePlacements(nvmlGpuInstance_t gpuInstance, nvmlVgpuCreatablePlacementInfo_t* pCreatablePlacementInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGpuInstanceGetVgpuHeterogeneousMode(nvmlGpuInstance_t gpuInstance, nvmlVgpuHeterogeneousMode_t* pHeterogeneousMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGpuInstanceSetVgpuHeterogeneousMode(nvmlGpuInstance_t gpuInstance, const nvmlVgpuHeterogeneousMode_t* pHeterogeneousMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetMetadata(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuMetadata_t* vgpuMetadata, unsigned int* bufferSize) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetVgpuMetadata(nvmlDevice_t device, nvmlVgpuPgpuMetadata_t* pgpuMetadata, unsigned int* bufferSize) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGetVgpuCompatibility(nvmlVgpuMetadata_t* vgpuMetadata, nvmlVgpuPgpuMetadata_t* pgpuMetadata, nvmlVgpuPgpuCompatibility_t* compatibilityInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetPgpuMetadataString(nvmlDevice_t device, char* pgpuMetadata, unsigned int* bufferSize) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetVgpuSchedulerLog(nvmlDevice_t device, nvmlVgpuSchedulerLog_t* pSchedulerLog) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetVgpuSchedulerState(nvmlDevice_t device, nvmlVgpuSchedulerGetState_t* pSchedulerState) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetVgpuSchedulerCapabilities(nvmlDevice_t device, nvmlVgpuSchedulerCapabilities_t* pCapabilities) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetVgpuSchedulerState(nvmlDevice_t device, nvmlVgpuSchedulerSetState_t* pSchedulerState) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGetVgpuVersion(nvmlVgpuVersion_t* supported, nvmlVgpuVersion_t* current) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlSetVgpuVersion(nvmlVgpuVersion_t* vgpuVersion) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetVgpuUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, nvmlValueType_t* sampleValType, unsigned int* vgpuInstanceSamplesCount, nvmlVgpuInstanceUtilizationSample_t* utilizationSamples) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetVgpuInstancesUtilizationInfo(nvmlDevice_t device, nvmlVgpuInstancesUtilizationInfo_t* vgpuUtilInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetVgpuProcessUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, unsigned int* vgpuProcessSamplesCount, nvmlVgpuProcessUtilizationSample_t* utilizationSamples) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetVgpuProcessesUtilizationInfo(nvmlDevice_t device, nvmlVgpuProcessesUtilizationInfo_t* vgpuProcUtilInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetAccountingMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t* mode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetAccountingPids(nvmlVgpuInstance_t vgpuInstance, unsigned int* count, unsigned int* pids) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetAccountingStats(nvmlVgpuInstance_t vgpuInstance, unsigned int pid, nvmlAccountingStats_t* stats) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceClearAccountingPids(nvmlVgpuInstance_t vgpuInstance) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlVgpuInstanceGetLicenseInfo_v2(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuLicenseInfo_t* licenseInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGetExcludedDeviceCount(unsigned int* deviceCount) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGetExcludedDeviceInfoByIndex(unsigned int index, nvmlExcludedDeviceInfo_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetMigMode(nvmlDevice_t device, unsigned int mode, nvmlReturn_t* activationStatus) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int* currentMode, unsigned int* pendingMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetGpuInstanceProfileInfoV(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_v2_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetGpuInstancePossiblePlacements_v2(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstancePlacement_t* placements, unsigned int* count) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetGpuInstanceRemainingCapacity(nvmlDevice_t device, unsigned int profileId, unsigned int* count) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceCreateGpuInstance(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t* gpuInstance) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceCreateGpuInstanceWithPlacement(nvmlDevice_t device, unsigned int profileId, const nvmlGpuInstancePlacement_t* placement, nvmlGpuInstance_t* gpuInstance) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGpuInstanceDestroy(nvmlGpuInstance_t gpuInstance) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetGpuInstances(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t* gpuInstances, unsigned int* count) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetGpuInstanceById(nvmlDevice_t device, unsigned int id, nvmlGpuInstance_t* gpuInstance) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGpuInstanceGetInfo(nvmlGpuInstance_t gpuInstance, nvmlGpuInstanceInfo_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGpuInstanceGetComputeInstanceProfileInfoV(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_v2_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGpuInstanceGetComputeInstanceRemainingCapacity(nvmlGpuInstance_t gpuInstance, unsigned int profileId, unsigned int* count) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGpuInstanceGetComputeInstancePossiblePlacements(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstancePlacement_t* placements, unsigned int* count) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGpuInstanceCreateComputeInstance(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t* computeInstance) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGpuInstanceCreateComputeInstanceWithPlacement(nvmlGpuInstance_t gpuInstance, unsigned int profileId, const nvmlComputeInstancePlacement_t* placement, nvmlComputeInstance_t* computeInstance) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlComputeInstanceDestroy(nvmlComputeInstance_t computeInstance) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGpuInstanceGetComputeInstances(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t* computeInstances, unsigned int* count) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGpuInstanceGetComputeInstanceById(nvmlGpuInstance_t gpuInstance, unsigned int id, nvmlComputeInstance_t* computeInstance) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlComputeInstanceGetInfo_v2(nvmlComputeInstance_t computeInstance, nvmlComputeInstanceInfo_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device, unsigned int* isMigDevice) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetGpuInstanceId(nvmlDevice_t device, unsigned int* id) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetComputeInstanceId(nvmlDevice_t device, unsigned int* id) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device, unsigned int* count) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t device, unsigned int index, nvmlDevice_t* migDevice) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t migDevice, nvmlDevice_t* device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetCapabilities(nvmlDevice_t device, nvmlDeviceCapabilities_t* caps) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDevicePowerSmoothingActivatePresetProfile(nvmlDevice_t device, nvmlPowerSmoothingProfile_t* profile) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDevicePowerSmoothingUpdatePresetProfileParam(nvmlDevice_t device, nvmlPowerSmoothingProfile_t* profile) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDevicePowerSmoothingSetState(nvmlDevice_t device, nvmlPowerSmoothingState_t* state) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetAddressingMode(nvmlDevice_t device, nvmlDeviceAddressingMode_t* mode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetRepairStatus(nvmlDevice_t device, nvmlRepairStatus_t* repairStatus) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetPowerMizerMode_v1(nvmlDevice_t device, nvmlDevicePowerMizerModes_v1_t* powerMizerMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetPowerMizerMode_v1(nvmlDevice_t device, nvmlDevicePowerMizerModes_v1_t* powerMizerMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetPdi(nvmlDevice_t device, nvmlPdi_t* pdi) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetHostname_v1(nvmlDevice_t device, nvmlHostname_v1_t* hostname) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetHostname_v1(nvmlDevice_t device, nvmlHostname_v1_t* hostname) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetNvLinkInfo(nvmlDevice_t device, nvmlNvLinkInfo_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceReadWritePRM_v1(nvmlDevice_t device, nvmlPRMTLV_v1_t* buffer) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetGpuInstanceProfileInfoByIdV(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstanceProfileInfo_v2_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts(nvmlDevice_t device, nvmlEccSramUniqueUncorrectedErrorCounts_t* errorCounts) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetUnrepairableMemoryFlag_v1(nvmlDevice_t device, nvmlUnrepairableMemoryStatus_v1_t* unrepairableMemoryStatus) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceReadPRMCounters_v1(nvmlDevice_t device, nvmlPRMCounterList_v1_t* counterList) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetRusdSettings_v1(nvmlDevice_t device, nvmlRusdSettings_v1_t* settings) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceVgpuForceGspUnload(nvmlDevice_t device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetVgpuSchedulerState_v2(nvmlDevice_t device, nvmlVgpuSchedulerStateInfo_v2_t* pSchedulerStateInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGpuInstanceGetVgpuSchedulerState_v2(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerStateInfo_v2_t* pSchedulerStateInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceGetVgpuSchedulerLog_v2(nvmlDevice_t device, nvmlVgpuSchedulerLogInfo_v2_t* pSchedulerLogInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGpuInstanceGetVgpuSchedulerLog_v2(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerLogInfo_v2_t* pSchedulerLogInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlDeviceSetVgpuSchedulerState_v2(nvmlDevice_t device, nvmlVgpuSchedulerState_v2_t* pSchedulerState) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t _nvmlGpuInstanceSetVgpuSchedulerState_v2(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerState_v2_t* pSchedulerState) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
