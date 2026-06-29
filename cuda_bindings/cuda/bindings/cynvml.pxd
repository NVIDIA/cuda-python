# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.9.1 to 13.3.0, generator version 0.3.1.dev1779+ga8cc71818.d20260626. Do not modify it directly.

from libc.stdint cimport int64_t


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
cdef extern from 'nvml.h':
    ctypedef enum nvmlBridgeChipType_t:
        NVML_BRIDGE_CHIP_PLX
        NVML_BRIDGE_CHIP_BRO4

cdef extern from 'nvml.h':
    ctypedef enum nvmlNvLinkUtilizationCountUnits_t:
        NVML_NVLINK_COUNTER_UNIT_CYCLES
        NVML_NVLINK_COUNTER_UNIT_PACKETS
        NVML_NVLINK_COUNTER_UNIT_BYTES
        NVML_NVLINK_COUNTER_UNIT_RESERVED
        NVML_NVLINK_COUNTER_UNIT_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlNvLinkUtilizationCountPktTypes_t:
        NVML_NVLINK_COUNTER_PKTFILTER_NOP
        NVML_NVLINK_COUNTER_PKTFILTER_READ
        NVML_NVLINK_COUNTER_PKTFILTER_WRITE
        NVML_NVLINK_COUNTER_PKTFILTER_RATOM
        NVML_NVLINK_COUNTER_PKTFILTER_NRATOM
        NVML_NVLINK_COUNTER_PKTFILTER_FLUSH
        NVML_NVLINK_COUNTER_PKTFILTER_RESPDATA
        NVML_NVLINK_COUNTER_PKTFILTER_RESPNODATA
        NVML_NVLINK_COUNTER_PKTFILTER_ALL

cdef extern from 'nvml.h':
    ctypedef enum nvmlNvLinkCapability_t:
        NVML_NVLINK_CAP_P2P_SUPPORTED
        NVML_NVLINK_CAP_SYSMEM_ACCESS
        NVML_NVLINK_CAP_P2P_ATOMICS
        NVML_NVLINK_CAP_SYSMEM_ATOMICS
        NVML_NVLINK_CAP_SLI_BRIDGE
        NVML_NVLINK_CAP_VALID
        NVML_NVLINK_CAP_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlNvLinkErrorCounter_t:
        NVML_NVLINK_ERROR_DL_REPLAY
        NVML_NVLINK_ERROR_DL_RECOVERY
        NVML_NVLINK_ERROR_DL_CRC_FLIT
        NVML_NVLINK_ERROR_DL_CRC_DATA
        NVML_NVLINK_ERROR_DL_ECC_DATA
        NVML_NVLINK_ERROR_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlIntNvLinkDeviceType_t:
        NVML_NVLINK_DEVICE_TYPE_GPU
        NVML_NVLINK_DEVICE_TYPE_IBMNPU
        NVML_NVLINK_DEVICE_TYPE_SWITCH
        NVML_NVLINK_DEVICE_TYPE_UNKNOWN

cdef extern from 'nvml.h':
    ctypedef enum nvmlGpuTopologyLevel_t:
        NVML_TOPOLOGY_INTERNAL
        NVML_TOPOLOGY_SINGLE
        NVML_TOPOLOGY_MULTIPLE
        NVML_TOPOLOGY_HOSTBRIDGE
        NVML_TOPOLOGY_NODE
        NVML_TOPOLOGY_SYSTEM

cdef extern from 'nvml.h':
    ctypedef enum nvmlGpuP2PStatus_t:
        NVML_P2P_STATUS_OK
        NVML_P2P_STATUS_CHIPSET_NOT_SUPPORED
        NVML_P2P_STATUS_CHIPSET_NOT_SUPPORTED
        NVML_P2P_STATUS_GPU_NOT_SUPPORTED
        NVML_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED
        NVML_P2P_STATUS_DISABLED_BY_REGKEY
        NVML_P2P_STATUS_NOT_SUPPORTED
        NVML_P2P_STATUS_UNKNOWN

cdef extern from 'nvml.h':
    ctypedef enum nvmlGpuP2PCapsIndex_t:
        NVML_P2P_CAPS_INDEX_READ
        NVML_P2P_CAPS_INDEX_WRITE
        NVML_P2P_CAPS_INDEX_NVLINK
        NVML_P2P_CAPS_INDEX_ATOMICS
        NVML_P2P_CAPS_INDEX_PCI
        NVML_P2P_CAPS_INDEX_PROP
        NVML_P2P_CAPS_INDEX_UNKNOWN

cdef extern from 'nvml.h':
    ctypedef enum nvmlSamplingType_t:
        NVML_TOTAL_POWER_SAMPLES
        NVML_GPU_UTILIZATION_SAMPLES
        NVML_MEMORY_UTILIZATION_SAMPLES
        NVML_ENC_UTILIZATION_SAMPLES
        NVML_DEC_UTILIZATION_SAMPLES
        NVML_PROCESSOR_CLK_SAMPLES
        NVML_MEMORY_CLK_SAMPLES
        NVML_MODULE_POWER_SAMPLES
        NVML_JPG_UTILIZATION_SAMPLES
        NVML_OFA_UTILIZATION_SAMPLES
        NVML_SAMPLINGTYPE_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlPcieUtilCounter_t:
        NVML_PCIE_UTIL_TX_BYTES
        NVML_PCIE_UTIL_RX_BYTES
        NVML_PCIE_UTIL_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlValueType_t:
        NVML_VALUE_TYPE_DOUBLE
        NVML_VALUE_TYPE_UNSIGNED_INT
        NVML_VALUE_TYPE_UNSIGNED_LONG
        NVML_VALUE_TYPE_UNSIGNED_LONG_LONG
        NVML_VALUE_TYPE_SIGNED_LONG_LONG
        NVML_VALUE_TYPE_SIGNED_INT
        NVML_VALUE_TYPE_UNSIGNED_SHORT
        NVML_VALUE_TYPE_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlPerfPolicyType_t:
        NVML_PERF_POLICY_POWER
        NVML_PERF_POLICY_THERMAL
        NVML_PERF_POLICY_SYNC_BOOST
        NVML_PERF_POLICY_BOARD_LIMIT
        NVML_PERF_POLICY_LOW_UTILIZATION
        NVML_PERF_POLICY_RELIABILITY
        NVML_PERF_POLICY_TOTAL_APP_CLOCKS
        NVML_PERF_POLICY_TOTAL_BASE_CLOCKS
        NVML_PERF_POLICY_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlThermalTarget_t:
        NVML_THERMAL_TARGET_NONE
        NVML_THERMAL_TARGET_GPU
        NVML_THERMAL_TARGET_MEMORY
        NVML_THERMAL_TARGET_POWER_SUPPLY
        NVML_THERMAL_TARGET_BOARD
        NVML_THERMAL_TARGET_VCD_BOARD
        NVML_THERMAL_TARGET_VCD_INLET
        NVML_THERMAL_TARGET_VCD_OUTLET
        NVML_THERMAL_TARGET_ALL
        NVML_THERMAL_TARGET_UNKNOWN

cdef extern from 'nvml.h':
    ctypedef enum nvmlThermalController_t:
        NVML_THERMAL_CONTROLLER_NONE
        NVML_THERMAL_CONTROLLER_GPU_INTERNAL
        NVML_THERMAL_CONTROLLER_ADM1032
        NVML_THERMAL_CONTROLLER_ADT7461
        NVML_THERMAL_CONTROLLER_MAX6649
        NVML_THERMAL_CONTROLLER_MAX1617
        NVML_THERMAL_CONTROLLER_LM99
        NVML_THERMAL_CONTROLLER_LM89
        NVML_THERMAL_CONTROLLER_LM64
        NVML_THERMAL_CONTROLLER_G781
        NVML_THERMAL_CONTROLLER_ADT7473
        NVML_THERMAL_CONTROLLER_SBMAX6649
        NVML_THERMAL_CONTROLLER_VBIOSEVT
        NVML_THERMAL_CONTROLLER_OS
        NVML_THERMAL_CONTROLLER_NVSYSCON_CANOAS
        NVML_THERMAL_CONTROLLER_NVSYSCON_E551
        NVML_THERMAL_CONTROLLER_MAX6649R
        NVML_THERMAL_CONTROLLER_ADT7473S
        NVML_THERMAL_CONTROLLER_UNKNOWN

cdef extern from 'nvml.h':
    ctypedef enum nvmlCoolerControl_t:
        NVML_THERMAL_COOLER_SIGNAL_NONE
        NVML_THERMAL_COOLER_SIGNAL_TOGGLE
        NVML_THERMAL_COOLER_SIGNAL_VARIABLE
        NVML_THERMAL_COOLER_SIGNAL_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlCoolerTarget_t:
        NVML_THERMAL_COOLER_TARGET_NONE
        NVML_THERMAL_COOLER_TARGET_GPU
        NVML_THERMAL_COOLER_TARGET_MEMORY
        NVML_THERMAL_COOLER_TARGET_POWER_SUPPLY
        NVML_THERMAL_COOLER_TARGET_GPU_RELATED

cdef extern from 'nvml.h':
    ctypedef enum nvmlUUIDType_t:
        NVML_UUID_TYPE_NONE
        NVML_UUID_TYPE_ASCII
        NVML_UUID_TYPE_BINARY

cdef extern from 'nvml.h':
    ctypedef enum nvmlEnableState_t:
        NVML_FEATURE_DISABLED
        NVML_FEATURE_ENABLED

cdef extern from 'nvml.h':
    ctypedef enum nvmlBrandType_t:
        NVML_BRAND_UNKNOWN
        NVML_BRAND_QUADRO
        NVML_BRAND_TESLA
        NVML_BRAND_NVS
        NVML_BRAND_GRID
        NVML_BRAND_GEFORCE
        NVML_BRAND_TITAN
        NVML_BRAND_NVIDIA_VAPPS
        NVML_BRAND_NVIDIA_VPC
        NVML_BRAND_NVIDIA_VCS
        NVML_BRAND_NVIDIA_VWS
        NVML_BRAND_NVIDIA_CLOUD_GAMING
        NVML_BRAND_NVIDIA_VGAMING
        NVML_BRAND_QUADRO_RTX
        NVML_BRAND_NVIDIA_RTX
        NVML_BRAND_NVIDIA
        NVML_BRAND_GEFORCE_RTX
        NVML_BRAND_TITAN_RTX
        NVML_BRAND_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlTemperatureThresholds_t:
        NVML_TEMPERATURE_THRESHOLD_SHUTDOWN
        NVML_TEMPERATURE_THRESHOLD_SLOWDOWN
        NVML_TEMPERATURE_THRESHOLD_MEM_MAX
        NVML_TEMPERATURE_THRESHOLD_GPU_MAX
        NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MIN
        NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_CURR
        NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MAX
        NVML_TEMPERATURE_THRESHOLD_GPS_CURR
        NVML_TEMPERATURE_THRESHOLD_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlTemperatureSensors_t:
        NVML_TEMPERATURE_GPU
        NVML_TEMPERATURE_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlComputeMode_t:
        NVML_COMPUTEMODE_DEFAULT
        NVML_COMPUTEMODE_EXCLUSIVE_THREAD
        NVML_COMPUTEMODE_PROHIBITED
        NVML_COMPUTEMODE_EXCLUSIVE_PROCESS
        NVML_COMPUTEMODE_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlMemoryErrorType_t:
        NVML_MEMORY_ERROR_TYPE_CORRECTED
        NVML_MEMORY_ERROR_TYPE_UNCORRECTED
        NVML_MEMORY_ERROR_TYPE_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlNvlinkVersion_t:
        NVML_NVLINK_VERSION_INVALID
        NVML_NVLINK_VERSION_1_0
        NVML_NVLINK_VERSION_2_0
        NVML_NVLINK_VERSION_2_2
        NVML_NVLINK_VERSION_3_0
        NVML_NVLINK_VERSION_3_1
        NVML_NVLINK_VERSION_4_0
        NVML_NVLINK_VERSION_5_0
        NVML_NVLINK_VERSION_6_0

cdef extern from 'nvml.h':
    ctypedef enum nvmlEccCounterType_t:
        NVML_VOLATILE_ECC
        NVML_AGGREGATE_ECC
        NVML_ECC_COUNTER_TYPE_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlClockType_t:
        NVML_CLOCK_GRAPHICS
        NVML_CLOCK_SM
        NVML_CLOCK_MEM
        NVML_CLOCK_VIDEO
        NVML_CLOCK_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlClockId_t:
        NVML_CLOCK_ID_CURRENT
        NVML_CLOCK_ID_APP_CLOCK_TARGET
        NVML_CLOCK_ID_APP_CLOCK_DEFAULT
        NVML_CLOCK_ID_CUSTOMER_BOOST_MAX
        NVML_CLOCK_ID_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlDriverModel_t:
        NVML_DRIVER_WDDM
        NVML_DRIVER_WDM
        NVML_DRIVER_MCDM

cdef extern from 'nvml.h':
    ctypedef enum nvmlPstates_t:
        NVML_PSTATE_0
        NVML_PSTATE_1
        NVML_PSTATE_2
        NVML_PSTATE_3
        NVML_PSTATE_4
        NVML_PSTATE_5
        NVML_PSTATE_6
        NVML_PSTATE_7
        NVML_PSTATE_8
        NVML_PSTATE_9
        NVML_PSTATE_10
        NVML_PSTATE_11
        NVML_PSTATE_12
        NVML_PSTATE_13
        NVML_PSTATE_14
        NVML_PSTATE_15
        NVML_PSTATE_UNKNOWN

cdef extern from 'nvml.h':
    ctypedef enum nvmlGpuOperationMode_t:
        NVML_GOM_ALL_ON
        NVML_GOM_COMPUTE
        NVML_GOM_LOW_DP

cdef extern from 'nvml.h':
    ctypedef enum nvmlInforomObject_t:
        NVML_INFOROM_OEM
        NVML_INFOROM_ECC
        NVML_INFOROM_POWER
        NVML_INFOROM_DEN
        NVML_INFOROM_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlReturn_t:
        NVML_SUCCESS
        NVML_ERROR_UNINITIALIZED
        NVML_ERROR_INVALID_ARGUMENT
        NVML_ERROR_NOT_SUPPORTED
        NVML_ERROR_NO_PERMISSION
        NVML_ERROR_ALREADY_INITIALIZED
        NVML_ERROR_NOT_FOUND
        NVML_ERROR_INSUFFICIENT_SIZE
        NVML_ERROR_INSUFFICIENT_POWER
        NVML_ERROR_DRIVER_NOT_LOADED
        NVML_ERROR_TIMEOUT
        NVML_ERROR_IRQ_ISSUE
        NVML_ERROR_LIBRARY_NOT_FOUND
        NVML_ERROR_FUNCTION_NOT_FOUND
        NVML_ERROR_CORRUPTED_INFOROM
        NVML_ERROR_GPU_IS_LOST
        NVML_ERROR_RESET_REQUIRED
        NVML_ERROR_OPERATING_SYSTEM
        NVML_ERROR_LIB_RM_VERSION_MISMATCH
        NVML_ERROR_IN_USE
        NVML_ERROR_MEMORY
        NVML_ERROR_NO_DATA
        NVML_ERROR_VGPU_ECC_NOT_SUPPORTED
        NVML_ERROR_INSUFFICIENT_RESOURCES
        NVML_ERROR_FREQ_NOT_SUPPORTED
        NVML_ERROR_ARGUMENT_VERSION_MISMATCH
        NVML_ERROR_DEPRECATED
        NVML_ERROR_NOT_READY
        NVML_ERROR_GPU_NOT_FOUND
        NVML_ERROR_INVALID_STATE
        NVML_ERROR_RESET_TYPE_NOT_SUPPORTED
        NVML_ERROR_UNKNOWN

cdef extern from 'nvml.h':
    ctypedef enum nvmlMemoryLocation_t:
        NVML_MEMORY_LOCATION_L1_CACHE
        NVML_MEMORY_LOCATION_L2_CACHE
        NVML_MEMORY_LOCATION_DRAM
        NVML_MEMORY_LOCATION_DEVICE_MEMORY
        NVML_MEMORY_LOCATION_REGISTER_FILE
        NVML_MEMORY_LOCATION_TEXTURE_MEMORY
        NVML_MEMORY_LOCATION_TEXTURE_SHM
        NVML_MEMORY_LOCATION_CBU
        NVML_MEMORY_LOCATION_SRAM
        NVML_MEMORY_LOCATION_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlPageRetirementCause_t:
        NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS
        NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR
        NVML_PAGE_RETIREMENT_CAUSE_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlRestrictedAPI_t:
        NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS
        NVML_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS
        NVML_RESTRICTED_API_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlGpuUtilizationDomainId_t:
        NVML_GPU_UTILIZATION_DOMAIN_GPU
        NVML_GPU_UTILIZATION_DOMAIN_FB
        NVML_GPU_UTILIZATION_DOMAIN_VID
        NVML_GPU_UTILIZATION_DOMAIN_BUS

cdef extern from 'nvml.h':
    ctypedef enum nvmlGpuVirtualizationMode_t:
        NVML_GPU_VIRTUALIZATION_MODE_NONE
        NVML_GPU_VIRTUALIZATION_MODE_PASSTHROUGH
        NVML_GPU_VIRTUALIZATION_MODE_VGPU
        NVML_GPU_VIRTUALIZATION_MODE_HOST_VGPU
        NVML_GPU_VIRTUALIZATION_MODE_HOST_VSGA

cdef extern from 'nvml.h':
    ctypedef enum nvmlHostVgpuMode_t:
        NVML_HOST_VGPU_MODE_NON_SRIOV
        NVML_HOST_VGPU_MODE_SRIOV

cdef extern from 'nvml.h':
    ctypedef enum nvmlVgpuVmIdType_t:
        NVML_VGPU_VM_ID_DOMAIN_ID
        NVML_VGPU_VM_ID_UUID

cdef extern from 'nvml.h':
    ctypedef enum nvmlVgpuGuestInfoState_t:
        NVML_VGPU_INSTANCE_GUEST_INFO_STATE_UNINITIALIZED
        NVML_VGPU_INSTANCE_GUEST_INFO_STATE_INITIALIZED

cdef extern from 'nvml.h':
    ctypedef enum nvmlGridLicenseFeatureCode_t:
        NVML_GRID_LICENSE_FEATURE_CODE_UNKNOWN
        NVML_GRID_LICENSE_FEATURE_CODE_VGPU
        NVML_GRID_LICENSE_FEATURE_CODE_NVIDIA_RTX
        NVML_GRID_LICENSE_FEATURE_CODE_VWORKSTATION
        NVML_GRID_LICENSE_FEATURE_CODE_GAMING
        NVML_GRID_LICENSE_FEATURE_CODE_COMPUTE

cdef extern from 'nvml.h':
    ctypedef enum nvmlVgpuCapability_t:
        NVML_VGPU_CAP_NVLINK_P2P
        NVML_VGPU_CAP_GPUDIRECT
        NVML_VGPU_CAP_MULTI_VGPU_EXCLUSIVE
        NVML_VGPU_CAP_EXCLUSIVE_TYPE
        NVML_VGPU_CAP_EXCLUSIVE_SIZE
        NVML_VGPU_CAP_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlVgpuDriverCapability_t:
        NVML_VGPU_DRIVER_CAP_HETEROGENEOUS_MULTI_VGPU
        NVML_VGPU_DRIVER_CAP_WARM_UPDATE
        NVML_VGPU_DRIVER_CAP_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlDeviceVgpuCapability_t:
        NVML_DEVICE_VGPU_CAP_FRACTIONAL_MULTI_VGPU
        NVML_DEVICE_VGPU_CAP_HETEROGENEOUS_TIMESLICE_PROFILES
        NVML_DEVICE_VGPU_CAP_HETEROGENEOUS_TIMESLICE_SIZES
        NVML_DEVICE_VGPU_CAP_READ_DEVICE_BUFFER_BW
        NVML_DEVICE_VGPU_CAP_WRITE_DEVICE_BUFFER_BW
        NVML_DEVICE_VGPU_CAP_DEVICE_STREAMING
        NVML_DEVICE_VGPU_CAP_MINI_QUARTER_GPU
        NVML_DEVICE_VGPU_CAP_COMPUTE_MEDIA_ENGINE_GPU
        NVML_DEVICE_VGPU_CAP_WARM_UPDATE
        NVML_DEVICE_VGPU_CAP_HOMOGENEOUS_PLACEMENTS
        NVML_DEVICE_VGPU_CAP_MIG_TIMESLICING_SUPPORTED
        NVML_DEVICE_VGPU_CAP_MIG_TIMESLICING_ENABLED
        NVML_DEVICE_VGPU_CAP_COUNT

cdef extern from 'nvml.h':
    ctypedef enum nvmlDeviceGpuRecoveryAction_t:
        NVML_GPU_RECOVERY_ACTION_NONE
        NVML_GPU_RECOVERY_ACTION_GPU_RESET
        NVML_GPU_RECOVERY_ACTION_NODE_REBOOT
        NVML_GPU_RECOVERY_ACTION_DRAIN_P2P
        NVML_GPU_RECOVERY_ACTION_DRAIN_AND_RESET
        NVML_GPU_RECOVERY_ACTION_RECOVER_IMEX_DOMAIN

cdef extern from 'nvml.h':
    ctypedef enum nvmlFanState_t:
        NVML_FAN_NORMAL
        NVML_FAN_FAILED

cdef extern from 'nvml.h':
    ctypedef enum nvmlLedColor_t:
        NVML_LED_COLOR_GREEN
        NVML_LED_COLOR_AMBER

cdef extern from 'nvml.h':
    ctypedef enum nvmlEncoderType_t:
        NVML_ENCODER_QUERY_H264
        NVML_ENCODER_QUERY_HEVC
        NVML_ENCODER_QUERY_AV1
        NVML_ENCODER_QUERY_UNKNOWN

cdef extern from 'nvml.h':
    ctypedef enum nvmlFBCSessionType_t:
        NVML_FBC_SESSION_TYPE_UNKNOWN
        NVML_FBC_SESSION_TYPE_TOSYS
        NVML_FBC_SESSION_TYPE_CUDA
        NVML_FBC_SESSION_TYPE_VID
        NVML_FBC_SESSION_TYPE_HWENC

cdef extern from 'nvml.h':
    ctypedef enum nvmlDetachGpuState_t:
        NVML_DETACH_GPU_KEEP
        NVML_DETACH_GPU_REMOVE

cdef extern from 'nvml.h':
    ctypedef enum nvmlPcieLinkState_t:
        NVML_PCIE_LINK_KEEP
        NVML_PCIE_LINK_SHUT_DOWN

cdef extern from 'nvml.h':
    ctypedef enum nvmlClockLimitId_t:
        NVML_CLOCK_LIMIT_ID_RANGE_START
        NVML_CLOCK_LIMIT_ID_TDP
        NVML_CLOCK_LIMIT_ID_UNLIMITED

cdef extern from 'nvml.h':
    ctypedef enum nvmlVgpuVmCompatibility_t:
        NVML_VGPU_VM_COMPATIBILITY_NONE
        NVML_VGPU_VM_COMPATIBILITY_COLD
        NVML_VGPU_VM_COMPATIBILITY_HIBERNATE
        NVML_VGPU_VM_COMPATIBILITY_SLEEP
        NVML_VGPU_VM_COMPATIBILITY_LIVE

cdef extern from 'nvml.h':
    ctypedef enum nvmlVgpuPgpuCompatibilityLimitCode_t:
        NVML_VGPU_COMPATIBILITY_LIMIT_NONE
        NVML_VGPU_COMPATIBILITY_LIMIT_HOST_DRIVER
        NVML_VGPU_COMPATIBILITY_LIMIT_GUEST_DRIVER
        NVML_VGPU_COMPATIBILITY_LIMIT_GPU
        NVML_VGPU_COMPATIBILITY_LIMIT_OTHER

cdef extern from 'nvml.h':
    ctypedef enum nvmlGpmMetricId_t:
        NVML_GPM_METRIC_GRAPHICS_UTIL
        NVML_GPM_METRIC_SM_UTIL
        NVML_GPM_METRIC_SM_OCCUPANCY
        NVML_GPM_METRIC_INTEGER_UTIL
        NVML_GPM_METRIC_ANY_TENSOR_UTIL
        NVML_GPM_METRIC_DFMA_TENSOR_UTIL
        NVML_GPM_METRIC_HMMA_TENSOR_UTIL
        NVML_GPM_METRIC_DMMA_TENSOR_UTIL
        NVML_GPM_METRIC_IMMA_TENSOR_UTIL
        NVML_GPM_METRIC_DRAM_BW_UTIL
        NVML_GPM_METRIC_FP64_UTIL
        NVML_GPM_METRIC_FP32_UTIL
        NVML_GPM_METRIC_FP16_UTIL
        NVML_GPM_METRIC_PCIE_TX_PER_SEC
        NVML_GPM_METRIC_PCIE_RX_PER_SEC
        NVML_GPM_METRIC_NVDEC_0_UTIL
        NVML_GPM_METRIC_NVDEC_1_UTIL
        NVML_GPM_METRIC_NVDEC_2_UTIL
        NVML_GPM_METRIC_NVDEC_3_UTIL
        NVML_GPM_METRIC_NVDEC_4_UTIL
        NVML_GPM_METRIC_NVDEC_5_UTIL
        NVML_GPM_METRIC_NVDEC_6_UTIL
        NVML_GPM_METRIC_NVDEC_7_UTIL
        NVML_GPM_METRIC_NVJPG_0_UTIL
        NVML_GPM_METRIC_NVJPG_1_UTIL
        NVML_GPM_METRIC_NVJPG_2_UTIL
        NVML_GPM_METRIC_NVJPG_3_UTIL
        NVML_GPM_METRIC_NVJPG_4_UTIL
        NVML_GPM_METRIC_NVJPG_5_UTIL
        NVML_GPM_METRIC_NVJPG_6_UTIL
        NVML_GPM_METRIC_NVJPG_7_UTIL
        NVML_GPM_METRIC_NVOFA_0_UTIL
        NVML_GPM_METRIC_NVOFA_1_UTIL
        NVML_GPM_METRIC_NVLINK_TOTAL_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_TOTAL_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L0_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L0_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L1_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L1_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L2_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L2_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L3_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L3_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L4_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L4_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L5_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L5_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L6_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L6_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L7_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L7_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L8_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L8_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L9_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L9_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L10_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L10_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L11_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L11_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L12_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L12_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L13_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L13_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L14_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L14_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L15_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L15_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L16_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L16_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L17_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L17_TX_PER_SEC
        NVML_GPM_METRIC_C2C_TOTAL_TX_PER_SEC
        NVML_GPM_METRIC_C2C_TOTAL_RX_PER_SEC
        NVML_GPM_METRIC_C2C_DATA_TX_PER_SEC
        NVML_GPM_METRIC_C2C_DATA_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK0_TOTAL_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK0_TOTAL_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK0_DATA_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK0_DATA_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK1_TOTAL_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK1_TOTAL_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK1_DATA_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK1_DATA_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK2_TOTAL_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK2_TOTAL_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK2_DATA_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK2_DATA_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK3_TOTAL_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK3_TOTAL_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK3_DATA_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK3_DATA_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK4_TOTAL_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK4_TOTAL_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK4_DATA_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK4_DATA_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK5_TOTAL_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK5_TOTAL_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK5_DATA_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK5_DATA_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK6_TOTAL_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK6_TOTAL_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK6_DATA_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK6_DATA_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK7_TOTAL_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK7_TOTAL_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK7_DATA_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK7_DATA_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK8_TOTAL_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK8_TOTAL_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK8_DATA_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK8_DATA_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK9_TOTAL_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK9_TOTAL_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK9_DATA_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK9_DATA_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK10_TOTAL_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK10_TOTAL_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK10_DATA_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK10_DATA_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK11_TOTAL_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK11_TOTAL_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK11_DATA_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK11_DATA_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK12_TOTAL_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK12_TOTAL_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK12_DATA_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK12_DATA_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK13_TOTAL_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK13_TOTAL_RX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK13_DATA_TX_PER_SEC
        NVML_GPM_METRIC_C2C_LINK13_DATA_RX_PER_SEC
        NVML_GPM_METRIC_HOSTMEM_CACHE_HIT
        NVML_GPM_METRIC_HOSTMEM_CACHE_MISS
        NVML_GPM_METRIC_PEERMEM_CACHE_HIT
        NVML_GPM_METRIC_PEERMEM_CACHE_MISS
        NVML_GPM_METRIC_DRAM_CACHE_HIT
        NVML_GPM_METRIC_DRAM_CACHE_MISS
        NVML_GPM_METRIC_NVENC_0_UTIL
        NVML_GPM_METRIC_NVENC_1_UTIL
        NVML_GPM_METRIC_NVENC_2_UTIL
        NVML_GPM_METRIC_NVENC_3_UTIL
        NVML_GPM_METRIC_GR0_CTXSW_CYCLES_ELAPSED
        NVML_GPM_METRIC_GR0_CTXSW_CYCLES_ACTIVE
        NVML_GPM_METRIC_GR0_CTXSW_REQUESTS
        NVML_GPM_METRIC_GR0_CTXSW_CYCLES_PER_REQ
        NVML_GPM_METRIC_GR0_CTXSW_ACTIVE_PCT
        NVML_GPM_METRIC_GR1_CTXSW_CYCLES_ELAPSED
        NVML_GPM_METRIC_GR1_CTXSW_CYCLES_ACTIVE
        NVML_GPM_METRIC_GR1_CTXSW_REQUESTS
        NVML_GPM_METRIC_GR1_CTXSW_CYCLES_PER_REQ
        NVML_GPM_METRIC_GR1_CTXSW_ACTIVE_PCT
        NVML_GPM_METRIC_GR2_CTXSW_CYCLES_ELAPSED
        NVML_GPM_METRIC_GR2_CTXSW_CYCLES_ACTIVE
        NVML_GPM_METRIC_GR2_CTXSW_REQUESTS
        NVML_GPM_METRIC_GR2_CTXSW_CYCLES_PER_REQ
        NVML_GPM_METRIC_GR2_CTXSW_ACTIVE_PCT
        NVML_GPM_METRIC_GR3_CTXSW_CYCLES_ELAPSED
        NVML_GPM_METRIC_GR3_CTXSW_CYCLES_ACTIVE
        NVML_GPM_METRIC_GR3_CTXSW_REQUESTS
        NVML_GPM_METRIC_GR3_CTXSW_CYCLES_PER_REQ
        NVML_GPM_METRIC_GR3_CTXSW_ACTIVE_PCT
        NVML_GPM_METRIC_GR4_CTXSW_CYCLES_ELAPSED
        NVML_GPM_METRIC_GR4_CTXSW_CYCLES_ACTIVE
        NVML_GPM_METRIC_GR4_CTXSW_REQUESTS
        NVML_GPM_METRIC_GR4_CTXSW_CYCLES_PER_REQ
        NVML_GPM_METRIC_GR4_CTXSW_ACTIVE_PCT
        NVML_GPM_METRIC_GR5_CTXSW_CYCLES_ELAPSED
        NVML_GPM_METRIC_GR5_CTXSW_CYCLES_ACTIVE
        NVML_GPM_METRIC_GR5_CTXSW_REQUESTS
        NVML_GPM_METRIC_GR5_CTXSW_CYCLES_PER_REQ
        NVML_GPM_METRIC_GR5_CTXSW_ACTIVE_PCT
        NVML_GPM_METRIC_GR6_CTXSW_CYCLES_ELAPSED
        NVML_GPM_METRIC_GR6_CTXSW_CYCLES_ACTIVE
        NVML_GPM_METRIC_GR6_CTXSW_REQUESTS
        NVML_GPM_METRIC_GR6_CTXSW_CYCLES_PER_REQ
        NVML_GPM_METRIC_GR6_CTXSW_ACTIVE_PCT
        NVML_GPM_METRIC_GR7_CTXSW_CYCLES_ELAPSED
        NVML_GPM_METRIC_GR7_CTXSW_CYCLES_ACTIVE
        NVML_GPM_METRIC_GR7_CTXSW_REQUESTS
        NVML_GPM_METRIC_GR7_CTXSW_CYCLES_PER_REQ
        NVML_GPM_METRIC_GR7_CTXSW_ACTIVE_PCT
        NVML_GPM_METRIC_NVLINK_L18_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L18_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L19_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L19_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L20_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L20_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L21_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L21_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L22_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L22_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L23_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L23_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L24_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L24_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L25_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L25_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L26_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L26_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L27_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L27_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L28_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L28_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L29_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L29_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L30_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L30_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L31_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L31_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L32_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L32_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L33_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L33_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L34_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L34_TX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L35_RX_PER_SEC
        NVML_GPM_METRIC_NVLINK_L35_TX_PER_SEC
        NVML_GPM_METRIC_SM_CYCLES_ELAPSED
        NVML_GPM_METRIC_SM_CYCLES_ACTIVE
        NVML_GPM_METRIC_MMA_CYCLES_ACTIVE
        NVML_GPM_METRIC_DMMA_CYCLES_ACTIVE
        NVML_GPM_METRIC_HMMA_CYCLES_ACTIVE
        NVML_GPM_METRIC_IMMA_CYCLES_ACTIVE
        NVML_GPM_METRIC_DFMA_CYCLES_ACTIVE
        NVML_GPM_METRIC_PCIE_TX
        NVML_GPM_METRIC_PCIE_RX
        NVML_GPM_METRIC_INTEGER_CYCLES_ACTIVE
        NVML_GPM_METRIC_FP64_CYCLES_ACTIVE
        NVML_GPM_METRIC_FP32_CYCLES_ACTIVE
        NVML_GPM_METRIC_FP16_CYCLES_ACTIVE
        NVML_GPM_METRIC_NVLINK_L0_RX
        NVML_GPM_METRIC_NVLINK_L0_TX
        NVML_GPM_METRIC_NVLINK_L1_RX
        NVML_GPM_METRIC_NVLINK_L1_TX
        NVML_GPM_METRIC_NVLINK_L2_RX
        NVML_GPM_METRIC_NVLINK_L2_TX
        NVML_GPM_METRIC_NVLINK_L3_RX
        NVML_GPM_METRIC_NVLINK_L3_TX
        NVML_GPM_METRIC_NVLINK_L4_RX
        NVML_GPM_METRIC_NVLINK_L4_TX
        NVML_GPM_METRIC_NVLINK_L5_RX
        NVML_GPM_METRIC_NVLINK_L5_TX
        NVML_GPM_METRIC_NVLINK_L6_RX
        NVML_GPM_METRIC_NVLINK_L6_TX
        NVML_GPM_METRIC_NVLINK_L7_RX
        NVML_GPM_METRIC_NVLINK_L7_TX
        NVML_GPM_METRIC_NVLINK_L8_RX
        NVML_GPM_METRIC_NVLINK_L8_TX
        NVML_GPM_METRIC_NVLINK_L9_RX
        NVML_GPM_METRIC_NVLINK_L9_TX
        NVML_GPM_METRIC_NVLINK_L10_RX
        NVML_GPM_METRIC_NVLINK_L10_TX
        NVML_GPM_METRIC_NVLINK_L11_RX
        NVML_GPM_METRIC_NVLINK_L11_TX
        NVML_GPM_METRIC_NVLINK_L12_RX
        NVML_GPM_METRIC_NVLINK_L12_TX
        NVML_GPM_METRIC_NVLINK_L13_RX
        NVML_GPM_METRIC_NVLINK_L13_TX
        NVML_GPM_METRIC_NVLINK_L14_RX
        NVML_GPM_METRIC_NVLINK_L14_TX
        NVML_GPM_METRIC_NVLINK_L15_RX
        NVML_GPM_METRIC_NVLINK_L15_TX
        NVML_GPM_METRIC_NVLINK_L16_RX
        NVML_GPM_METRIC_NVLINK_L16_TX
        NVML_GPM_METRIC_NVLINK_L17_RX
        NVML_GPM_METRIC_NVLINK_L17_TX
        NVML_GPM_METRIC_NVLINK_L18_RX
        NVML_GPM_METRIC_NVLINK_L18_TX
        NVML_GPM_METRIC_NVLINK_L19_RX
        NVML_GPM_METRIC_NVLINK_L19_TX
        NVML_GPM_METRIC_NVLINK_L20_RX
        NVML_GPM_METRIC_NVLINK_L20_TX
        NVML_GPM_METRIC_NVLINK_L21_RX
        NVML_GPM_METRIC_NVLINK_L21_TX
        NVML_GPM_METRIC_NVLINK_L22_RX
        NVML_GPM_METRIC_NVLINK_L22_TX
        NVML_GPM_METRIC_NVLINK_L23_RX
        NVML_GPM_METRIC_NVLINK_L23_TX
        NVML_GPM_METRIC_NVLINK_L24_RX
        NVML_GPM_METRIC_NVLINK_L24_TX
        NVML_GPM_METRIC_NVLINK_L25_RX
        NVML_GPM_METRIC_NVLINK_L25_TX
        NVML_GPM_METRIC_NVLINK_L26_RX
        NVML_GPM_METRIC_NVLINK_L26_TX
        NVML_GPM_METRIC_NVLINK_L27_RX
        NVML_GPM_METRIC_NVLINK_L27_TX
        NVML_GPM_METRIC_NVLINK_L28_RX
        NVML_GPM_METRIC_NVLINK_L28_TX
        NVML_GPM_METRIC_NVLINK_L29_RX
        NVML_GPM_METRIC_NVLINK_L29_TX
        NVML_GPM_METRIC_NVLINK_L30_RX
        NVML_GPM_METRIC_NVLINK_L30_TX
        NVML_GPM_METRIC_NVLINK_L31_RX
        NVML_GPM_METRIC_NVLINK_L31_TX
        NVML_GPM_METRIC_NVLINK_L32_RX
        NVML_GPM_METRIC_NVLINK_L32_TX
        NVML_GPM_METRIC_NVLINK_L33_RX
        NVML_GPM_METRIC_NVLINK_L33_TX
        NVML_GPM_METRIC_NVLINK_L34_RX
        NVML_GPM_METRIC_NVLINK_L34_TX
        NVML_GPM_METRIC_NVLINK_L35_RX
        NVML_GPM_METRIC_NVLINK_L35_TX
        NVML_GPM_METRIC_MAX

cdef extern from 'nvml.h':
    ctypedef enum nvmlPowerProfileType_t:
        NVML_POWER_PROFILE_MAX_P
        NVML_POWER_PROFILE_MAX_Q
        NVML_POWER_PROFILE_COMPUTE
        NVML_POWER_PROFILE_MEMORY_BOUND
        NVML_POWER_PROFILE_NETWORK
        NVML_POWER_PROFILE_BALANCED
        NVML_POWER_PROFILE_LLM_INFERENCE
        NVML_POWER_PROFILE_LLM_TRAINING
        NVML_POWER_PROFILE_RBM
        NVML_POWER_PROFILE_DCPCIE
        NVML_POWER_PROFILE_HMMA_SPARSE
        NVML_POWER_PROFILE_HMMA_DENSE
        NVML_POWER_PROFILE_SYNC_BALANCED
        NVML_POWER_PROFILE_HPC
        NVML_POWER_PROFILE_MIG
        NVML_POWER_PROFILE_MAX

cdef extern from 'nvml.h':
    ctypedef enum nvmlDeviceAddressingModeType_t:
        NVML_DEVICE_ADDRESSING_MODE_NONE
        NVML_DEVICE_ADDRESSING_MODE_HMM
        NVML_DEVICE_ADDRESSING_MODE_ATS

cdef extern from 'nvml.h':
    ctypedef enum nvmlPRMCounterId_t:
        NVML_PRM_COUNTER_ID_NONE
        NVML_PRM_COUNTER_ID_PPCNT_PHYSICAL_LAYER_CTRS_LINK_DOWN_EVENTS
        NVML_PRM_COUNTER_ID_PPCNT_PHYSICAL_LAYER_CTRS_SUCCESSFUL_RECOVERY_EVENTS
        NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TOTAL_SUCCESSFUL_RECOVERY_EVENTS
        NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TIME_SINCE_LAST_RECOVERY
        NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TIME_BETWEEN_LAST_TWO_RECOVERIES
        NVML_PRM_COUNTER_ID_PPCNT_PORTCOUNTERS_PORT_XMIT_WAIT
        NVML_PRM_COUNTER_ID_PPCNT_PLR_RCV_CODES
        NVML_PRM_COUNTER_ID_PPCNT_PLR_RCV_CODE_ERR
        NVML_PRM_COUNTER_ID_PPCNT_PLR_RCV_UNCORRECTABLE_CODE
        NVML_PRM_COUNTER_ID_PPCNT_PLR_XMIT_CODES
        NVML_PRM_COUNTER_ID_PPCNT_PLR_XMIT_RETRY_CODES
        NVML_PRM_COUNTER_ID_PPCNT_PLR_XMIT_RETRY_EVENTS
        NVML_PRM_COUNTER_ID_PPCNT_PLR_SYNC_EVENTS
        NVML_PRM_COUNTER_ID_PPRM_OPER_RECOVERY

cdef extern from 'nvml.h':
    ctypedef enum nvmlPowerProfileOperation_t:
        NVML_POWER_PROFILE_OPERATION_CLEAR
        NVML_POWER_PROFILE_OPERATION_SET
        NVML_POWER_PROFILE_OPERATION_SET_AND_OVERWRITE
        NVML_POWER_PROFILE_OPERATION_MAX

cdef extern from 'nvml.h':
    ctypedef enum nvmlProcessMode_t:
        NVML_PROCESS_MODE_COMPUTE
        NVML_PROCESS_MODE_GRAPHICS
        NVML_PROCESS_MODE_MPS
        NVML_PROCESS_MODE_ALL
        NVML_PROCESS_MODE_MAX

cdef extern from 'nvml.h':
    ctypedef enum nvmlCPERType_t:
        NVML_CPER_ACCESS_TYPE_GPU
cdef enum: _NVMLRETURN_T_INTERNAL_LOADING_ERROR = -42


# types
cdef extern from 'nvml.h':
    ctypedef struct nvmlPciInfoExt_v1_t 'nvmlPciInfoExt_v1_t':
        unsigned int version
        unsigned int domain
        unsigned int bus
        unsigned int device
        unsigned int pciDeviceId
        unsigned int pciSubSystemId
        unsigned int baseClass
        unsigned int subClass
        char busId[32]

cdef extern from 'nvml.h':
    ctypedef struct nvmlCoolerInfo_v1_t 'nvmlCoolerInfo_v1_t':
        unsigned int version
        unsigned int index
        nvmlCoolerControl_t signalType
        nvmlCoolerTarget_t target

cdef extern from 'nvml.h':
    ctypedef struct nvmlDramEncryptionInfo_v1_t 'nvmlDramEncryptionInfo_v1_t':
        unsigned int version
        nvmlEnableState_t encryptionState

cdef extern from 'nvml.h':
    ctypedef struct nvmlMarginTemperature_v1_t 'nvmlMarginTemperature_v1_t':
        unsigned int version
        int marginTemperature

cdef extern from 'nvml.h':
    ctypedef struct nvmlClockOffset_v1_t 'nvmlClockOffset_v1_t':
        unsigned int version
        nvmlClockType_t type
        nvmlPstates_t pstate
        int clockOffsetMHz
        int minClockOffsetMHz
        int maxClockOffsetMHz

cdef extern from 'nvml.h':
    ctypedef struct nvmlFanSpeedInfo_v1_t 'nvmlFanSpeedInfo_v1_t':
        unsigned int version
        unsigned int fan
        unsigned int speed

cdef extern from 'nvml.h':
    ctypedef struct nvmlDevicePerfModes_v1_t 'nvmlDevicePerfModes_v1_t':
        unsigned int version
        char str[2048]

cdef extern from 'nvml.h':
    ctypedef struct nvmlDeviceCurrentClockFreqs_v1_t 'nvmlDeviceCurrentClockFreqs_v1_t':
        unsigned int version
        char str[2048]

cdef extern from 'nvml.h':
    ctypedef struct nvmlEccSramErrorStatus_v1_t 'nvmlEccSramErrorStatus_v1_t':
        unsigned int version
        unsigned long long aggregateUncParity
        unsigned long long aggregateUncSecDed
        unsigned long long aggregateCor
        unsigned long long volatileUncParity
        unsigned long long volatileUncSecDed
        unsigned long long volatileCor
        unsigned long long aggregateUncBucketL2
        unsigned long long aggregateUncBucketSm
        unsigned long long aggregateUncBucketPcie
        unsigned long long aggregateUncBucketMcu
        unsigned long long aggregateUncBucketOther
        unsigned int bThresholdExceeded

cdef extern from 'nvml.h':
    ctypedef struct nvmlPlatformInfo_v2_t 'nvmlPlatformInfo_v2_t':
        unsigned int version
        unsigned char ibGuid[16]
        unsigned char chassisSerialNumber[16]
        unsigned char slotNumber
        unsigned char trayIndex
        unsigned char hostId
        unsigned char peerType
        unsigned char moduleId

cdef extern from 'nvml.h':
    ctypedef unsigned int nvmlDeviceArchitecture_t 'nvmlDeviceArchitecture_t'


cdef extern from 'nvml.h':
    ctypedef unsigned int nvmlBusType_t 'nvmlBusType_t'


cdef extern from 'nvml.h':
    ctypedef unsigned int nvmlFanControlPolicy_t 'nvmlFanControlPolicy_t'


cdef extern from 'nvml.h':
    ctypedef unsigned int nvmlPowerSource_t 'nvmlPowerSource_t'


cdef extern from 'nvml.h':
    ctypedef unsigned char nvmlPowerScopeType_t 'nvmlPowerScopeType_t'


cdef extern from 'nvml.h':
    ctypedef unsigned int nvmlVgpuTypeId_t 'nvmlVgpuTypeId_t'


cdef extern from 'nvml.h':
    ctypedef unsigned int nvmlVgpuInstance_t 'nvmlVgpuInstance_t'


cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuHeterogeneousMode_v1_t 'nvmlVgpuHeterogeneousMode_v1_t':
        unsigned int version
        unsigned int mode

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuPlacementId_v1_t 'nvmlVgpuPlacementId_v1_t':
        unsigned int version
        unsigned int placementId

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuPlacementList_v2_t 'nvmlVgpuPlacementList_v2_t':
        unsigned int version
        unsigned int placementSize
        unsigned int count
        unsigned int* placementIds
        unsigned int mode

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuTypeBar1Info_v1_t 'nvmlVgpuTypeBar1Info_v1_t':
        unsigned int version
        unsigned long long bar1Size

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuRuntimeState_v1_t 'nvmlVgpuRuntimeState_v1_t':
        unsigned int version
        unsigned long long size

cdef extern from 'nvml.h':
    ctypedef struct nvmlSystemConfComputeSettings_v1_t 'nvmlSystemConfComputeSettings_v1_t':
        unsigned int version
        unsigned int environment
        unsigned int ccFeature
        unsigned int devToolsMode
        unsigned int multiGpuMode

cdef extern from 'nvml.h':
    ctypedef struct nvmlConfComputeSetKeyRotationThresholdInfo_v1_t 'nvmlConfComputeSetKeyRotationThresholdInfo_v1_t':
        unsigned int version
        unsigned long long maxAttackerAdvantage

cdef extern from 'nvml.h':
    ctypedef struct nvmlConfComputeGetKeyRotationThresholdInfo_v1_t 'nvmlConfComputeGetKeyRotationThresholdInfo_v1_t':
        unsigned int version
        unsigned long long attackerAdvantage

cdef extern from 'nvml.h':
    ctypedef unsigned char nvmlGpuFabricState_t 'nvmlGpuFabricState_t'


cdef extern from 'nvml.h':
    ctypedef struct nvmlSystemDriverBranchInfo_v1_t 'nvmlSystemDriverBranchInfo_v1_t':
        unsigned int version
        char branch[80]

cdef extern from 'nvml.h':
    ctypedef unsigned int nvmlAffinityScope_t 'nvmlAffinityScope_t'


cdef extern from 'nvml.h':
    ctypedef struct nvmlTemperature_v1_t 'nvmlTemperature_v1_t':
        unsigned int version
        nvmlTemperatureSensors_t sensorType
        int temperature

cdef extern from 'nvml.h':
    ctypedef struct nvmlNvlinkSupportedBwModes_v1_t 'nvmlNvlinkSupportedBwModes_v1_t':
        unsigned int version
        unsigned char bwModes[23]
        unsigned char totalBwModes

cdef extern from 'nvml.h':
    ctypedef struct nvmlNvlinkGetBwMode_v1_t 'nvmlNvlinkGetBwMode_v1_t':
        unsigned int version
        unsigned int bIsBest
        unsigned char bwMode

cdef extern from 'nvml.h':
    ctypedef struct nvmlNvlinkSetBwMode_v1_t 'nvmlNvlinkSetBwMode_v1_t':
        unsigned int version
        unsigned int bSetBest
        unsigned char bwMode

cdef extern from 'nvml.h':
    ctypedef struct nvmlDeviceCapabilities_v1_t 'nvmlDeviceCapabilities_v1_t':
        unsigned int version
        unsigned int capMask

cdef extern from 'nvml.h':
    ctypedef struct nvmlPowerSmoothingProfile_v1_t 'nvmlPowerSmoothingProfile_v1_t':
        unsigned int version
        unsigned int profileId
        unsigned int paramId
        double value

cdef extern from 'nvml.h':
    ctypedef struct nvmlPowerSmoothingState_v1_t 'nvmlPowerSmoothingState_v1_t':
        unsigned int version
        nvmlEnableState_t state

cdef extern from 'nvml.h':
    ctypedef struct nvmlDeviceAddressingMode_v1_t 'nvmlDeviceAddressingMode_v1_t':
        unsigned int version
        unsigned int value

cdef extern from 'nvml.h':
    ctypedef struct nvmlRepairStatus_v1_t 'nvmlRepairStatus_v1_t':
        unsigned int version
        unsigned int bChannelRepairPending
        unsigned int bTpcRepairPending

cdef extern from 'nvml.h':
    ctypedef struct nvmlPdi_v1_t 'nvmlPdi_v1_t':
        unsigned int version
        unsigned long long value

cdef extern from 'nvml.h':
    ctypedef unsigned long long nvmlCPERCursorHandle_t 'nvmlCPERCursorHandle_t'


cdef extern from 'nvml.h':
    ctypedef void* nvmlDevice_t 'nvmlDevice_t'


cdef extern from 'nvml.h':
    ctypedef void* nvmlGpuInstance_t 'nvmlGpuInstance_t'


cdef extern from 'nvml.h':
    ctypedef void* nvmlUnit_t 'nvmlUnit_t'


cdef extern from 'nvml.h':
    ctypedef void* nvmlEventSet_t 'nvmlEventSet_t'


cdef extern from 'nvml.h':
    ctypedef void* nvmlSystemEventSet_t 'nvmlSystemEventSet_t'


cdef extern from 'nvml.h':
    ctypedef void* nvmlComputeInstance_t 'nvmlComputeInstance_t'


cdef extern from 'nvml.h':
    ctypedef void* nvmlGpmSample_t 'nvmlGpmSample_t'


cdef extern from 'nvml.h':
    ctypedef struct nvmlPciInfo_t 'nvmlPciInfo_t':
        char busIdLegacy[16]
        unsigned int domain
        unsigned int bus
        unsigned int device
        unsigned int pciDeviceId
        unsigned int pciSubSystemId
        char busId[32]

cdef extern from 'nvml.h':
    ctypedef struct nvmlEccErrorCounts_t 'nvmlEccErrorCounts_t':
        unsigned long long l1Cache
        unsigned long long l2Cache
        unsigned long long deviceMemory
        unsigned long long registerFile

cdef extern from 'nvml.h':
    ctypedef struct nvmlUtilization_t 'nvmlUtilization_t':
        unsigned int gpu
        unsigned int memory

cdef extern from 'nvml.h':
    ctypedef struct nvmlMemory_t 'nvmlMemory_t':
        unsigned long long total
        unsigned long long free
        unsigned long long used

cdef extern from 'nvml.h':
    ctypedef struct nvmlMemory_v2_t 'nvmlMemory_v2_t':
        unsigned int version
        unsigned long long total
        unsigned long long reserved
        unsigned long long free
        unsigned long long used

cdef extern from 'nvml.h':
    ctypedef struct nvmlBAR1Memory_t 'nvmlBAR1Memory_t':
        unsigned long long bar1Total
        unsigned long long bar1Free
        unsigned long long bar1Used

cdef extern from 'nvml.h':
    ctypedef struct nvmlProcessInfo_v1_t 'nvmlProcessInfo_v1_t':
        unsigned int pid
        unsigned long long usedGpuMemory

cdef extern from 'nvml.h':
    ctypedef struct nvmlProcessInfo_v2_t 'nvmlProcessInfo_v2_t':
        unsigned int pid
        unsigned long long usedGpuMemory
        unsigned int gpuInstanceId
        unsigned int computeInstanceId

cdef extern from 'nvml.h':
    ctypedef struct nvmlProcessInfo_t 'nvmlProcessInfo_t':
        unsigned int pid
        unsigned long long usedGpuMemory
        unsigned int gpuInstanceId
        unsigned int computeInstanceId

cdef extern from 'nvml.h':
    ctypedef struct nvmlProcessDetail_v1_t 'nvmlProcessDetail_v1_t':
        unsigned int pid
        unsigned long long usedGpuMemory
        unsigned int gpuInstanceId
        unsigned int computeInstanceId
        unsigned long long usedGpuCcProtectedMemory

cdef extern from 'nvml.h':
    ctypedef struct nvmlDeviceAttributes_t 'nvmlDeviceAttributes_t':
        unsigned int multiprocessorCount
        unsigned int sharedCopyEngineCount
        unsigned int sharedDecoderCount
        unsigned int sharedEncoderCount
        unsigned int sharedJpegCount
        unsigned int sharedOfaCount
        unsigned int gpuInstanceSliceCount
        unsigned int computeInstanceSliceCount
        unsigned long long memorySizeMB

cdef extern from 'nvml.h':
    ctypedef struct nvmlC2cModeInfo_v1_t 'nvmlC2cModeInfo_v1_t':
        unsigned int isC2cEnabled

cdef extern from 'nvml.h':
    ctypedef struct nvmlRowRemapperHistogramValues_t 'nvmlRowRemapperHistogramValues_t':
        unsigned int max
        unsigned int high
        unsigned int partial
        unsigned int low
        unsigned int none

cdef extern from 'nvml.h':
    ctypedef struct nvmlNvLinkUtilizationControl_t 'nvmlNvLinkUtilizationControl_t':
        nvmlNvLinkUtilizationCountUnits_t units
        nvmlNvLinkUtilizationCountPktTypes_t pktfilter

cdef extern from 'nvml.h':
    ctypedef struct nvmlBridgeChipInfo_t 'nvmlBridgeChipInfo_t':
        nvmlBridgeChipType_t type
        unsigned int fwVersion

cdef extern from 'nvml.h':
    ctypedef union nvmlValue_t 'nvmlValue_t':
        double dVal
        int siVal
        unsigned int uiVal
        unsigned long ulVal
        unsigned long long ullVal
        signed long long sllVal
        unsigned short usVal

cdef extern from 'nvml.h':
    ctypedef struct nvmlViolationTime_t 'nvmlViolationTime_t':
        unsigned long long referenceTime
        unsigned long long violationTime

ctypedef struct cuda_bindings_nvml__anon_pod0:
    nvmlThermalController_t controller
    int defaultMinTemp
    int defaultMaxTemp
    int currentTemp
    nvmlThermalTarget_t target

cdef extern from 'nvml.h':
    ctypedef union nvmlUUIDValue_t 'nvmlUUIDValue_t':
        char str[41]
        unsigned char bytes[16]

cdef extern from 'nvml.h':
    ctypedef struct nvmlClkMonFaultInfo_t 'nvmlClkMonFaultInfo_t':
        unsigned int clkApiDomain
        unsigned int clkDomainFaultMask

cdef extern from 'nvml.h':
    ctypedef struct nvmlProcessUtilizationSample_t 'nvmlProcessUtilizationSample_t':
        unsigned int pid
        unsigned long long timeStamp
        unsigned int smUtil
        unsigned int memUtil
        unsigned int encUtil
        unsigned int decUtil

cdef extern from 'nvml.h':
    ctypedef struct nvmlProcessUtilizationInfo_v1_t 'nvmlProcessUtilizationInfo_v1_t':
        unsigned long long timeStamp
        unsigned int pid
        unsigned int smUtil
        unsigned int memUtil
        unsigned int encUtil
        unsigned int decUtil
        unsigned int jpgUtil
        unsigned int ofaUtil

cdef extern from 'nvml.h':
    ctypedef struct nvmlPlatformInfo_v1_t 'nvmlPlatformInfo_v1_t':
        unsigned int version
        unsigned char ibGuid[16]
        unsigned char rackGuid[16]
        unsigned char chassisPhysicalSlotNumber
        unsigned char computeSlotIndex
        unsigned char nodeIndex
        unsigned char peerType
        unsigned char moduleId

ctypedef struct cuda_bindings_nvml__anon_pod1:
    unsigned int bIsPresent
    unsigned int percentage
    unsigned int incThreshold
    unsigned int decThreshold

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuPlacementList_v1_t 'nvmlVgpuPlacementList_v1_t':
        unsigned int version
        unsigned int placementSize
        unsigned int count
        unsigned int* placementIds

ctypedef struct cuda_bindings_nvml__anon_pod2:
    unsigned int avgFactor
    unsigned int timeslice

ctypedef struct cuda_bindings_nvml__anon_pod3:
    unsigned int timeslice

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuSchedulerLogEntry_t 'nvmlVgpuSchedulerLogEntry_t':
        unsigned long long timestamp
        unsigned long long timeRunTotal
        unsigned long long timeRun
        unsigned int swRunlistId
        unsigned long long targetTimeSlice
        unsigned long long cumulativePreemptionTime

ctypedef struct cuda_bindings_nvml__anon_pod4:
    unsigned int avgFactor
    unsigned int frequency

ctypedef struct cuda_bindings_nvml__anon_pod5:
    unsigned int timeslice

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuSchedulerCapabilities_t 'nvmlVgpuSchedulerCapabilities_t':
        unsigned int supportedSchedulers[3]
        unsigned int maxTimeslice
        unsigned int minTimeslice
        unsigned int isArrModeSupported
        unsigned int maxFrequencyForARR
        unsigned int minFrequencyForARR
        unsigned int maxAvgFactorForARR
        unsigned int minAvgFactorForARR

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuLicenseExpiry_t 'nvmlVgpuLicenseExpiry_t':
        unsigned int year
        unsigned short month
        unsigned short day
        unsigned short hour
        unsigned short min
        unsigned short sec
        unsigned char status

cdef extern from 'nvml.h':
    ctypedef struct nvmlGridLicenseExpiry_t 'nvmlGridLicenseExpiry_t':
        unsigned int year
        unsigned short month
        unsigned short day
        unsigned short hour
        unsigned short min
        unsigned short sec
        unsigned char status

cdef extern from 'nvml.h':
    ctypedef struct nvmlNvLinkPowerThres_t 'nvmlNvLinkPowerThres_t':
        unsigned int lowPwrThreshold

cdef extern from 'nvml.h':
    ctypedef struct nvmlHwbcEntry_t 'nvmlHwbcEntry_t':
        unsigned int hwbcId
        char firmwareVersion[32]

cdef extern from 'nvml.h':
    ctypedef struct nvmlLedState_t 'nvmlLedState_t':
        char cause[256]
        nvmlLedColor_t color

cdef extern from 'nvml.h':
    ctypedef struct nvmlUnitInfo_t 'nvmlUnitInfo_t':
        char name[96]
        char id[96]
        char serial[96]
        char firmwareVersion[96]

cdef extern from 'nvml.h':
    ctypedef struct nvmlPSUInfo_t 'nvmlPSUInfo_t':
        char state[256]
        unsigned int current
        unsigned int voltage
        unsigned int power

cdef extern from 'nvml.h':
    ctypedef struct nvmlUnitFanInfo_t 'nvmlUnitFanInfo_t':
        unsigned int speed
        nvmlFanState_t state

cdef extern from 'nvml.h':
    ctypedef struct nvmlSystemEventData_v1_t 'nvmlSystemEventData_v1_t':
        unsigned long long eventType
        unsigned int gpuId

cdef extern from 'nvml.h':
    ctypedef struct nvmlAccountingStats_t 'nvmlAccountingStats_t':
        unsigned int gpuUtilization
        unsigned int memoryUtilization
        unsigned long long maxMemoryUsage
        unsigned long long time
        unsigned long long startTime
        unsigned int isRunning
        unsigned int reserved[5]

cdef extern from 'nvml.h':
    ctypedef struct nvmlFBCStats_t 'nvmlFBCStats_t':
        unsigned int sessionsCount
        unsigned int averageFPS
        unsigned int averageLatency

cdef extern from 'nvml.h':
    ctypedef struct nvmlConfComputeSystemCaps_t 'nvmlConfComputeSystemCaps_t':
        unsigned int cpuCaps
        unsigned int gpusCaps

cdef extern from 'nvml.h':
    ctypedef struct nvmlConfComputeSystemState_t 'nvmlConfComputeSystemState_t':
        unsigned int environment
        unsigned int ccFeature
        unsigned int devToolsMode

cdef extern from 'nvml.h':
    ctypedef struct nvmlConfComputeMemSizeInfo_t 'nvmlConfComputeMemSizeInfo_t':
        unsigned long long protectedMemSizeKib
        unsigned long long unprotectedMemSizeKib

cdef extern from 'nvml.h':
    ctypedef struct nvmlConfComputeGpuCertificate_t 'nvmlConfComputeGpuCertificate_t':
        unsigned int certChainSize
        unsigned int attestationCertChainSize
        unsigned char certChain[0x1000]
        unsigned char attestationCertChain[0x1400]

cdef extern from 'nvml.h':
    ctypedef struct nvmlConfComputeGpuAttestationReport_t 'nvmlConfComputeGpuAttestationReport_t':
        unsigned int isCecAttestationReportPresent
        unsigned int attestationReportSize
        unsigned int cecAttestationReportSize
        unsigned char nonce[0x20]
        unsigned char attestationReport[0x2000]
        unsigned char cecAttestationReport[0x1000]

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuVersion_t 'nvmlVgpuVersion_t':
        unsigned int minVersion
        unsigned int maxVersion

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuMetadata_t 'nvmlVgpuMetadata_t':
        unsigned int version
        unsigned int revision
        nvmlVgpuGuestInfoState_t guestInfoState
        char guestDriverVersion[80]
        char hostDriverVersion[80]
        unsigned int reserved[6]
        unsigned int vgpuVirtualizationCaps
        unsigned int guestVgpuVersion
        unsigned int opaqueDataSize
        char opaqueData[4]

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuPgpuCompatibility_t 'nvmlVgpuPgpuCompatibility_t':
        nvmlVgpuVmCompatibility_t vgpuVmCompatibility
        nvmlVgpuPgpuCompatibilityLimitCode_t compatibilityLimitCode

cdef extern from 'nvml.h':
    ctypedef struct nvmlGpuInstancePlacement_t 'nvmlGpuInstancePlacement_t':
        unsigned int start
        unsigned int size

cdef extern from 'nvml.h':
    ctypedef struct nvmlGpuInstanceProfileInfo_t 'nvmlGpuInstanceProfileInfo_t':
        unsigned int id
        unsigned int isP2pSupported
        unsigned int sliceCount
        unsigned int instanceCount
        unsigned int multiprocessorCount
        unsigned int copyEngineCount
        unsigned int decoderCount
        unsigned int encoderCount
        unsigned int jpegCount
        unsigned int ofaCount
        unsigned long long memorySizeMB

cdef extern from 'nvml.h':
    ctypedef struct nvmlGpuInstanceProfileInfo_v2_t 'nvmlGpuInstanceProfileInfo_v2_t':
        unsigned int version
        unsigned int id
        unsigned int isP2pSupported
        unsigned int sliceCount
        unsigned int instanceCount
        unsigned int multiprocessorCount
        unsigned int copyEngineCount
        unsigned int decoderCount
        unsigned int encoderCount
        unsigned int jpegCount
        unsigned int ofaCount
        unsigned long long memorySizeMB
        char name[96]

cdef extern from 'nvml.h':
    ctypedef struct nvmlGpuInstanceProfileInfo_v3_t 'nvmlGpuInstanceProfileInfo_v3_t':
        unsigned int version
        unsigned int id
        unsigned int sliceCount
        unsigned int instanceCount
        unsigned int multiprocessorCount
        unsigned int copyEngineCount
        unsigned int decoderCount
        unsigned int encoderCount
        unsigned int jpegCount
        unsigned int ofaCount
        unsigned long long memorySizeMB
        char name[96]
        unsigned int capabilities

cdef extern from 'nvml.h':
    ctypedef struct nvmlComputeInstancePlacement_t 'nvmlComputeInstancePlacement_t':
        unsigned int start
        unsigned int size

cdef extern from 'nvml.h':
    ctypedef struct nvmlComputeInstanceProfileInfo_t 'nvmlComputeInstanceProfileInfo_t':
        unsigned int id
        unsigned int sliceCount
        unsigned int instanceCount
        unsigned int multiprocessorCount
        unsigned int sharedCopyEngineCount
        unsigned int sharedDecoderCount
        unsigned int sharedEncoderCount
        unsigned int sharedJpegCount
        unsigned int sharedOfaCount

cdef extern from 'nvml.h':
    ctypedef struct nvmlComputeInstanceProfileInfo_v2_t 'nvmlComputeInstanceProfileInfo_v2_t':
        unsigned int version
        unsigned int id
        unsigned int sliceCount
        unsigned int instanceCount
        unsigned int multiprocessorCount
        unsigned int sharedCopyEngineCount
        unsigned int sharedDecoderCount
        unsigned int sharedEncoderCount
        unsigned int sharedJpegCount
        unsigned int sharedOfaCount
        char name[96]

cdef extern from 'nvml.h':
    ctypedef struct nvmlComputeInstanceProfileInfo_v3_t 'nvmlComputeInstanceProfileInfo_v3_t':
        unsigned int version
        unsigned int id
        unsigned int sliceCount
        unsigned int instanceCount
        unsigned int multiprocessorCount
        unsigned int sharedCopyEngineCount
        unsigned int sharedDecoderCount
        unsigned int sharedEncoderCount
        unsigned int sharedJpegCount
        unsigned int sharedOfaCount
        char name[96]
        unsigned int capabilities

ctypedef struct cuda_bindings_nvml__anon_pod6:
    char* shortName
    char* longName
    char* unit

cdef extern from 'nvml.h':
    ctypedef struct nvmlGpmSupport_t 'nvmlGpmSupport_t':
        unsigned int version
        unsigned int isSupportedDevice

cdef extern from 'nvml.h':
    ctypedef struct nvmlMask255_t 'nvmlMask255_t':
        unsigned int mask[8]

cdef extern from 'nvml.h':
    ctypedef struct nvmlDevicePowerMizerModes_v1_t 'nvmlDevicePowerMizerModes_v1_t':
        unsigned int currentMode
        unsigned int mode
        unsigned int supportedPowerMizerModes

cdef extern from 'nvml.h':
    ctypedef struct nvmlHostname_v1_t 'nvmlHostname_v1_t':
        char value[64]

cdef extern from 'nvml.h':
    ctypedef struct nvmlEccSramUniqueUncorrectedErrorEntry_v1_t 'nvmlEccSramUniqueUncorrectedErrorEntry_v1_t':
        unsigned int unit
        unsigned int location
        unsigned int sublocation
        unsigned int extlocation
        unsigned int address
        unsigned int isParity
        unsigned int count

cdef extern from 'nvml.h':
    ctypedef struct nvmlNvLinkInfo_v1_t 'nvmlNvLinkInfo_v1_t':
        unsigned int version
        unsigned int isNvleEnabled

cdef extern from 'nvml.h':
    ctypedef struct nvmlNvlinkFirmwareVersion_t 'nvmlNvlinkFirmwareVersion_t':
        unsigned char ucodeType
        unsigned int major
        unsigned int minor
        unsigned int subMinor

ctypedef union cuda_bindings_nvml__anon_pod7:
    unsigned char inData[496]
    unsigned char outData[496]

cdef extern from 'nvml.h':
    ctypedef struct nvmlUnrepairableMemoryStatus_v1_t 'nvmlUnrepairableMemoryStatus_v1_t':
        unsigned int bUnrepairableMemory

cdef extern from 'nvml.h':
    ctypedef struct nvmlRusdSettings_v1_t 'nvmlRusdSettings_v1_t':
        unsigned int version
        unsigned long long pollMask

cdef extern from 'nvml.h':
    ctypedef struct nvmlPRMCounterInput_v1_t 'nvmlPRMCounterInput_v1_t':
        unsigned int localPort

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuSchedulerStateInfo_v2_t 'nvmlVgpuSchedulerStateInfo_v2_t':
        unsigned int engineId
        unsigned int schedulerPolicy
        unsigned int avgFactor
        unsigned int timeslice

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuSchedulerLogEntry_v2_t 'nvmlVgpuSchedulerLogEntry_v2_t':
        unsigned long long timestamp
        unsigned long long timeRunTotal
        unsigned long long timeRun
        unsigned int swRunlistId
        unsigned long long targetTimeSlice
        unsigned long long cumulativePreemptionTime
        unsigned int weight

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuSchedulerState_v2_t 'nvmlVgpuSchedulerState_v2_t':
        unsigned int engineId
        unsigned int schedulerPolicy
        unsigned int avgFactor
        unsigned int frequency

cdef extern from 'nvml.h':
    ctypedef struct nvmlBBXTimeData_v1_t 'nvmlBBXTimeData_v1_t':
        unsigned int timeRun

cdef extern from 'nvml.h':
    ctypedef struct nvmlRemappedRowsInfo_v2_t 'nvmlRemappedRowsInfo_v2_t':
        unsigned int corrActiveRemaps
        unsigned int corrInactiveRemaps
        unsigned int uncActiveRemaps
        unsigned int uncInactiveRemaps
        unsigned int bPending
        unsigned int bFailureOccurred

cdef extern from 'nvml.h':
    ctypedef struct nvmlAccountingStats_v2_t 'nvmlAccountingStats_v2_t':
        unsigned int pid
        unsigned int isRunning
        unsigned int gpuUtilization
        unsigned int memoryUtilization
        unsigned long long maxMemoryUsage
        unsigned int sampleCount
        unsigned long long sumGpuUtil
        unsigned long long sumFbUtil
        unsigned long long time
        unsigned long long startTime

cdef extern from 'nvml.h':
    ctypedef nvmlPciInfoExt_v1_t nvmlPciInfoExt_t 'nvmlPciInfoExt_t'


cdef extern from 'nvml.h':
    ctypedef nvmlCoolerInfo_v1_t nvmlCoolerInfo_t 'nvmlCoolerInfo_t'


cdef extern from 'nvml.h':
    ctypedef nvmlDramEncryptionInfo_v1_t nvmlDramEncryptionInfo_t 'nvmlDramEncryptionInfo_t'


cdef extern from 'nvml.h':
    ctypedef nvmlMarginTemperature_v1_t nvmlMarginTemperature_t 'nvmlMarginTemperature_t'


cdef extern from 'nvml.h':
    ctypedef nvmlClockOffset_v1_t nvmlClockOffset_t 'nvmlClockOffset_t'


cdef extern from 'nvml.h':
    ctypedef nvmlFanSpeedInfo_v1_t nvmlFanSpeedInfo_t 'nvmlFanSpeedInfo_t'


cdef extern from 'nvml.h':
    ctypedef nvmlDevicePerfModes_v1_t nvmlDevicePerfModes_t 'nvmlDevicePerfModes_t'


cdef extern from 'nvml.h':
    ctypedef nvmlDeviceCurrentClockFreqs_v1_t nvmlDeviceCurrentClockFreqs_t 'nvmlDeviceCurrentClockFreqs_t'


cdef extern from 'nvml.h':
    ctypedef nvmlEccSramErrorStatus_v1_t nvmlEccSramErrorStatus_t 'nvmlEccSramErrorStatus_t'


cdef extern from 'nvml.h':
    ctypedef nvmlPlatformInfo_v2_t nvmlPlatformInfo_t 'nvmlPlatformInfo_t'


cdef extern from 'nvml.h':
    ctypedef struct nvmlPowerValue_v2_t 'nvmlPowerValue_v2_t':
        unsigned int version
        nvmlPowerScopeType_t powerScope
        unsigned int powerValueMw

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuTypeIdInfo_v1_t 'nvmlVgpuTypeIdInfo_v1_t':
        unsigned int version
        unsigned int vgpuCount
        nvmlVgpuTypeId_t* vgpuTypeIds

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuTypeMaxInstance_v1_t 'nvmlVgpuTypeMaxInstance_v1_t':
        unsigned int version
        nvmlVgpuTypeId_t vgpuTypeId
        unsigned int maxInstancePerGI

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuCreatablePlacementInfo_v1_t 'nvmlVgpuCreatablePlacementInfo_v1_t':
        unsigned int version
        nvmlVgpuTypeId_t vgpuTypeId
        unsigned int count
        unsigned int* placementIds
        unsigned int placementSize

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuProcessUtilizationSample_t 'nvmlVgpuProcessUtilizationSample_t':
        nvmlVgpuInstance_t vgpuInstance
        unsigned int pid
        char processName[64]
        unsigned long long timeStamp
        unsigned int smUtil
        unsigned int memUtil
        unsigned int encUtil
        unsigned int decUtil

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuProcessUtilizationInfo_v1_t 'nvmlVgpuProcessUtilizationInfo_v1_t':
        char processName[64]
        unsigned long long timeStamp
        nvmlVgpuInstance_t vgpuInstance
        unsigned int pid
        unsigned int smUtil
        unsigned int memUtil
        unsigned int encUtil
        unsigned int decUtil
        unsigned int jpgUtil
        unsigned int ofaUtil

cdef extern from 'nvml.h':
    ctypedef struct nvmlActiveVgpuInstanceInfo_v1_t 'nvmlActiveVgpuInstanceInfo_v1_t':
        unsigned int version
        unsigned int vgpuCount
        nvmlVgpuInstance_t* vgpuInstances

cdef extern from 'nvml.h':
    ctypedef struct nvmlEncoderSessionInfo_t 'nvmlEncoderSessionInfo_t':
        unsigned int sessionId
        unsigned int pid
        nvmlVgpuInstance_t vgpuInstance
        nvmlEncoderType_t codecType
        unsigned int hResolution
        unsigned int vResolution
        unsigned int averageFps
        unsigned int averageLatency

cdef extern from 'nvml.h':
    ctypedef struct nvmlFBCSessionInfo_t 'nvmlFBCSessionInfo_t':
        unsigned int sessionId
        unsigned int pid
        nvmlVgpuInstance_t vgpuInstance
        unsigned int displayOrdinal
        nvmlFBCSessionType_t sessionType
        unsigned int sessionFlags
        unsigned int hMaxResolution
        unsigned int vMaxResolution
        unsigned int hResolution
        unsigned int vResolution
        unsigned int averageFPS
        unsigned int averageLatency

cdef extern from 'nvml.h':
    ctypedef nvmlVgpuHeterogeneousMode_v1_t nvmlVgpuHeterogeneousMode_t 'nvmlVgpuHeterogeneousMode_t'


cdef extern from 'nvml.h':
    ctypedef nvmlVgpuPlacementId_v1_t nvmlVgpuPlacementId_t 'nvmlVgpuPlacementId_t'


cdef extern from 'nvml.h':
    ctypedef nvmlVgpuPlacementList_v2_t nvmlVgpuPlacementList_t 'nvmlVgpuPlacementList_t'


cdef extern from 'nvml.h':
    ctypedef nvmlVgpuTypeBar1Info_v1_t nvmlVgpuTypeBar1Info_t 'nvmlVgpuTypeBar1Info_t'


cdef extern from 'nvml.h':
    ctypedef nvmlVgpuRuntimeState_v1_t nvmlVgpuRuntimeState_t 'nvmlVgpuRuntimeState_t'


cdef extern from 'nvml.h':
    ctypedef nvmlSystemConfComputeSettings_v1_t nvmlSystemConfComputeSettings_t 'nvmlSystemConfComputeSettings_t'


cdef extern from 'nvml.h':
    ctypedef nvmlConfComputeSetKeyRotationThresholdInfo_v1_t nvmlConfComputeSetKeyRotationThresholdInfo_t 'nvmlConfComputeSetKeyRotationThresholdInfo_t'


cdef extern from 'nvml.h':
    ctypedef nvmlConfComputeGetKeyRotationThresholdInfo_v1_t nvmlConfComputeGetKeyRotationThresholdInfo_t 'nvmlConfComputeGetKeyRotationThresholdInfo_t'


cdef extern from 'nvml.h':
    ctypedef struct nvmlGpuFabricInfo_t 'nvmlGpuFabricInfo_t':
        unsigned char clusterUuid[16]
        nvmlReturn_t status
        unsigned int cliqueId
        nvmlGpuFabricState_t state

cdef extern from 'nvml.h':
    ctypedef struct nvmlGpuFabricInfo_v2_t 'nvmlGpuFabricInfo_v2_t':
        unsigned int version
        unsigned char clusterUuid[16]
        nvmlReturn_t status
        unsigned int cliqueId
        nvmlGpuFabricState_t state
        unsigned int healthMask

cdef extern from 'nvml.h':
    ctypedef struct nvmlGpuFabricInfo_v3_t 'nvmlGpuFabricInfo_v3_t':
        unsigned int version
        unsigned char clusterUuid[16]
        nvmlReturn_t status
        unsigned int cliqueId
        nvmlGpuFabricState_t state
        unsigned int healthMask
        unsigned char healthSummary

cdef extern from 'nvml.h':
    ctypedef nvmlSystemDriverBranchInfo_v1_t nvmlSystemDriverBranchInfo_t 'nvmlSystemDriverBranchInfo_t'


cdef extern from 'nvml.h':
    ctypedef nvmlTemperature_v1_t nvmlTemperature_t 'nvmlTemperature_t'


cdef extern from 'nvml.h':
    ctypedef nvmlNvlinkSupportedBwModes_v1_t nvmlNvlinkSupportedBwModes_t 'nvmlNvlinkSupportedBwModes_t'


cdef extern from 'nvml.h':
    ctypedef nvmlNvlinkGetBwMode_v1_t nvmlNvlinkGetBwMode_t 'nvmlNvlinkGetBwMode_t'


cdef extern from 'nvml.h':
    ctypedef nvmlNvlinkSetBwMode_v1_t nvmlNvlinkSetBwMode_t 'nvmlNvlinkSetBwMode_t'


cdef extern from 'nvml.h':
    ctypedef nvmlDeviceCapabilities_v1_t nvmlDeviceCapabilities_t 'nvmlDeviceCapabilities_t'


cdef extern from 'nvml.h':
    ctypedef nvmlPowerSmoothingProfile_v1_t nvmlPowerSmoothingProfile_t 'nvmlPowerSmoothingProfile_t'


cdef extern from 'nvml.h':
    ctypedef nvmlPowerSmoothingState_v1_t nvmlPowerSmoothingState_t 'nvmlPowerSmoothingState_t'


cdef extern from 'nvml.h':
    ctypedef nvmlDeviceAddressingMode_v1_t nvmlDeviceAddressingMode_t 'nvmlDeviceAddressingMode_t'


cdef extern from 'nvml.h':
    ctypedef nvmlRepairStatus_v1_t nvmlRepairStatus_t 'nvmlRepairStatus_t'


cdef extern from 'nvml.h':
    ctypedef nvmlPdi_v1_t nvmlPdi_t 'nvmlPdi_t'


cdef extern from 'nvml.h':
    ctypedef struct nvmlCPERCursor_v1_t 'nvmlCPERCursor_v1_t':
        unsigned int cperTypeMask
        char uuid[80]
        nvmlCPERCursorHandle_t handle

cdef extern from 'nvml.h':
    ctypedef struct nvmlEventData_t 'nvmlEventData_t':
        nvmlDevice_t device
        unsigned long long eventType
        unsigned long long eventData
        unsigned int gpuInstanceId
        unsigned int computeInstanceId

cdef extern from 'nvml.h':
    ctypedef struct nvmlSystemEventSetCreateRequest_v1_t 'nvmlSystemEventSetCreateRequest_v1_t':
        unsigned int version
        nvmlSystemEventSet_t set

cdef extern from 'nvml.h':
    ctypedef struct nvmlSystemEventSetFreeRequest_v1_t 'nvmlSystemEventSetFreeRequest_v1_t':
        unsigned int version
        nvmlSystemEventSet_t set

cdef extern from 'nvml.h':
    ctypedef struct nvmlSystemRegisterEventRequest_v1_t 'nvmlSystemRegisterEventRequest_v1_t':
        unsigned int version
        unsigned long long eventTypes
        nvmlSystemEventSet_t set

cdef extern from 'nvml.h':
    ctypedef struct nvmlExcludedDeviceInfo_t 'nvmlExcludedDeviceInfo_t':
        nvmlPciInfo_t pciInfo
        char uuid[80]

cdef extern from 'nvml.h':
    ctypedef struct nvmlProcessDetailList_v1_t 'nvmlProcessDetailList_v1_t':
        unsigned int version
        unsigned int mode
        unsigned int numProcArrayEntries
        nvmlProcessDetail_v1_t* procArray

cdef extern from 'nvml.h':
    ctypedef struct nvmlBridgeChipHierarchy_t 'nvmlBridgeChipHierarchy_t':
        unsigned char bridgeCount
        nvmlBridgeChipInfo_t bridgeChipInfo[128]

cdef extern from 'nvml.h':
    ctypedef struct nvmlSample_t 'nvmlSample_t':
        unsigned long long timeStamp
        nvmlValue_t sampleValue

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuInstanceUtilizationSample_t 'nvmlVgpuInstanceUtilizationSample_t':
        nvmlVgpuInstance_t vgpuInstance
        unsigned long long timeStamp
        nvmlValue_t smUtil
        nvmlValue_t memUtil
        nvmlValue_t encUtil
        nvmlValue_t decUtil

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuInstanceUtilizationInfo_v1_t 'nvmlVgpuInstanceUtilizationInfo_v1_t':
        unsigned long long timeStamp
        nvmlVgpuInstance_t vgpuInstance
        nvmlValue_t smUtil
        nvmlValue_t memUtil
        nvmlValue_t encUtil
        nvmlValue_t decUtil
        nvmlValue_t jpgUtil
        nvmlValue_t ofaUtil

cdef extern from 'nvml.h':
    ctypedef struct nvmlFieldValue_t 'nvmlFieldValue_t':
        unsigned int fieldId
        unsigned int scopeId
        long long timestamp
        long long latencyUsec
        nvmlValueType_t valueType
        nvmlReturn_t nvmlReturn
        nvmlValue_t value

cdef extern from 'nvml.h':
    ctypedef struct nvmlPRMCounterValue_v1_t 'nvmlPRMCounterValue_v1_t':
        nvmlReturn_t status
        nvmlValueType_t outputType
        nvmlValue_t outputValue

cdef extern from 'nvml.h':
    ctypedef struct nvmlGpuThermalSettings_t 'nvmlGpuThermalSettings_t':
        unsigned int count
        cuda_bindings_nvml__anon_pod0 sensor[3]

cdef extern from 'nvml.h':
    ctypedef struct nvmlUUID_v1_t 'nvmlUUID_v1_t':
        unsigned int version
        unsigned int type
        nvmlUUIDValue_t value

cdef extern from 'nvml.h':
    ctypedef struct nvmlClkMonStatus_t 'nvmlClkMonStatus_t':
        unsigned int bGlobalStatus
        unsigned int clkMonListSize
        nvmlClkMonFaultInfo_t clkMonList[32]

cdef extern from 'nvml.h':
    ctypedef struct nvmlProcessesUtilizationInfo_v1_t 'nvmlProcessesUtilizationInfo_v1_t':
        unsigned int version
        unsigned int processSamplesCount
        unsigned long long lastSeenTimeStamp
        nvmlProcessUtilizationInfo_v1_t* procUtilArray

cdef extern from 'nvml.h':
    ctypedef struct nvmlGpuDynamicPstatesInfo_t 'nvmlGpuDynamicPstatesInfo_t':
        unsigned int flags
        cuda_bindings_nvml__anon_pod1 utilization[8]

cdef extern from 'nvml.h':
    ctypedef union nvmlVgpuSchedulerParams_t 'nvmlVgpuSchedulerParams_t':
        cuda_bindings_nvml__anon_pod2 vgpuSchedDataWithARR
        cuda_bindings_nvml__anon_pod3 vgpuSchedData

cdef extern from 'nvml.h':
    ctypedef union nvmlVgpuSchedulerSetParams_t 'nvmlVgpuSchedulerSetParams_t':
        cuda_bindings_nvml__anon_pod4 vgpuSchedDataWithARR
        cuda_bindings_nvml__anon_pod5 vgpuSchedData

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuLicenseInfo_t 'nvmlVgpuLicenseInfo_t':
        unsigned char isLicensed
        nvmlVgpuLicenseExpiry_t licenseExpiry
        unsigned int currentState

cdef extern from 'nvml.h':
    ctypedef struct nvmlGridLicensableFeature_t 'nvmlGridLicensableFeature_t':
        nvmlGridLicenseFeatureCode_t featureCode
        unsigned int featureState
        char licenseInfo[128]
        char productName[128]
        unsigned int featureEnabled
        nvmlGridLicenseExpiry_t licenseExpiry

cdef extern from 'nvml.h':
    ctypedef struct nvmlUnitFanSpeeds_t 'nvmlUnitFanSpeeds_t':
        nvmlUnitFanInfo_t fans[24]
        unsigned int count

cdef extern from 'nvml.h':
    ctypedef struct nvmlSystemEventSetWaitRequest_v1_t 'nvmlSystemEventSetWaitRequest_v1_t':
        unsigned int version
        unsigned int timeoutms
        nvmlSystemEventSet_t set
        nvmlSystemEventData_v1_t* data
        unsigned int dataSize
        unsigned int numEvent

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuPgpuMetadata_t 'nvmlVgpuPgpuMetadata_t':
        unsigned int version
        unsigned int revision
        char hostDriverVersion[80]
        unsigned int pgpuVirtualizationCaps
        unsigned int reserved[5]
        nvmlVgpuVersion_t hostSupportedVgpuRange
        unsigned int opaqueDataSize
        char opaqueData[4]

cdef extern from 'nvml.h':
    ctypedef struct nvmlGpuInstanceInfo_t 'nvmlGpuInstanceInfo_t':
        nvmlDevice_t device
        unsigned int id
        unsigned int profileId
        nvmlGpuInstancePlacement_t placement

cdef extern from 'nvml.h':
    ctypedef struct nvmlComputeInstanceInfo_t 'nvmlComputeInstanceInfo_t':
        nvmlDevice_t device
        nvmlGpuInstance_t gpuInstance
        unsigned int id
        unsigned int profileId
        nvmlComputeInstancePlacement_t placement

cdef extern from 'nvml.h':
    ctypedef struct nvmlGpmMetric_t 'nvmlGpmMetric_t':
        unsigned int metricId
        nvmlReturn_t nvmlReturn
        double value
        cuda_bindings_nvml__anon_pod6 metricInfo

cdef extern from 'nvml.h':
    ctypedef struct nvmlWorkloadPowerProfileInfo_v1_t 'nvmlWorkloadPowerProfileInfo_v1_t':
        unsigned int version
        unsigned int profileId
        unsigned int priority
        nvmlMask255_t conflictingMask

cdef extern from 'nvml.h':
    ctypedef struct nvmlWorkloadPowerProfileCurrentProfiles_v1_t 'nvmlWorkloadPowerProfileCurrentProfiles_v1_t':
        unsigned int version
        nvmlMask255_t perfProfilesMask
        nvmlMask255_t requestedProfilesMask
        nvmlMask255_t enforcedProfilesMask

cdef extern from 'nvml.h':
    ctypedef struct nvmlWorkloadPowerProfileRequestedProfiles_v1_t 'nvmlWorkloadPowerProfileRequestedProfiles_v1_t':
        unsigned int version
        nvmlMask255_t requestedProfilesMask

cdef extern from 'nvml.h':
    ctypedef struct nvmlWorkloadPowerProfileUpdateProfiles_v1_t 'nvmlWorkloadPowerProfileUpdateProfiles_v1_t':
        nvmlPowerProfileOperation_t operation
        nvmlMask255_t updateProfilesMask

cdef extern from 'nvml.h':
    ctypedef struct nvmlEccSramUniqueUncorrectedErrorCounts_v1_t 'nvmlEccSramUniqueUncorrectedErrorCounts_v1_t':
        unsigned int version
        unsigned int entryCount
        nvmlEccSramUniqueUncorrectedErrorEntry_v1_t* entries

cdef extern from 'nvml.h':
    ctypedef struct nvmlNvlinkFirmwareInfo_t 'nvmlNvlinkFirmwareInfo_t':
        nvmlNvlinkFirmwareVersion_t firmwareVersion[100]
        unsigned int numValidEntries

cdef extern from 'nvml.h':
    ctypedef struct nvmlPRMTLV_v1_t 'nvmlPRMTLV_v1_t':
        unsigned dataSize
        unsigned status
        cuda_bindings_nvml__anon_pod7 _anon_pod_member0

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuSchedulerLogInfo_v2_t 'nvmlVgpuSchedulerLogInfo_v2_t':
        unsigned int engineId
        unsigned int schedulerPolicy
        unsigned int avgFactor
        unsigned int timeslice
        unsigned int entriesCount
        nvmlVgpuSchedulerLogEntry_v2_t logEntries[200]

cdef extern from 'nvml.h':
    ctypedef nvmlVgpuTypeIdInfo_v1_t nvmlVgpuTypeIdInfo_t 'nvmlVgpuTypeIdInfo_t'


cdef extern from 'nvml.h':
    ctypedef nvmlVgpuTypeMaxInstance_v1_t nvmlVgpuTypeMaxInstance_t 'nvmlVgpuTypeMaxInstance_t'


cdef extern from 'nvml.h':
    ctypedef nvmlVgpuCreatablePlacementInfo_v1_t nvmlVgpuCreatablePlacementInfo_t 'nvmlVgpuCreatablePlacementInfo_t'


cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuProcessesUtilizationInfo_v1_t 'nvmlVgpuProcessesUtilizationInfo_v1_t':
        unsigned int version
        unsigned int vgpuProcessCount
        unsigned long long lastSeenTimeStamp
        nvmlVgpuProcessUtilizationInfo_v1_t* vgpuProcUtilArray

cdef extern from 'nvml.h':
    ctypedef nvmlActiveVgpuInstanceInfo_v1_t nvmlActiveVgpuInstanceInfo_t 'nvmlActiveVgpuInstanceInfo_t'


cdef extern from 'nvml.h':
    ctypedef nvmlGpuFabricInfo_v3_t nvmlGpuFabricInfoV_t 'nvmlGpuFabricInfoV_t'


cdef extern from 'nvml.h':
    ctypedef struct nvmlGetCPER_v1_t 'nvmlGetCPER_v1_t':
        nvmlCPERCursor_v1_t cursor
        unsigned char* buffer
        unsigned int bufferSize

cdef extern from 'nvml.h':
    ctypedef nvmlSystemEventSetCreateRequest_v1_t nvmlSystemEventSetCreateRequest_t 'nvmlSystemEventSetCreateRequest_t'


cdef extern from 'nvml.h':
    ctypedef nvmlSystemEventSetFreeRequest_v1_t nvmlSystemEventSetFreeRequest_t 'nvmlSystemEventSetFreeRequest_t'


cdef extern from 'nvml.h':
    ctypedef nvmlSystemRegisterEventRequest_v1_t nvmlSystemRegisterEventRequest_t 'nvmlSystemRegisterEventRequest_t'


cdef extern from 'nvml.h':
    ctypedef nvmlProcessDetailList_v1_t nvmlProcessDetailList_t 'nvmlProcessDetailList_t'


cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuInstancesUtilizationInfo_v1_t 'nvmlVgpuInstancesUtilizationInfo_v1_t':
        unsigned int version
        nvmlValueType_t sampleValType
        unsigned int vgpuInstanceCount
        unsigned long long lastSeenTimeStamp
        nvmlVgpuInstanceUtilizationInfo_v1_t* vgpuUtilArray

cdef extern from 'nvml.h':
    ctypedef struct nvmlPRMCounter_v1_t 'nvmlPRMCounter_v1_t':
        unsigned int counterId
        nvmlPRMCounterInput_v1_t inData
        nvmlPRMCounterValue_v1_t counterValue

cdef extern from 'nvml.h':
    ctypedef nvmlUUID_v1_t nvmlUUID_t 'nvmlUUID_t'


cdef extern from 'nvml.h':
    ctypedef nvmlProcessesUtilizationInfo_v1_t nvmlProcessesUtilizationInfo_t 'nvmlProcessesUtilizationInfo_t'


cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuSchedulerLog_t 'nvmlVgpuSchedulerLog_t':
        unsigned int engineId
        unsigned int schedulerPolicy
        unsigned int arrMode
        nvmlVgpuSchedulerParams_t schedulerParams
        unsigned int entriesCount
        nvmlVgpuSchedulerLogEntry_t logEntries[200]

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuSchedulerGetState_t 'nvmlVgpuSchedulerGetState_t':
        unsigned int schedulerPolicy
        unsigned int arrMode
        nvmlVgpuSchedulerParams_t schedulerParams

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuSchedulerStateInfo_v1_t 'nvmlVgpuSchedulerStateInfo_v1_t':
        unsigned int version
        unsigned int engineId
        unsigned int schedulerPolicy
        unsigned int arrMode
        nvmlVgpuSchedulerParams_t schedulerParams

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuSchedulerLogInfo_v1_t 'nvmlVgpuSchedulerLogInfo_v1_t':
        unsigned int version
        unsigned int engineId
        unsigned int schedulerPolicy
        unsigned int arrMode
        nvmlVgpuSchedulerParams_t schedulerParams
        unsigned int entriesCount
        nvmlVgpuSchedulerLogEntry_t logEntries[200]

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuSchedulerSetState_t 'nvmlVgpuSchedulerSetState_t':
        unsigned int schedulerPolicy
        unsigned int enableARRMode
        nvmlVgpuSchedulerSetParams_t schedulerParams

cdef extern from 'nvml.h':
    ctypedef struct nvmlVgpuSchedulerState_v1_t 'nvmlVgpuSchedulerState_v1_t':
        unsigned int version
        unsigned int engineId
        unsigned int schedulerPolicy
        unsigned int enableARRMode
        nvmlVgpuSchedulerSetParams_t schedulerParams

cdef extern from 'nvml.h':
    ctypedef struct nvmlGridLicensableFeatures_t 'nvmlGridLicensableFeatures_t':
        int isGridLicenseSupported
        unsigned int licensableFeaturesCount
        nvmlGridLicensableFeature_t gridLicensableFeatures[3]

cdef extern from 'nvml.h':
    ctypedef nvmlSystemEventSetWaitRequest_v1_t nvmlSystemEventSetWaitRequest_t 'nvmlSystemEventSetWaitRequest_t'


cdef extern from 'nvml.h':
    ctypedef struct nvmlGpmMetricsGet_t 'nvmlGpmMetricsGet_t':
        unsigned int version
        unsigned int numMetrics
        nvmlGpmSample_t sample1
        nvmlGpmSample_t sample2
        nvmlGpmMetric_t metrics[333]

cdef extern from 'nvml.h':
    ctypedef nvmlWorkloadPowerProfileInfo_v1_t nvmlWorkloadPowerProfileInfo_t 'nvmlWorkloadPowerProfileInfo_t'


cdef extern from 'nvml.h':
    ctypedef nvmlWorkloadPowerProfileCurrentProfiles_v1_t nvmlWorkloadPowerProfileCurrentProfiles_t 'nvmlWorkloadPowerProfileCurrentProfiles_t'


cdef extern from 'nvml.h':
    ctypedef nvmlWorkloadPowerProfileRequestedProfiles_v1_t nvmlWorkloadPowerProfileRequestedProfiles_t 'nvmlWorkloadPowerProfileRequestedProfiles_t'


cdef extern from 'nvml.h':
    ctypedef nvmlEccSramUniqueUncorrectedErrorCounts_v1_t nvmlEccSramUniqueUncorrectedErrorCounts_t 'nvmlEccSramUniqueUncorrectedErrorCounts_t'


cdef extern from 'nvml.h':
    ctypedef struct nvmlNvLinkInfo_v2_t 'nvmlNvLinkInfo_v2_t':
        unsigned int version
        unsigned int isNvleEnabled
        nvmlNvlinkFirmwareInfo_t firmwareInfo

cdef extern from 'nvml.h':
    ctypedef nvmlVgpuProcessesUtilizationInfo_v1_t nvmlVgpuProcessesUtilizationInfo_t 'nvmlVgpuProcessesUtilizationInfo_t'


cdef extern from 'nvml.h':
    ctypedef nvmlVgpuInstancesUtilizationInfo_v1_t nvmlVgpuInstancesUtilizationInfo_t 'nvmlVgpuInstancesUtilizationInfo_t'


cdef extern from 'nvml.h':
    ctypedef struct nvmlPRMCounterList_v1_t 'nvmlPRMCounterList_v1_t':
        unsigned int numCounters
        nvmlPRMCounter_v1_t* counters

cdef extern from 'nvml.h':
    ctypedef nvmlVgpuSchedulerStateInfo_v1_t nvmlVgpuSchedulerStateInfo_t 'nvmlVgpuSchedulerStateInfo_t'


cdef extern from 'nvml.h':
    ctypedef nvmlVgpuSchedulerLogInfo_v1_t nvmlVgpuSchedulerLogInfo_t 'nvmlVgpuSchedulerLogInfo_t'


cdef extern from 'nvml.h':
    ctypedef nvmlVgpuSchedulerState_v1_t nvmlVgpuSchedulerState_t 'nvmlVgpuSchedulerState_t'


cdef extern from 'nvml.h':
    ctypedef struct nvmlWorkloadPowerProfileProfilesInfo_v1_t 'nvmlWorkloadPowerProfileProfilesInfo_v1_t':
        unsigned int version
        nvmlMask255_t perfProfilesMask
        nvmlWorkloadPowerProfileInfo_t perfProfile[255]

cdef extern from 'nvml.h':
    ctypedef nvmlNvLinkInfo_v2_t nvmlNvLinkInfo_t 'nvmlNvLinkInfo_t'


cdef extern from 'nvml.h':
    ctypedef nvmlWorkloadPowerProfileProfilesInfo_v1_t nvmlWorkloadPowerProfileProfilesInfo_t 'nvmlWorkloadPowerProfileProfilesInfo_t'



###############################################################################
# Functions
###############################################################################

cdef nvmlReturn_t nvmlInit_v2() except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlInitWithFlags(unsigned int flags) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlShutdown() except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef const char* nvmlErrorString(nvmlReturn_t result) except?NULL nogil
cdef nvmlReturn_t nvmlSystemGetDriverVersion(char* version, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlSystemGetNVMLVersion(char* version, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlSystemGetCudaDriverVersion(int* cudaDriverVersion) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int* cudaDriverVersion) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlSystemGetProcessName(unsigned int pid, char* name, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlSystemGetHicVersion(unsigned int* hwbcCount, nvmlHwbcEntry_t* hwbcEntries) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber, unsigned int* count, nvmlDevice_t* deviceArray) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlSystemGetDriverBranch(nvmlSystemDriverBranchInfo_t* branchInfo, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlUnitGetCount(unsigned int* unitCount) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlUnitGetHandleByIndex(unsigned int index, nvmlUnit_t* unit) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlUnitGetUnitInfo(nvmlUnit_t unit, nvmlUnitInfo_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlUnitGetLedState(nvmlUnit_t unit, nvmlLedState_t* state) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlUnitGetPsuInfo(nvmlUnit_t unit, nvmlPSUInfo_t* psu) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlUnitGetTemperature(nvmlUnit_t unit, unsigned int type, unsigned int* temp) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlUnitGetFanSpeedInfo(nvmlUnit_t unit, nvmlUnitFanSpeeds_t* fanSpeeds) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlUnitGetDevices(nvmlUnit_t unit, unsigned int* deviceCount, nvmlDevice_t* devices) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* deviceCount) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetAttributes_v2(nvmlDevice_t device, nvmlDeviceAttributes_t* attributes) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t* device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetHandleBySerial(const char* serial, nvmlDevice_t* device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetHandleByUUID(const char* uuid, nvmlDevice_t* device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetHandleByUUIDV(const nvmlUUID_t* uuid, nvmlDevice_t* device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char* pciBusId, nvmlDevice_t* device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device, nvmlBrandType_t* type) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t device, char* serial, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetModuleId(nvmlDevice_t device, unsigned int* moduleId) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetC2cModeInfoV(nvmlDevice_t device, nvmlC2cModeInfo_v1_t* c2cModeInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetMemoryAffinity(nvmlDevice_t device, unsigned int nodeSetSize, unsigned long* nodeSet, nvmlAffinityScope_t scope) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetCpuAffinityWithinScope(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long* cpuSet, nvmlAffinityScope_t scope) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetCpuAffinity(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long* cpuSet) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetCpuAffinity(nvmlDevice_t device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceClearCpuAffinity(nvmlDevice_t device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetNumaNodeId(nvmlDevice_t device, unsigned int* node) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuTopologyLevel_t* pathInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetTopologyNearestGpus(nvmlDevice_t device, nvmlGpuTopologyLevel_t level, unsigned int* count, nvmlDevice_t* deviceArray) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetP2PStatus(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex, nvmlGpuP2PStatus_t* p2pStatus) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char* uuid, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int* minorNumber) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetBoardPartNumber(nvmlDevice_t device, char* partNumber, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetInforomVersion(nvmlDevice_t device, nvmlInforomObject_t object, char* version, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetInforomImageVersion(nvmlDevice_t device, char* version, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetInforomConfigurationChecksum(nvmlDevice_t device, unsigned int* checksum) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceValidateInforom(nvmlDevice_t device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetLastBBXFlushTime(nvmlDevice_t device, unsigned long long* timestamp, unsigned long* durationUs) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device, nvmlEnableState_t* display) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, nvmlEnableState_t* isActive) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t* mode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetPciInfoExt(nvmlDevice_t device, nvmlPciInfoExt_t* pci) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t* pci) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int* maxLinkGen) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetGpuMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int* maxLinkGenDevice) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device, unsigned int* maxLinkWidth) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int* currLinkGen) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int* currLinkWidth) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int* value) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device, unsigned int* value) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetGpcClkVfOffset(nvmlDevice_t device, int* offset) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int* clockMHz) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetMaxCustomerBoostClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int* clockMHz) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device, unsigned int* count, unsigned int* clocksMHz) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t device, unsigned int memoryClockMHz, unsigned int* count, unsigned int* clocksMHz) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t* isEnabled, nvmlEnableState_t* defaultIsEnabled) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int* speed) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int* speed) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetFanSpeedRPM(nvmlDevice_t device, nvmlFanSpeedInfo_t* fanSpeed) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetTargetFanSpeed(nvmlDevice_t device, unsigned int fan, unsigned int* targetSpeed) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetMinMaxFanSpeed(nvmlDevice_t device, unsigned int* minSpeed, unsigned int* maxSpeed) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetFanControlPolicy_v2(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t* policy) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetNumFans(nvmlDevice_t device, unsigned int* numFans) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetCoolerInfo(nvmlDevice_t device, nvmlCoolerInfo_t* coolerInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetTemperatureV(nvmlDevice_t device, nvmlTemperature_t* temperature) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, unsigned int* temp) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetMarginTemperature(nvmlDevice_t device, nvmlMarginTemperature_t* marginTempInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetThermalSettings(nvmlDevice_t device, unsigned int sensorIndex, nvmlGpuThermalSettings_t* pThermalSettings) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t* pState) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetCurrentClocksEventReasons(nvmlDevice_t device, unsigned long long* clocksEventReasons) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetSupportedClocksEventReasons(nvmlDevice_t device, unsigned long long* supportedClocksEventReasons) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t device, nvmlPstates_t* pState) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetDynamicPstatesInfo(nvmlDevice_t device, nvmlGpuDynamicPstatesInfo_t* pDynamicPstatesInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetMemClkVfOffset(nvmlDevice_t device, int* offset) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetMinMaxClockOfPState(nvmlDevice_t device, nvmlClockType_t type, nvmlPstates_t pstate, unsigned int* minClockMHz, unsigned int* maxClockMHz) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetSupportedPerformanceStates(nvmlDevice_t device, nvmlPstates_t* pstates, unsigned int size) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetGpcClkMinMaxVfOffset(nvmlDevice_t device, int* minOffset, int* maxOffset) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetMemClkMinMaxVfOffset(nvmlDevice_t device, int* minOffset, int* maxOffset) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetClockOffsets(nvmlDevice_t device, nvmlClockOffset_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetClockOffsets(nvmlDevice_t device, nvmlClockOffset_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetPerformanceModes(nvmlDevice_t device, nvmlDevicePerfModes_t* perfModes) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetCurrentClockFreqs(nvmlDevice_t device, nvmlDeviceCurrentClockFreqs_t* currentClockFreqs) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int* limit) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device, unsigned int* minLimit, unsigned int* maxLimit) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device, unsigned int* defaultLimit) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int* power) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t device, unsigned long long* energy) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int* limit) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t* current, nvmlGpuOperationMode_t* pending) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t* memory) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t* mode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major, int* minor) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetDramEncryptionMode(nvmlDevice_t device, nvmlDramEncryptionInfo_t* current, nvmlDramEncryptionInfo_t* pending) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetDramEncryptionMode(nvmlDevice_t device, const nvmlDramEncryptionInfo_t* dramEncryption) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, nvmlEnableState_t* current, nvmlEnableState_t* pending) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetDefaultEccMode(nvmlDevice_t device, nvmlEnableState_t* defaultMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int* boardId) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device, unsigned int* multiGpuBool) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetTotalEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long* eccCounts) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlMemoryLocation_t locationType, unsigned long long* count) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t* utilization) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetEncoderCapacity(nvmlDevice_t device, nvmlEncoderType_t encoderQueryType, unsigned int* encoderCapacity) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetEncoderStats(nvmlDevice_t device, unsigned int* sessionCount, unsigned int* averageFps, unsigned int* averageLatency) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetEncoderSessions(nvmlDevice_t device, unsigned int* sessionCount, nvmlEncoderSessionInfo_t* sessionInfos) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetDecoderUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetJpgUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetOfaUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetFBCStats(nvmlDevice_t device, nvmlFBCStats_t* fbcStats) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetFBCSessions(nvmlDevice_t device, unsigned int* sessionCount, nvmlFBCSessionInfo_t* sessionInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetDriverModel_v2(nvmlDevice_t device, nvmlDriverModel_t* current, nvmlDriverModel_t* pending) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetVbiosVersion(nvmlDevice_t device, char* version, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetBridgeChipInfo(nvmlDevice_t device, nvmlBridgeChipHierarchy_t* bridgeHierarchy) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetRunningProcessDetailList(nvmlDevice_t device, nvmlProcessDetailList_t* plist) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceOnSameBoard(nvmlDevice_t device1, nvmlDevice_t device2, int* onSameBoard) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t* isRestricted) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetSamples(nvmlDevice_t device, nvmlSamplingType_t type, unsigned long long lastSeenTimeStamp, nvmlValueType_t* sampleValType, unsigned int* sampleCount, nvmlSample_t* samples) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t device, nvmlBAR1Memory_t* bar1Memory) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetIrqNum(nvmlDevice_t device, unsigned int* irqNum) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetNumGpuCores(nvmlDevice_t device, unsigned int* numCores) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetPowerSource(nvmlDevice_t device, nvmlPowerSource_t* powerSource) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetMemoryBusWidth(nvmlDevice_t device, unsigned int* busWidth) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetPcieLinkMaxSpeed(nvmlDevice_t device, unsigned int* maxSpeed) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetPcieSpeed(nvmlDevice_t device, unsigned int* pcieSpeed) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetAdaptiveClockInfoStatus(nvmlDevice_t device, unsigned int* adaptiveClockStatus) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetBusType(nvmlDevice_t device, nvmlBusType_t* type) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetGpuFabricInfoV(nvmlDevice_t device, nvmlGpuFabricInfoV_t* gpuFabricInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlSystemGetConfComputeCapabilities(nvmlConfComputeSystemCaps_t* capabilities) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlSystemGetConfComputeState(nvmlConfComputeSystemState_t* state) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetConfComputeMemSizeInfo(nvmlDevice_t device, nvmlConfComputeMemSizeInfo_t* memInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlSystemGetConfComputeGpusReadyState(unsigned int* isAcceptingWork) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetConfComputeProtectedMemoryUsage(nvmlDevice_t device, nvmlMemory_t* memory) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetConfComputeGpuCertificate(nvmlDevice_t device, nvmlConfComputeGpuCertificate_t* gpuCert) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetConfComputeGpuAttestationReport(nvmlDevice_t device, nvmlConfComputeGpuAttestationReport_t* gpuAtstReport) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlSystemGetConfComputeKeyRotationThresholdInfo(nvmlConfComputeGetKeyRotationThresholdInfo_t* pKeyRotationThrInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetConfComputeUnprotectedMemSize(nvmlDevice_t device, unsigned long long sizeKiB) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlSystemSetConfComputeGpusReadyState(unsigned int isAcceptingWork) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlSystemSetConfComputeKeyRotationThresholdInfo(nvmlConfComputeSetKeyRotationThresholdInfo_t* pKeyRotationThrInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlSystemGetConfComputeSettings(nvmlSystemConfComputeSettings_t* settings) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetGspFirmwareVersion(nvmlDevice_t device, char* version) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetGspFirmwareMode(nvmlDevice_t device, unsigned int* isEnabled, unsigned int* defaultMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetSramEccErrorStatus(nvmlDevice_t device, nvmlEccSramErrorStatus_t* status) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetAccountingMode(nvmlDevice_t device, nvmlEnableState_t* mode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetAccountingStats(nvmlDevice_t device, unsigned int pid, nvmlAccountingStats_t* stats) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetAccountingPids(nvmlDevice_t device, unsigned int* count, unsigned int* pids) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetAccountingBufferSize(nvmlDevice_t device, unsigned int* bufferSize) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetRetiredPages(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int* pageCount, unsigned long long* addresses) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int* pageCount, unsigned long long* addresses, unsigned long long* timestamps) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device, nvmlEnableState_t* isPending) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetRemappedRows(nvmlDevice_t device, unsigned int* corrRows, unsigned int* uncRows, unsigned int* isPending, unsigned int* failureOccurred) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetRowRemapperHistogram(nvmlDevice_t device, nvmlRowRemapperHistogramValues_t* values) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t device, nvmlDeviceArchitecture_t* arch) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetClkMonStatus(nvmlDevice_t device, nvmlClkMonStatus_t* status) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetProcessUtilization(nvmlDevice_t device, nvmlProcessUtilizationSample_t* utilization, unsigned int* processSamplesCount, unsigned long long lastSeenTimeStamp) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetProcessesUtilizationInfo(nvmlDevice_t device, nvmlProcessesUtilizationInfo_t* procesesUtilInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetPlatformInfo(nvmlDevice_t device, nvmlPlatformInfo_t* platformInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlUnitSetLedState(nvmlUnit_t unit, nvmlLedColor_t color) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t mode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetComputeMode(nvmlDevice_t device, nvmlComputeMode_t mode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetEccMode(nvmlDevice_t device, nvmlEnableState_t ecc) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceClearEccErrorCounts(nvmlDevice_t device, nvmlEccCounterType_t counterType) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetDriverModel(nvmlDevice_t device, nvmlDriverModel_t driverModel, unsigned int flags) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetGpuLockedClocks(nvmlDevice_t device, unsigned int minGpuClockMHz, unsigned int maxGpuClockMHz) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceResetGpuLockedClocks(nvmlDevice_t device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetMemoryLockedClocks(nvmlDevice_t device, unsigned int minMemClockMHz, unsigned int maxMemClockMHz) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceResetMemoryLockedClocks(nvmlDevice_t device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetDefaultAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled, unsigned int flags) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetDefaultFanSpeed_v2(nvmlDevice_t device, unsigned int fan) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetFanControlPolicy(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t policy) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, int* temp) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t mode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t isRestricted) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int speed) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetAccountingMode(nvmlDevice_t device, nvmlEnableState_t mode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceClearAccountingPids(nvmlDevice_t device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetPowerManagementLimit_v2(nvmlDevice_t device, nvmlPowerValue_v2_t* powerValue) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t* isActive) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetNvLinkVersion(nvmlDevice_t device, unsigned int link, unsigned int* version) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int* capResult) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t* pci) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetNvLinkErrorCounter(nvmlDevice_t device, unsigned int link, nvmlNvLinkErrorCounter_t counter, unsigned long long* counterValue) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceResetNvLinkErrorCounters(nvmlDevice_t device, unsigned int link) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link, nvmlIntNvLinkDeviceType_t* pNvLinkDeviceType) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetNvLinkDeviceLowPowerThreshold(nvmlDevice_t device, nvmlNvLinkPowerThres_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlSystemSetNvlinkBwMode(unsigned int nvlinkBwMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlSystemGetNvlinkBwMode(unsigned int* nvlinkBwMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetNvlinkSupportedBwModes(nvmlDevice_t device, nvmlNvlinkSupportedBwModes_t* supportedBwMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetNvlinkBwMode(nvmlDevice_t device, nvmlNvlinkGetBwMode_t* getBwMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetNvlinkBwMode(nvmlDevice_t device, nvmlNvlinkSetBwMode_t* setBwMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlEventSetCreate(nvmlEventSet_t* set) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceRegisterEvents(nvmlDevice_t device, unsigned long long eventTypes, nvmlEventSet_t set) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetSupportedEventTypes(nvmlDevice_t device, unsigned long long* eventTypes) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlEventSetWait_v2(nvmlEventSet_t set, nvmlEventData_t* data, unsigned int timeoutms) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlEventSetFree(nvmlEventSet_t set) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlSystemEventSetCreate(nvmlSystemEventSetCreateRequest_t* request) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlSystemEventSetFree(nvmlSystemEventSetFreeRequest_t* request) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlSystemRegisterEvents(nvmlSystemRegisterEventRequest_t* request) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlSystemEventSetWait(nvmlSystemEventSetWaitRequest_t* request) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceModifyDrainState(nvmlPciInfo_t* pciInfo, nvmlEnableState_t newState) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceQueryDrainState(nvmlPciInfo_t* pciInfo, nvmlEnableState_t* currentState) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceRemoveGpu_v2(nvmlPciInfo_t* pciInfo, nvmlDetachGpuState_t gpuState, nvmlPcieLinkState_t linkState) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceDiscoverGpus(nvmlPciInfo_t* pciInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceClearFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t* pVirtualMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetHostVgpuMode(nvmlDevice_t device, nvmlHostVgpuMode_t* pHostVgpuMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t virtualMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetVgpuHeterogeneousMode(nvmlDevice_t device, nvmlVgpuHeterogeneousMode_t* pHeterogeneousMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetVgpuHeterogeneousMode(nvmlDevice_t device, const nvmlVgpuHeterogeneousMode_t* pHeterogeneousMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetPlacementId(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuPlacementId_t* pPlacement) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetVgpuTypeSupportedPlacements(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuPlacementList_t* pPlacementList) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetVgpuTypeCreatablePlacements(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuPlacementList_t* pPlacementList) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuTypeGetGspHeapSize(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* gspHeapSize) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuTypeGetFbReservation(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* fbReservation) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetRuntimeStateSize(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuRuntimeState_t* pState) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetVgpuCapabilities(nvmlDevice_t device, nvmlDeviceVgpuCapability_t capability, nvmlEnableState_t state) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetGridLicensableFeatures_v4(nvmlDevice_t device, nvmlGridLicensableFeatures_t* pGridLicensableFeatures) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGetVgpuDriverCapabilities(nvmlVgpuDriverCapability_t capability, unsigned int* capResult) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetVgpuCapabilities(nvmlDevice_t device, nvmlDeviceVgpuCapability_t capability, unsigned int* capResult) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetSupportedVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuTypeId_t* vgpuTypeIds) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetCreatableVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuTypeId_t* vgpuTypeIds) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuTypeGetClass(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeClass, unsigned int* size) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuTypeGetName(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeName, unsigned int* size) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuTypeGetGpuInstanceProfileId(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* gpuInstanceProfileId) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuTypeGetDeviceID(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* deviceID, unsigned long long* subsystemID) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuTypeGetFramebufferSize(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* fbSize) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuTypeGetNumDisplayHeads(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* numDisplayHeads) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuTypeGetResolution(nvmlVgpuTypeId_t vgpuTypeId, unsigned int displayIndex, unsigned int* xdim, unsigned int* ydim) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuTypeGetLicense(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeLicenseString, unsigned int size) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuTypeGetFrameRateLimit(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* frameRateLimit) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuTypeGetMaxInstances(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, unsigned int* vgpuInstanceCount) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuTypeGetMaxInstancesPerVm(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* vgpuInstanceCountPerVm) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuTypeGetBAR1Info(nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuTypeBar1Info_t* bar1Info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetActiveVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuInstance_t* vgpuInstances) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetVmID(nvmlVgpuInstance_t vgpuInstance, char* vmId, unsigned int size, nvmlVgpuVmIdType_t* vmIdType) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetUUID(nvmlVgpuInstance_t vgpuInstance, char* uuid, unsigned int size) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetVmDriverVersion(nvmlVgpuInstance_t vgpuInstance, char* version, unsigned int length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetFbUsage(nvmlVgpuInstance_t vgpuInstance, unsigned long long* fbUsage) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetLicenseStatus(nvmlVgpuInstance_t vgpuInstance, unsigned int* licensed) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetType(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuTypeId_t* vgpuTypeId) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetFrameRateLimit(nvmlVgpuInstance_t vgpuInstance, unsigned int* frameRateLimit) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetEccMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t* eccMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int* encoderCapacity) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceSetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int encoderCapacity) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetEncoderStats(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, unsigned int* averageFps, unsigned int* averageLatency) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetEncoderSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, nvmlEncoderSessionInfo_t* sessionInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetFBCStats(nvmlVgpuInstance_t vgpuInstance, nvmlFBCStats_t* fbcStats) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetFBCSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, nvmlFBCSessionInfo_t* sessionInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetGpuInstanceId(nvmlVgpuInstance_t vgpuInstance, unsigned int* gpuInstanceId) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetGpuPciId(nvmlVgpuInstance_t vgpuInstance, char* vgpuPciId, unsigned int* length) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuTypeGetCapabilities(nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuCapability_t capability, unsigned int* capResult) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetMdevUUID(nvmlVgpuInstance_t vgpuInstance, char* mdevUuid, unsigned int size) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGpuInstanceGetCreatableVgpus(nvmlGpuInstance_t gpuInstance, nvmlVgpuTypeIdInfo_t* pVgpus) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuTypeGetMaxInstancesPerGpuInstance(nvmlVgpuTypeMaxInstance_t* pMaxInstance) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGpuInstanceGetActiveVgpus(nvmlGpuInstance_t gpuInstance, nvmlActiveVgpuInstanceInfo_t* pVgpuInstanceInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGpuInstanceSetVgpuSchedulerState(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerState_t* pScheduler) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGpuInstanceGetVgpuSchedulerState(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerStateInfo_t* pSchedulerStateInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGpuInstanceGetVgpuSchedulerLog(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerLogInfo_t* pSchedulerLogInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGpuInstanceGetVgpuTypeCreatablePlacements(nvmlGpuInstance_t gpuInstance, nvmlVgpuCreatablePlacementInfo_t* pCreatablePlacementInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGpuInstanceGetVgpuHeterogeneousMode(nvmlGpuInstance_t gpuInstance, nvmlVgpuHeterogeneousMode_t* pHeterogeneousMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGpuInstanceSetVgpuHeterogeneousMode(nvmlGpuInstance_t gpuInstance, const nvmlVgpuHeterogeneousMode_t* pHeterogeneousMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetMetadata(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuMetadata_t* vgpuMetadata, unsigned int* bufferSize) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetVgpuMetadata(nvmlDevice_t device, nvmlVgpuPgpuMetadata_t* pgpuMetadata, unsigned int* bufferSize) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGetVgpuCompatibility(nvmlVgpuMetadata_t* vgpuMetadata, nvmlVgpuPgpuMetadata_t* pgpuMetadata, nvmlVgpuPgpuCompatibility_t* compatibilityInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetPgpuMetadataString(nvmlDevice_t device, char* pgpuMetadata, unsigned int* bufferSize) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetVgpuSchedulerLog(nvmlDevice_t device, nvmlVgpuSchedulerLog_t* pSchedulerLog) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetVgpuSchedulerState(nvmlDevice_t device, nvmlVgpuSchedulerGetState_t* pSchedulerState) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetVgpuSchedulerCapabilities(nvmlDevice_t device, nvmlVgpuSchedulerCapabilities_t* pCapabilities) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetVgpuSchedulerState(nvmlDevice_t device, nvmlVgpuSchedulerSetState_t* pSchedulerState) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGetVgpuVersion(nvmlVgpuVersion_t* supported, nvmlVgpuVersion_t* current) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlSetVgpuVersion(nvmlVgpuVersion_t* vgpuVersion) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetVgpuUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, nvmlValueType_t* sampleValType, unsigned int* vgpuInstanceSamplesCount, nvmlVgpuInstanceUtilizationSample_t* utilizationSamples) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetVgpuInstancesUtilizationInfo(nvmlDevice_t device, nvmlVgpuInstancesUtilizationInfo_t* vgpuUtilInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetVgpuProcessUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, unsigned int* vgpuProcessSamplesCount, nvmlVgpuProcessUtilizationSample_t* utilizationSamples) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetVgpuProcessesUtilizationInfo(nvmlDevice_t device, nvmlVgpuProcessesUtilizationInfo_t* vgpuProcUtilInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetAccountingMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t* mode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetAccountingPids(nvmlVgpuInstance_t vgpuInstance, unsigned int* count, unsigned int* pids) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetAccountingStats(nvmlVgpuInstance_t vgpuInstance, unsigned int pid, nvmlAccountingStats_t* stats) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceClearAccountingPids(nvmlVgpuInstance_t vgpuInstance) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlVgpuInstanceGetLicenseInfo_v2(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuLicenseInfo_t* licenseInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGetExcludedDeviceCount(unsigned int* deviceCount) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGetExcludedDeviceInfoByIndex(unsigned int index, nvmlExcludedDeviceInfo_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetMigMode(nvmlDevice_t device, unsigned int mode, nvmlReturn_t* activationStatus) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int* currentMode, unsigned int* pendingMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfoV(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_v2_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetGpuInstancePossiblePlacements_v2(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstancePlacement_t* placements, unsigned int* count) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetGpuInstanceRemainingCapacity(nvmlDevice_t device, unsigned int profileId, unsigned int* count) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceCreateGpuInstance(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t* gpuInstance) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceCreateGpuInstanceWithPlacement(nvmlDevice_t device, unsigned int profileId, const nvmlGpuInstancePlacement_t* placement, nvmlGpuInstance_t* gpuInstance) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGpuInstanceDestroy(nvmlGpuInstance_t gpuInstance) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetGpuInstances(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t* gpuInstances, unsigned int* count) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetGpuInstanceById(nvmlDevice_t device, unsigned int id, nvmlGpuInstance_t* gpuInstance) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGpuInstanceGetInfo(nvmlGpuInstance_t gpuInstance, nvmlGpuInstanceInfo_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfoV(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_v2_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGpuInstanceGetComputeInstanceRemainingCapacity(nvmlGpuInstance_t gpuInstance, unsigned int profileId, unsigned int* count) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGpuInstanceGetComputeInstancePossiblePlacements(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstancePlacement_t* placements, unsigned int* count) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGpuInstanceCreateComputeInstance(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t* computeInstance) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGpuInstanceCreateComputeInstanceWithPlacement(nvmlGpuInstance_t gpuInstance, unsigned int profileId, const nvmlComputeInstancePlacement_t* placement, nvmlComputeInstance_t* computeInstance) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlComputeInstanceDestroy(nvmlComputeInstance_t computeInstance) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGpuInstanceGetComputeInstances(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t* computeInstances, unsigned int* count) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGpuInstanceGetComputeInstanceById(nvmlGpuInstance_t gpuInstance, unsigned int id, nvmlComputeInstance_t* computeInstance) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlComputeInstanceGetInfo_v2(nvmlComputeInstance_t computeInstance, nvmlComputeInstanceInfo_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device, unsigned int* isMigDevice) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetGpuInstanceId(nvmlDevice_t device, unsigned int* id) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetComputeInstanceId(nvmlDevice_t device, unsigned int* id) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device, unsigned int* count) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t device, unsigned int index, nvmlDevice_t* migDevice) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t migDevice, nvmlDevice_t* device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetCapabilities(nvmlDevice_t device, nvmlDeviceCapabilities_t* caps) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDevicePowerSmoothingActivatePresetProfile(nvmlDevice_t device, nvmlPowerSmoothingProfile_t* profile) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDevicePowerSmoothingUpdatePresetProfileParam(nvmlDevice_t device, nvmlPowerSmoothingProfile_t* profile) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDevicePowerSmoothingSetState(nvmlDevice_t device, nvmlPowerSmoothingState_t* state) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetAddressingMode(nvmlDevice_t device, nvmlDeviceAddressingMode_t* mode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetRepairStatus(nvmlDevice_t device, nvmlRepairStatus_t* repairStatus) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetPowerMizerMode_v1(nvmlDevice_t device, nvmlDevicePowerMizerModes_v1_t* powerMizerMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetPowerMizerMode_v1(nvmlDevice_t device, nvmlDevicePowerMizerModes_v1_t* powerMizerMode) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetPdi(nvmlDevice_t device, nvmlPdi_t* pdi) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetHostname_v1(nvmlDevice_t device, nvmlHostname_v1_t* hostname) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetHostname_v1(nvmlDevice_t device, nvmlHostname_v1_t* hostname) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetNvLinkInfo(nvmlDevice_t device, nvmlNvLinkInfo_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceReadWritePRM_v1(nvmlDevice_t device, nvmlPRMTLV_v1_t* buffer) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfoByIdV(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstanceProfileInfo_v2_t* info) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts(nvmlDevice_t device, nvmlEccSramUniqueUncorrectedErrorCounts_t* errorCounts) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetUnrepairableMemoryFlag_v1(nvmlDevice_t device, nvmlUnrepairableMemoryStatus_v1_t* unrepairableMemoryStatus) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceReadPRMCounters_v1(nvmlDevice_t device, nvmlPRMCounterList_v1_t* counterList) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetRusdSettings_v1(nvmlDevice_t device, nvmlRusdSettings_v1_t* settings) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceVgpuForceGspUnload(nvmlDevice_t device) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetVgpuSchedulerState_v2(nvmlDevice_t device, nvmlVgpuSchedulerStateInfo_v2_t* pSchedulerStateInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGpuInstanceGetVgpuSchedulerState_v2(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerStateInfo_v2_t* pSchedulerStateInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceGetVgpuSchedulerLog_v2(nvmlDevice_t device, nvmlVgpuSchedulerLogInfo_v2_t* pSchedulerLogInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGpuInstanceGetVgpuSchedulerLog_v2(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerLogInfo_v2_t* pSchedulerLogInfo) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlDeviceSetVgpuSchedulerState_v2(nvmlDevice_t device, nvmlVgpuSchedulerState_v2_t* pSchedulerState) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
cdef nvmlReturn_t nvmlGpuInstanceSetVgpuSchedulerState_v2(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerState_v2_t* pSchedulerState) except?<nvmlReturn_t>_NVMLRETURN_T_INTERNAL_LOADING_ERROR nogil
