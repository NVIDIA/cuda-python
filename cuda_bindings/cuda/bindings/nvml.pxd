# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.9.1 to 13.1.1, generator version 0.3.1.dev1322+g646ce84ec. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cynvml cimport *


###############################################################################
# Types
###############################################################################

ctypedef nvmlDramEncryptionInfo_v1_t DramEncryptionInfo_v1
ctypedef nvmlMarginTemperature_v1_t MarginTemperature_v1
ctypedef nvmlFanSpeedInfo_v1_t FanSpeedInfo_v1
ctypedef nvmlDevicePerfModes_v1_t DevicePerfModes_v1
ctypedef nvmlDeviceCurrentClockFreqs_v1_t DeviceCurrentClockFreqs_v1
ctypedef nvmlVgpuHeterogeneousMode_v1_t VgpuHeterogeneousMode_v1
ctypedef nvmlVgpuPlacementId_v1_t VgpuPlacementId_v1
ctypedef nvmlVgpuRuntimeState_v1_t VgpuRuntimeState_v1
ctypedef nvmlConfComputeSetKeyRotationThresholdInfo_v1_t ConfComputeSetKeyRotationThresholdInfo_v1
ctypedef nvmlConfComputeGetKeyRotationThresholdInfo_v1_t ConfComputeGetKeyRotationThresholdInfo_v1
ctypedef nvmlSystemDriverBranchInfo_v1_t SystemDriverBranchInfo_v1
ctypedef nvmlTemperature_v1_t Temperature_v1
ctypedef nvmlDeviceCapabilities_v1_t DeviceCapabilities_v1
ctypedef nvmlPowerSmoothingProfile_v1_t PowerSmoothingProfile_v1
ctypedef nvmlPowerSmoothingState_v1_t PowerSmoothingState_v1
ctypedef nvmlPdi_v1_t Pdi_v1
ctypedef nvmlDevice_t Device
ctypedef nvmlGpuInstance_t GpuInstance
ctypedef nvmlUnit_t Unit
ctypedef nvmlEventSet_t EventSet
ctypedef nvmlSystemEventSet_t SystemEventSet
ctypedef nvmlComputeInstance_t ComputeInstance
ctypedef nvmlGpmSample_t GpmSample
ctypedef nvmlEccErrorCounts_t EccErrorCounts
ctypedef nvmlProcessInfo_v1_t ProcessInfo_v1
ctypedef nvmlProcessInfo_v2_t ProcessInfo_v2
ctypedef nvmlNvLinkUtilizationControl_t NvLinkUtilizationControl
ctypedef nvmlViolationTime_t ViolationTime
ctypedef nvmlUUIDValue_t UUIDValue
ctypedef nvmlVgpuPlacementList_v1_t VgpuPlacementList_v1
ctypedef nvmlNvLinkPowerThres_t NvLinkPowerThres
ctypedef nvmlGpuInstanceProfileInfo_t GpuInstanceProfileInfo
ctypedef nvmlGpuInstanceProfileInfo_v2_t GpuInstanceProfileInfo_v2
ctypedef nvmlComputeInstanceProfileInfo_t ComputeInstanceProfileInfo
ctypedef nvmlGpmSupport_t GpmSupport
ctypedef nvmlMask255_t Mask255
ctypedef nvmlHostname_v1_t Hostname_v1
ctypedef nvmlUnrepairableMemoryStatus_v1_t UnrepairableMemoryStatus_v1
ctypedef nvmlRusdSettings_v1_t RusdSettings_v1
ctypedef nvmlPowerValue_v2_t PowerValue_v2
ctypedef nvmlVgpuTypeMaxInstance_v1_t VgpuTypeMaxInstance_v1
ctypedef nvmlVgpuProcessUtilizationSample_t VgpuProcessUtilizationSample
ctypedef nvmlGpuFabricInfo_t GpuFabricInfo
ctypedef nvmlSystemEventSetCreateRequest_v1_t SystemEventSetCreateRequest_v1
ctypedef nvmlSystemEventSetFreeRequest_v1_t SystemEventSetFreeRequest_v1
ctypedef nvmlSystemRegisterEventRequest_v1_t SystemRegisterEventRequest_v1
ctypedef nvmlUUID_v1_t UUID_v1
ctypedef nvmlSystemEventSetWaitRequest_v1_t SystemEventSetWaitRequest_v1
ctypedef nvmlGpmMetric_t GpmMetric
ctypedef nvmlWorkloadPowerProfileInfo_v1_t WorkloadPowerProfileInfo_v1
ctypedef nvmlWorkloadPowerProfileCurrentProfiles_v1_t WorkloadPowerProfileCurrentProfiles_v1
ctypedef nvmlWorkloadPowerProfileRequestedProfiles_v1_t WorkloadPowerProfileRequestedProfiles_v1
ctypedef nvmlWorkloadPowerProfileUpdateProfiles_v1_t WorkloadPowerProfileUpdateProfiles_v1
ctypedef nvmlPRMTLV_v1_t PRMTLV_v1
ctypedef nvmlVgpuSchedulerSetState_t VgpuSchedulerSetState
ctypedef nvmlGpmMetricsGet_t GpmMetricsGet
ctypedef nvmlPRMCounterList_v1_t PRMCounterList_v1
ctypedef nvmlWorkloadPowerProfileProfilesInfo_v1_t WorkloadPowerProfileProfilesInfo_v1


###############################################################################
# Enum
###############################################################################

ctypedef nvmlBridgeChipType_t _BridgeChipType
ctypedef nvmlNvLinkUtilizationCountUnits_t _NvLinkUtilizationCountUnits
ctypedef nvmlNvLinkUtilizationCountPktTypes_t _NvLinkUtilizationCountPktTypes
ctypedef nvmlNvLinkCapability_t _NvLinkCapability
ctypedef nvmlNvLinkErrorCounter_t _NvLinkErrorCounter
ctypedef nvmlIntNvLinkDeviceType_t _IntNvLinkDeviceType
ctypedef nvmlGpuTopologyLevel_t _GpuTopologyLevel
ctypedef nvmlGpuP2PStatus_t _GpuP2PStatus
ctypedef nvmlGpuP2PCapsIndex_t _GpuP2PCapsIndex
ctypedef nvmlSamplingType_t _SamplingType
ctypedef nvmlPcieUtilCounter_t _PcieUtilCounter
ctypedef nvmlValueType_t _ValueType
ctypedef nvmlPerfPolicyType_t _PerfPolicyType
ctypedef nvmlThermalTarget_t _ThermalTarget
ctypedef nvmlThermalController_t _ThermalController
ctypedef nvmlCoolerControl_t _CoolerControl
ctypedef nvmlCoolerTarget_t _CoolerTarget
ctypedef nvmlUUIDType_t _UUIDType
ctypedef nvmlEnableState_t _EnableState
ctypedef nvmlBrandType_t _BrandType
ctypedef nvmlTemperatureThresholds_t _TemperatureThresholds
ctypedef nvmlTemperatureSensors_t _TemperatureSensors
ctypedef nvmlComputeMode_t _ComputeMode
ctypedef nvmlMemoryErrorType_t _MemoryErrorType
ctypedef nvmlNvlinkVersion_t _NvlinkVersion
ctypedef nvmlEccCounterType_t _EccCounterType
ctypedef nvmlClockType_t _ClockType
ctypedef nvmlClockId_t _ClockId
ctypedef nvmlDriverModel_t _DriverModel
ctypedef nvmlPstates_t _Pstates
ctypedef nvmlGpuOperationMode_t _GpuOperationMode
ctypedef nvmlInforomObject_t _InforomObject
ctypedef nvmlReturn_t _Return
ctypedef nvmlMemoryLocation_t _MemoryLocation
ctypedef nvmlPageRetirementCause_t _PageRetirementCause
ctypedef nvmlRestrictedAPI_t _RestrictedAPI
ctypedef nvmlGpuUtilizationDomainId_t _GpuUtilizationDomainId
ctypedef nvmlGpuVirtualizationMode_t _GpuVirtualizationMode
ctypedef nvmlHostVgpuMode_t _HostVgpuMode
ctypedef nvmlVgpuVmIdType_t _VgpuVmIdType
ctypedef nvmlVgpuGuestInfoState_t _VgpuGuestInfoState
ctypedef nvmlGridLicenseFeatureCode_t _GridLicenseFeatureCode
ctypedef nvmlVgpuCapability_t _VgpuCapability
ctypedef nvmlVgpuDriverCapability_t _VgpuDriverCapability
ctypedef nvmlDeviceVgpuCapability_t _DeviceVgpuCapability
ctypedef nvmlDeviceGpuRecoveryAction_t _DeviceGpuRecoveryAction
ctypedef nvmlFanState_t _FanState
ctypedef nvmlLedColor_t _LedColor
ctypedef nvmlEncoderType_t _EncoderType
ctypedef nvmlFBCSessionType_t _FBCSessionType
ctypedef nvmlDetachGpuState_t _DetachGpuState
ctypedef nvmlPcieLinkState_t _PcieLinkState
ctypedef nvmlClockLimitId_t _ClockLimitId
ctypedef nvmlVgpuVmCompatibility_t _VgpuVmCompatibility
ctypedef nvmlVgpuPgpuCompatibilityLimitCode_t _VgpuPgpuCompatibilityLimitCode
ctypedef nvmlGpmMetricId_t _GpmMetricId
ctypedef nvmlPowerProfileType_t _PowerProfileType
ctypedef nvmlDeviceAddressingModeType_t _DeviceAddressingModeType
ctypedef nvmlPRMCounterId_t _PRMCounterId
ctypedef nvmlPowerProfileOperation_t _PowerProfileOperation


###############################################################################
# Functions
###############################################################################

cpdef init_v2()
cpdef init_with_flags(unsigned int flags)
cpdef shutdown()
cpdef str error_string(int result)
cpdef str system_get_driver_version()
cpdef str system_get_nvml_version()
cpdef int system_get_cuda_driver_version() except *
cpdef int system_get_cuda_driver_version_v2() except 0
cpdef str system_get_process_name(unsigned int pid)
cpdef object system_get_hic_version()
cpdef unsigned int unit_get_count() except? 0
cpdef intptr_t unit_get_handle_by_index(unsigned int ind_ex) except? 0
cpdef object unit_get_unit_info(intptr_t unit)
cpdef object unit_get_led_state(intptr_t unit)
cpdef object unit_get_psu_info(intptr_t unit)
cpdef unsigned int unit_get_temperature(intptr_t unit, unsigned int type) except? 0
cpdef object unit_get_fan_speed_info(intptr_t unit)
cpdef unsigned int device_get_count_v2() except? 0
cpdef object device_get_attributes_v2(intptr_t device)
cpdef intptr_t device_get_handle_by_index_v2(unsigned int ind_ex) except? 0
cpdef intptr_t device_get_handle_by_serial(serial) except? 0
cpdef intptr_t device_get_handle_by_uuid(uuid) except? 0
cpdef intptr_t device_get_handle_by_pci_bus_id_v2(pci_bus_id) except? 0
cpdef str device_get_name(intptr_t device)
cpdef int device_get_brand(intptr_t device) except? -1
cpdef unsigned int device_get_index(intptr_t device) except? 0
cpdef str device_get_serial(intptr_t device)
cpdef unsigned int device_get_module_id(intptr_t device) except? 0
cpdef object device_get_c2c_mode_info_v(intptr_t device)
cpdef object device_get_memory_affinity(intptr_t device, unsigned int node_set_size, unsigned int scope)
cpdef object device_get_cpu_affinity_within_scope(intptr_t device, unsigned int cpu_set_size, unsigned int scope)
cpdef object device_get_cpu_affinity(intptr_t device, unsigned int cpu_set_size)
cpdef device_set_cpu_affinity(intptr_t device)
cpdef device_clear_cpu_affinity(intptr_t device)
cpdef unsigned int device_get_numa_node_id(intptr_t device) except? 0
cpdef int device_get_topology_common_ancestor(intptr_t device1, intptr_t device2) except? -1
cpdef int device_get_p2p_status(intptr_t device1, intptr_t device2, int p2p_ind_ex) except? -1
cpdef str device_get_uuid(intptr_t device)
cpdef unsigned int device_get_minor_number(intptr_t device) except? 0
cpdef str device_get_board_part_number(intptr_t device)
cpdef str device_get_inforom_version(intptr_t device, int object)
cpdef str device_get_inforom_image_version(intptr_t device)
cpdef unsigned int device_get_inforom_configuration_checksum(intptr_t device) except? 0
cpdef device_validate_inforom(intptr_t device)
cpdef tuple device_get_last_bbx_flush_time(intptr_t device)
cpdef int device_get_display_mode(intptr_t device) except? -1
cpdef int device_get_display_active(intptr_t device) except? -1
cpdef int device_get_persistence_mode(intptr_t device) except? -1
cpdef object device_get_pci_info_ext(intptr_t device)
cpdef object device_get_pci_info_v3(intptr_t device)
cpdef unsigned int device_get_max_pcie_link_generation(intptr_t device) except? 0
cpdef unsigned int device_get_gpu_max_pcie_link_generation(intptr_t device) except? 0
cpdef unsigned int device_get_max_pcie_link_width(intptr_t device) except? 0
cpdef unsigned int device_get_curr_pcie_link_generation(intptr_t device) except? 0
cpdef unsigned int device_get_curr_pcie_link_width(intptr_t device) except? 0
cpdef unsigned int device_get_pcie_throughput(intptr_t device, int counter) except? 0
cpdef unsigned int device_get_pcie_replay_counter(intptr_t device) except? 0
cpdef unsigned int device_get_clock_info(intptr_t device, int type) except? 0
cpdef unsigned int device_get_max_clock_info(intptr_t device, int type) except? 0
cpdef int device_get_gpc_clk_vf_offset(intptr_t device) except? 0
cpdef unsigned int device_get_clock(intptr_t device, int clock_type, int clock_id) except? 0
cpdef unsigned int device_get_max_customer_boost_clock(intptr_t device, int clock_type) except? 0
cpdef object device_get_supported_memory_clocks(intptr_t device)
cpdef object device_get_supported_graphics_clocks(intptr_t device, unsigned int memory_clock_m_hz)
cpdef tuple device_get_auto_boosted_clocks_enabled(intptr_t device)
cpdef unsigned int device_get_fan_speed(intptr_t device) except? 0
cpdef unsigned int device_get_fan_speed_v2(intptr_t device, unsigned int fan) except? 0
cpdef unsigned int device_get_target_fan_speed(intptr_t device, unsigned int fan) except? 0
cpdef tuple device_get_min_max_fan_speed(intptr_t device)
cpdef unsigned int device_get_fan_control_policy_v2(intptr_t device, unsigned int fan) except *
cpdef unsigned int device_get_num_fans(intptr_t device) except? 0
cpdef object device_get_cooler_info(intptr_t device)
cpdef unsigned int device_get_temperature_threshold(intptr_t device, int threshold_type) except? 0
cpdef object device_get_thermal_settings(intptr_t device, unsigned int sensor_ind_ex)
cpdef int device_get_performance_state(intptr_t device) except? -1
cpdef unsigned long long device_get_current_clocks_event_reasons(intptr_t device) except? 0
cpdef unsigned long long device_get_supported_clocks_event_reasons(intptr_t device) except? 0
cpdef int device_get_power_state(intptr_t device) except? -1
cpdef object device_get_dynamic_pstates_info(intptr_t device)
cpdef int device_get_mem_clk_vf_offset(intptr_t device) except? 0
cpdef tuple device_get_min_max_clock_of_p_state(intptr_t device, int type, int pstate)
cpdef tuple device_get_gpc_clk_min_max_vf_offset(intptr_t device)
cpdef tuple device_get_mem_clk_min_max_vf_offset(intptr_t device)
cpdef device_set_clock_offsets(intptr_t device, intptr_t info)
cpdef unsigned int device_get_power_management_limit(intptr_t device) except? 0
cpdef tuple device_get_power_management_limit_constraints(intptr_t device)
cpdef unsigned int device_get_power_management_default_limit(intptr_t device) except? 0
cpdef unsigned int device_get_power_usage(intptr_t device) except? 0
cpdef unsigned long long device_get_total_energy_consumption(intptr_t device) except? 0
cpdef unsigned int device_get_enforced_power_limit(intptr_t device) except? 0
cpdef tuple device_get_gpu_operation_mode(intptr_t device)
cpdef object device_get_memory_info_v2(intptr_t device)
cpdef int device_get_compute_mode(intptr_t device) except? -1
cpdef tuple device_get_cuda_compute_capability(intptr_t device)
cpdef tuple device_get_ecc_mode(intptr_t device)
cpdef int device_get_default_ecc_mode(intptr_t device) except? -1
cpdef unsigned int device_get_board_id(intptr_t device) except? 0
cpdef unsigned int device_get_multi_gpu_board(intptr_t device) except? 0
cpdef unsigned long long device_get_total_ecc_errors(intptr_t device, int error_type, int counter_type) except? 0
cpdef unsigned long long device_get_memory_error_counter(intptr_t device, int error_type, int counter_type, int location_type) except? 0
cpdef object device_get_utilization_rates(intptr_t device)
cpdef tuple device_get_encoder_utilization(intptr_t device)
cpdef unsigned int device_get_encoder_capacity(intptr_t device, int encoder_query_type) except? 0
cpdef tuple device_get_encoder_stats(intptr_t device)
cpdef object device_get_encoder_sessions(intptr_t device)
cpdef tuple device_get_decoder_utilization(intptr_t device)
cpdef tuple device_get_jpg_utilization(intptr_t device)
cpdef tuple device_get_ofa_utilization(intptr_t device)
cpdef object device_get_fbc_stats(intptr_t device)
cpdef object device_get_fbc_sessions(intptr_t device)
cpdef tuple device_get_driver_model_v2(intptr_t device)
cpdef str device_get_vbios_version(intptr_t device)
cpdef object device_get_bridge_chip_info(intptr_t device)
cpdef object device_get_compute_running_processes_v3(intptr_t device)
cpdef object device_get_mps_compute_running_processes_v3(intptr_t device)
cpdef int device_on_same_board(intptr_t device1, intptr_t device2) except? 0
cpdef int device_get_api_restriction(intptr_t device, int api_type) except? -1
cpdef object device_get_bar1_memory_info(intptr_t device)
cpdef unsigned int device_get_irq_num(intptr_t device) except? 0
cpdef unsigned int device_get_num_gpu_cores(intptr_t device) except? 0
cpdef unsigned int device_get_power_source(intptr_t device) except *
cpdef unsigned int device_get_memory_bus_width(intptr_t device) except? 0
cpdef unsigned int device_get_pcie_link_max_speed(intptr_t device) except? 0
cpdef unsigned int device_get_pcie_speed(intptr_t device) except? 0
cpdef unsigned int device_get_adaptive_clock_info_status(intptr_t device) except? 0
cpdef unsigned int device_get_bus_type(intptr_t device) except? 0
cpdef object system_get_conf_compute_capabilities()
cpdef object system_get_conf_compute_state()
cpdef object device_get_conf_compute_mem_size_info(intptr_t device)
cpdef unsigned int system_get_conf_compute_gpus_ready_state() except? 0
cpdef object device_get_conf_compute_protected_memory_usage(intptr_t device)
cpdef object device_get_conf_compute_gpu_certificate(intptr_t device)
cpdef device_set_conf_compute_unprotected_mem_size(intptr_t device, unsigned long long size_ki_b)
cpdef system_set_conf_compute_gpus_ready_state(unsigned int is_accepting_work)
cpdef object system_get_conf_compute_settings()
cpdef char device_get_gsp_firmware_version(intptr_t device) except? 0
cpdef tuple device_get_gsp_firmware_mode(intptr_t device)
cpdef object device_get_sram_ecc_error_status(intptr_t device)
cpdef int device_get_accounting_mode(intptr_t device) except? -1
cpdef object device_get_accounting_stats(intptr_t device, unsigned int pid)
cpdef object device_get_accounting_pids(intptr_t device)
cpdef unsigned int device_get_accounting_buffer_size(intptr_t device) except? 0
cpdef object device_get_retired_pages(intptr_t device, int cause)
cpdef int device_get_retired_pages_pending_status(intptr_t device) except? -1
cpdef tuple device_get_remapped_rows(intptr_t device)
cpdef object device_get_row_remapper_histogram(intptr_t device)
cpdef unsigned int device_get_architecture(intptr_t device) except? 0
cpdef object device_get_clk_mon_status(intptr_t device)
cpdef object device_get_process_utilization(intptr_t device, unsigned long long last_seen_time_stamp)
cpdef unit_set_led_state(intptr_t unit, int color)
cpdef device_set_persistence_mode(intptr_t device, int mode)
cpdef device_set_compute_mode(intptr_t device, int mode)
cpdef device_set_ecc_mode(intptr_t device, int ecc)
cpdef device_clear_ecc_error_counts(intptr_t device, int counter_type)
cpdef device_set_driver_model(intptr_t device, int driver_model, unsigned int flags)
cpdef device_set_gpu_locked_clocks(intptr_t device, unsigned int min_gpu_clock_m_hz, unsigned int max_gpu_clock_m_hz)
cpdef device_reset_gpu_locked_clocks(intptr_t device)
cpdef device_set_memory_locked_clocks(intptr_t device, unsigned int min_mem_clock_m_hz, unsigned int max_mem_clock_m_hz)
cpdef device_reset_memory_locked_clocks(intptr_t device)
cpdef device_set_auto_boosted_clocks_enabled(intptr_t device, int enabled)
cpdef device_set_default_auto_boosted_clocks_enabled(intptr_t device, int enabled, unsigned int flags)
cpdef device_set_default_fan_speed_v2(intptr_t device, unsigned int fan)
cpdef device_set_fan_control_policy(intptr_t device, unsigned int fan, unsigned int policy)
cpdef device_set_gpu_operation_mode(intptr_t device, int mode)
cpdef device_set_api_restriction(intptr_t device, int api_type, int is_restricted)
cpdef device_set_fan_speed_v2(intptr_t device, unsigned int fan, unsigned int speed)
cpdef device_set_accounting_mode(intptr_t device, int mode)
cpdef device_clear_accounting_pids(intptr_t device)
cpdef int device_get_nvlink_state(intptr_t device, unsigned int link) except? -1
cpdef unsigned int device_get_nvlink_version(intptr_t device, unsigned int link) except? 0
cpdef unsigned int device_get_nvlink_capability(intptr_t device, unsigned int link, int capability) except? 0
cpdef object device_get_nvlink_remote_pci_info_v2(intptr_t device, unsigned int link)
cpdef unsigned long long device_get_nvlink_error_counter(intptr_t device, unsigned int link, int counter) except? 0
cpdef device_reset_nvlink_error_counters(intptr_t device, unsigned int link)
cpdef int device_get_nvlink_remote_device_type(intptr_t device, unsigned int link) except? -1
cpdef system_set_nvlink_bw_mode(unsigned int nvlink_bw_mode)
cpdef unsigned int system_get_nvlink_bw_mode() except? 0
cpdef object device_get_nvlink_supported_bw_modes(intptr_t device)
cpdef object device_get_nvlink_bw_mode(intptr_t device)
cpdef device_set_nvlink_bw_mode(intptr_t device, intptr_t set_bw_mode)
cpdef intptr_t event_set_create() except? 0
cpdef device_register_events(intptr_t device, unsigned long long event_types, intptr_t set)
cpdef unsigned long long device_get_supported_event_types(intptr_t device) except? 0
cpdef object event_set_wait_v2(intptr_t set, unsigned int timeoutms)
cpdef event_set_free(intptr_t set)
cpdef device_modify_drain_state(intptr_t pci_info, int new_state)
cpdef int device_query_drain_state(intptr_t pci_info) except? -1
cpdef device_remove_gpu_v2(intptr_t pci_info, int gpu_state, int link_state)
cpdef device_discover_gpus(intptr_t pci_info)
cpdef int device_get_virtualization_mode(intptr_t device) except? -1
cpdef int device_get_host_vgpu_mode(intptr_t device) except? -1
cpdef device_set_virtualization_mode(intptr_t device, int virtual_mode)
cpdef unsigned long long vgpu_type_get_gsp_heap_size(unsigned int vgpu_type_id) except? 0
cpdef unsigned long long vgpu_type_get_fb_reservation(unsigned int vgpu_type_id) except? 0
cpdef device_set_vgpu_capabilities(intptr_t device, int capability, int state)
cpdef object device_get_grid_licensable_features_v4(intptr_t device)
cpdef unsigned int get_vgpu_driver_capabilities(int capability) except? 0
cpdef unsigned int device_get_vgpu_capabilities(intptr_t device, int capability) except? 0
cpdef str vgpu_type_get_class(unsigned int vgpu_type_id)
cpdef unsigned int vgpu_type_get_gpu_instance_profile_id(unsigned int vgpu_type_id) except? 0
cpdef tuple vgpu_type_get_device_id(unsigned int vgpu_type_id)
cpdef unsigned long long vgpu_type_get_framebuffer_size(unsigned int vgpu_type_id) except? 0
cpdef unsigned int vgpu_type_get_num_display_heads(unsigned int vgpu_type_id) except? 0
cpdef tuple vgpu_type_get_resolution(unsigned int vgpu_type_id, unsigned int display_ind_ex)
cpdef str vgpu_type_get_license(unsigned int vgpu_type_id)
cpdef unsigned int vgpu_type_get_frame_rate_limit(unsigned int vgpu_type_id) except? 0
cpdef unsigned int vgpu_type_get_max_instances(intptr_t device, unsigned int vgpu_type_id) except? 0
cpdef unsigned int vgpu_type_get_max_instances_per_vm(unsigned int vgpu_type_id) except? 0
cpdef object vgpu_type_get_bar1_info(unsigned int vgpu_type_id)
cpdef str vgpu_instance_get_uuid(unsigned int vgpu_instance)
cpdef str vgpu_instance_get_vm_driver_version(unsigned int vgpu_instance)
cpdef unsigned long long vgpu_instance_get_fb_usage(unsigned int vgpu_instance) except? 0
cpdef unsigned int vgpu_instance_get_license_status(unsigned int vgpu_instance) except? 0
cpdef unsigned int vgpu_instance_get_type(unsigned int vgpu_instance) except? 0
cpdef unsigned int vgpu_instance_get_frame_rate_limit(unsigned int vgpu_instance) except? 0
cpdef int vgpu_instance_get_ecc_mode(unsigned int vgpu_instance) except? -1
cpdef unsigned int vgpu_instance_get_encoder_capacity(unsigned int vgpu_instance) except? 0
cpdef vgpu_instance_set_encoder_capacity(unsigned int vgpu_instance, unsigned int encoder_capacity)
cpdef tuple vgpu_instance_get_encoder_stats(unsigned int vgpu_instance)
cpdef object vgpu_instance_get_encoder_sessions(unsigned int vgpu_instance)
cpdef object vgpu_instance_get_fbc_stats(unsigned int vgpu_instance)
cpdef object vgpu_instance_get_fbc_sessions(unsigned int vgpu_instance)
cpdef unsigned int vgpu_instance_get_gpu_instance_id(unsigned int vgpu_instance) except? 0
cpdef str vgpu_instance_get_gpu_pci_id(unsigned int vgpu_instance)
cpdef unsigned int vgpu_type_get_capabilities(unsigned int vgpu_type_id, int capability) except? 0
cpdef str vgpu_instance_get_mdev_uuid(unsigned int vgpu_instance)
cpdef gpu_instance_set_vgpu_scheduler_state(intptr_t gpu_instance, intptr_t p_scheduler)
cpdef object gpu_instance_get_vgpu_scheduler_state(intptr_t gpu_instance)
cpdef object gpu_instance_get_vgpu_scheduler_log(intptr_t gpu_instance)
cpdef str device_get_pgpu_metadata_string(intptr_t device)
cpdef object device_get_vgpu_scheduler_log(intptr_t device)
cpdef object device_get_vgpu_scheduler_state(intptr_t device)
cpdef object device_get_vgpu_scheduler_capabilities(intptr_t device)
cpdef device_set_vgpu_scheduler_state(intptr_t device, intptr_t p_scheduler_state)
cpdef set_vgpu_version(intptr_t vgpu_version)
cpdef tuple device_get_vgpu_process_utilization(intptr_t device, unsigned long long last_seen_time_stamp)
cpdef int vgpu_instance_get_accounting_mode(unsigned int vgpu_instance) except? -1
cpdef object vgpu_instance_get_accounting_pids(unsigned int vgpu_instance)
cpdef object vgpu_instance_get_accounting_stats(unsigned int vgpu_instance, unsigned int pid)
cpdef vgpu_instance_clear_accounting_pids(unsigned int vgpu_instance)
cpdef object vgpu_instance_get_license_info_v2(unsigned int vgpu_instance)
cpdef unsigned int get_excluded_device_count() except? 0
cpdef object get_excluded_device_info_by_index(unsigned int ind_ex)
cpdef int device_set_mig_mode(intptr_t device, unsigned int mode) except? -1
cpdef tuple device_get_mig_mode(intptr_t device)
cpdef object device_get_gpu_instance_possible_placements_v2(intptr_t device, unsigned int profile_id)
cpdef unsigned int device_get_gpu_instance_remaining_capacity(intptr_t device, unsigned int profile_id) except? 0
cpdef intptr_t device_create_gpu_instance(intptr_t device, unsigned int profile_id) except? 0
cpdef intptr_t device_create_gpu_instance_with_placement(intptr_t device, unsigned int profile_id, intptr_t placement) except? 0
cpdef gpu_instance_destroy(intptr_t gpu_instance)
cpdef intptr_t device_get_gpu_instance_by_id(intptr_t device, unsigned int id) except? 0
cpdef object gpu_instance_get_info(intptr_t gpu_instance)
cpdef object gpu_instance_get_compute_instance_profile_info_v(intptr_t gpu_instance, unsigned int profile, unsigned int eng_profile)
cpdef unsigned int gpu_instance_get_compute_instance_remaining_capacity(intptr_t gpu_instance, unsigned int profile_id) except? 0
cpdef object gpu_instance_get_compute_instance_possible_placements(intptr_t gpu_instance, unsigned int profile_id)
cpdef intptr_t gpu_instance_create_compute_instance(intptr_t gpu_instance, unsigned int profile_id) except? 0
cpdef intptr_t gpu_instance_create_compute_instance_with_placement(intptr_t gpu_instance, unsigned int profile_id, intptr_t placement) except? 0
cpdef compute_instance_destroy(intptr_t compute_instance)
cpdef intptr_t gpu_instance_get_compute_instance_by_id(intptr_t gpu_instance, unsigned int id) except? 0
cpdef object compute_instance_get_info_v2(intptr_t compute_instance)
cpdef unsigned int device_is_mig_device_handle(intptr_t device) except? 0
cpdef unsigned int device_get_gpu_instance_id(intptr_t device) except? 0
cpdef unsigned int device_get_compute_instance_id(intptr_t device) except? 0
cpdef unsigned int device_get_max_mig_device_count(intptr_t device) except? 0
cpdef intptr_t device_get_mig_device_handle_by_index(intptr_t device, unsigned int ind_ex) except? 0
cpdef intptr_t device_get_device_handle_from_mig_device_handle(intptr_t mig_device) except? 0
cpdef device_power_smoothing_activate_preset_profile(intptr_t device, intptr_t profile)
cpdef device_power_smoothing_update_preset_profile_param(intptr_t device, intptr_t profile)
cpdef device_power_smoothing_set_state(intptr_t device, intptr_t state)
cpdef object device_get_addressing_mode(intptr_t device)
cpdef object device_get_repair_status(intptr_t device)
cpdef object device_get_power_mizer_mode_v1(intptr_t device)
cpdef device_set_power_mizer_mode_v1(intptr_t device, intptr_t power_mizer_mode)
