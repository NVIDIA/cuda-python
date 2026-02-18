.. SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. module:: cuda.core

``cuda.core`` API Reference
===========================

This is the main API reference for ``cuda.core``. The package has not yet
reached version 1.0.0, and APIs may change between minor versions, possibly
without deprecation warnings. Once version 1.0.0 is released, APIs will
be considered stable and will follow semantic versioning with appropriate
deprecation periods for breaking changes.


CUDA runtime
------------

.. autosummary::
   :toctree: generated/

   Device
   Graph
   GraphBuilder
   launch

   :template: autosummary/cyclass.rst

   Buffer
   Stream
   Event
   MemoryResource
   DeviceMemoryResource
   GraphMemoryResource
   PinnedMemoryResource
   ManagedMemoryResource
   LegacyPinnedMemoryResource
   VirtualMemoryResource

   :template: dataclass.rst

   DeviceMemoryResourceOptions
   PinnedMemoryResourceOptions
   ManagedMemoryResourceOptions
   EventOptions
   GraphCompleteOptions
   GraphDebugPrintOptions
   StreamOptions
   LaunchConfig
   VirtualMemoryResourceOptions


CUDA compilation toolchain
--------------------------

.. autosummary::
   :toctree: generated/

   Program
   Linker
   ObjectCode
   Kernel

   :template: dataclass.rst

   ProgramOptions
   LinkerOptions


CUDA system information and NVIDIA Management Library (NVML)
------------------------------------------------------------

.. autosummary::
   :toctree: generated/

   system.get_driver_version
   system.get_driver_version_full
   system.get_driver_branch
   system.get_num_devices
   system.get_nvml_version
   system.get_process_name
   system.get_topology_common_ancestor
   system.get_p2p_status

   system.register_events
   system.RegisteredSystemEvents
   system.SystemEvent
   system.SystemEvents
   system.SystemEventType

   :template: autosummary/cyclass.rst

   system.Device
   system.AddressingMode
   system.AffinityScope
   system.BAR1MemoryInfo
   system.BrandType
   system.ClockId
   system.ClockInfo
   system.ClockOffsets
   system.ClocksEventReasons
   system.ClockType
   system.CoolerControl
   system.CoolerInfo
   system.CoolerTarget
   system.DeviceArch
   system.DeviceAttributes
   system.DeviceEvents
   system.EventData
   system.EventType
   system.FanControlPolicy
   system.FanInfo
   system.FieldId
   system.FieldValue
   system.FieldValues
   system.GpuDynamicPstatesInfo
   system.GpuDynamicPstatesUtilization
   system.GpuP2PCapsIndex
   system.GpuP2PStatus
   system.GpuTopologyLevel
   system.InforomInfo
   system.InforomObject
   system.MemoryInfo
   system.PcieUtilCounter
   system.PciInfo
   system.Pstates
   system.RepairStatus
   system.Temperature
   system.TemperatureSensors
   system.TemperatureThresholds
   system.ThermalController
   system.ThermalSensor
   system.ThermalSettings
   system.ThermalTarget

.. module:: cuda.core.utils

Utility functions
-----------------

.. autosummary::
   :toctree: generated/

   args_viewable_as_strided_memory
   make_aligned_dtype

   :template: autosummary/cyclass.rst

   StridedMemoryView
