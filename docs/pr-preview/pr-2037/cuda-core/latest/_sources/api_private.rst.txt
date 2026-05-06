.. SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

:orphan:

.. This page is to generate documentation for private classes exposed to users,
   i.e., users cannot instantiate them but may use their properties or methods
   via returned values from public APIs. These classes must be referred in
   public APIs returning their instances.

.. currentmodule:: cuda.core

CUDA runtime
------------

.. autosummary::
   :toctree: generated/

   _module.KernelAttributes
   _module.KernelOccupancy
   _module.MaxPotentialBlockSizeOccupancyResult
   _module.ParamInfo
   enums.CompilerBackendType
   enums.GraphConditionalType
   enums.GraphMemoryType
   enums.ManagedMemoryLocationType
   enums.ObjectCodeFormatType
   enums.PCHStatusType
   enums.SourceCodeType
   enums.VirtualMemoryAccessType
   enums.VirtualMemoryAllocationType
   enums.VirtualMemoryGranularityType
   enums.VirtualMemoryHandleType
   enums.VirtualMemoryLocationType
   typing.DevicePointerType
   typing.DeviceResourcesType
   typing.IsStreamType
   typing.ProcessStateType

   :template: autosummary/cyclass.rst

   _device.DeviceProperties
   _device_resources.DeviceResources
   _memory._ipc.IPCAllocationHandle
   _memory._ipc.IPCBufferDescriptor


CUDA graphs
-----------

.. autosummary::
   :toctree: generated/

   graph._adjacency_set_proxy.AdjacencySetProxy


CUDA protocols
--------------

.. autosummary::
   :toctree: generated/
   :template: protocol.rst

   typing.IsStreamType

NVML
----

.. autosummary::
   :toctree: generated/
   :template: autosummary/cyclass.rst

   system._device.BAR1MemoryInfo
   system._device.ClockInfo
   system._device.ClockOffsets
   system._device.CoolerInfo
   system._device.DeviceAttributes
   system._device.DeviceEvents
   system._device.EventData
   system._device.FanInfo
   system._device.FieldValue
   system._device.FieldValues
   system._device.GpuDynamicPstatesInfo
   system._device.GpuDynamicPstatesUtilization
   system._device.InforomInfo
   system._device.MemoryInfo
   system._device.MigInfo
   system._device.PciInfo
   system._device.ProcessInfo
   system._device.RepairStatus
   system._device.Temperature
   system._device.ThermalSensor
   system._device.ThermalSettings
   system._device.Utilization
   system._system_events.RegisteredSystemEvents
   system._system_events.SystemEvent
   system._system_events.SystemEvents

.. These are not technically private, but are included here to avoid cluttering the main API reference.

.. autosummary::
   :toctree: generated/

   system.enums.AddressingMode
   system.enums.AffinityScope
   system.enums.ClockId
   system.enums.ClocksEventReasons
   system.enums.ClockType
   system.enums.CoolerControl
   system.enums.CoolerTarget
   system.enums.DeviceArch
   system.enums.EventType
   system.enums.FanControlPolicy
   system.enums.FieldId
   system.enums.GpuP2PCapsIndex
   system.enums.GpuP2PStatus
   system.enums.GpuTopologyLevel
   system.enums.InforomObject
   system.enums.SystemEventType
   system.enums.TemperatureThresholds
   system.enums.ThermalController
   system.enums.ThermalTarget
