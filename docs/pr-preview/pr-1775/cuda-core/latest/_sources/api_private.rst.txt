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

   typing.DevicePointerT
   _memory._virtual_memory_resource.VirtualMemoryAllocationTypeT
   _memory._virtual_memory_resource.VirtualMemoryLocationTypeT
   _memory._virtual_memory_resource.VirtualMemoryGranularityT
   _memory._virtual_memory_resource.VirtualMemoryAccessTypeT
   _memory._virtual_memory_resource.VirtualMemoryHandleTypeT
   _module.KernelAttributes
   _module.KernelOccupancy
   _module.ParamInfo
   _module.MaxPotentialBlockSizeOccupancyResult

   :template: autosummary/cyclass.rst

   _device.DeviceProperties
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

   typing.IsStreamT

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
   system._device.GpuP2PCapsIndex
   system._device.GpuP2PStatus
   system._device.GpuTopologyLevel
   system._device.InforomInfo
   system._device.MemoryInfo
   system._device.MigInfo
   system._device.NvlinkInfo
   system._device.PciInfo
   system._device.ProcessInfo
   system._device.RepairStatus
   system._device.Temperature
   system._device.ThermalSensor
   system._device.ThermalSettings
   system._system_events.RegisteredSystemEvents
   system._system_events.SystemEvent
   system._system_events.SystemEvents
