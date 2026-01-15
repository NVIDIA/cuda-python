.. SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
   system.BAR1MemoryInfo
   system.BrandType
   system.DeviceArchitecture
   system.DeviceAttributes
   system.DeviceEvents
   system.EventData
   system.EventType
   system.FieldId
   system.FieldValue
   system.FieldValues
   system.GpuP2PCapsIndex
   system.GpuP2PStatus
   system.GpuTopologyLevel
   system.InforomInfo
   system.InforomObject
   system.MemoryInfo
   system.PcieUtilCounter
   system.PciInfo
   system.RepairStatus

.. module:: cuda.core.utils

Utility functions
-----------------

.. autosummary::
   :toctree: generated/

   args_viewable_as_strided_memory

   :template: autosummary/cyclass.rst

   StridedMemoryView
