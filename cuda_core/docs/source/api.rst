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


Devices and execution
---------------------

.. autosummary::
   :toctree: generated/

   Device
   launch

   :template: autosummary/cyclass.rst

   Stream
   Event

   :template: dataclass.rst

   StreamOptions
   EventOptions
   LaunchConfig

.. data:: LEGACY_DEFAULT_STREAM

   The legacy default CUDA stream. All devices share the same legacy default
   stream, and work launched on it is not concurrent with work on any other
   stream.

.. data:: PER_THREAD_DEFAULT_STREAM

   The per-thread default CUDA stream. Each host thread has its own per-thread
   default stream, and work launched on it can execute concurrently with work
   on other non-blocking streams.


Memory management
-----------------

.. autosummary::
   :toctree: generated/

   :template: autosummary/cyclass.rst

   Buffer
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
   VirtualMemoryResourceOptions


CUDA graphs
-----------

A CUDA graph captures a set of GPU operations and their dependencies,
allowing them to be defined once and launched repeatedly with minimal
CPU overhead. Graphs can be constructed in two ways:
:class:`~graph.GraphBuilder` captures operations from a stream, while
:class:`~graph.GraphDef` builds a graph explicitly by adding nodes and
edges. Both produce an executable :class:`~graph.Graph` that can be
launched on a :class:`Stream`.

.. autosummary::
   :toctree: generated/

   graph.Graph
   graph.GraphBuilder
   graph.GraphDef

   :template: autosummary/cyclass.rst

   graph.GraphNode
   graph.Condition

   :template: dataclass.rst

   graph.GraphAllocOptions
   graph.GraphCompleteOptions
   graph.GraphDebugPrintOptions

Node types
``````````

Every graph node is a subclass of :class:`~graph.GraphNode`, which
provides the common interface (dependencies, successors, destruction).
Each subclass exposes attributes unique to its operation type.

.. autosummary::
   :toctree: generated/

   :template: autosummary/cyclass.rst

   graph.EmptyNode
   graph.KernelNode
   graph.AllocNode
   graph.FreeNode
   graph.MemsetNode
   graph.MemcpyNode
   graph.ChildGraphNode
   graph.EventRecordNode
   graph.EventWaitNode
   graph.HostCallbackNode
   graph.ConditionalNode
   graph.IfNode
   graph.IfElseNode
   graph.WhileNode
   graph.SwitchNode


.. module:: cuda.core.managed_memory

Managed memory
--------------

.. autosummary::
   :toctree: generated/

   advise
   prefetch
   discard_prefetch

.. module:: cuda.core
   :no-index:

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

Basic functions
```````````````

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

Events
``````

.. autosummary::
   :toctree: generated/

   system.register_events
   system.RegisteredSystemEvents
   system.SystemEvent
   system.SystemEvents
   system.SystemEventType

Enums
`````

.. autosummary::
   :toctree: generated/

   system.AddressingMode
   system.AffinityScope
   system.BrandType
   system.ClockId
   system.ClocksEventReasons
   system.CoolerControl
   system.CoolerTarget
   system.DeviceArch
   system.EventType
   system.FanControlPolicy
   system.FieldId
   system.InforomObject
   system.PcieUtilCounter
   system.Pstates
   system.TemperatureSensors
   system.TemperatureThresholds
   system.ThermalController
   system.ThermalTarget

Types
`````

.. autosummary::
   :toctree: generated/

   :template: autosummary/cyclass.rst

   system.Device
   system.BAR1MemoryInfo
   system.ClockInfo
   system.ClockOffsets
   system.ClockType
   system.CoolerInfo
   system.DeviceAttributes
   system.DeviceEvents
   system.EventData
   system.FanInfo
   system.FieldValue
   system.FieldValues
   system.GpuDynamicPstatesInfo
   system.GpuDynamicPstatesUtilization
   system.GpuP2PCapsIndex
   system.GpuP2PStatus
   system.GpuTopologyLevel
   system.InforomInfo
   system.MemoryInfo
   system.PciInfo
   system.RepairStatus
   system.Temperature
   system.ThermalSensor
   system.ThermalSettings

.. module:: cuda.core.utils

Utility functions
-----------------

.. autosummary::
   :toctree: generated/

   args_viewable_as_strided_memory

   :template: autosummary/cyclass.rst

   StridedMemoryView
