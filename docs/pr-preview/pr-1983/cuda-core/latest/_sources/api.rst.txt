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


Graphics interoperability
-------------------------

.. autosummary::
   :toctree: generated/

   :template: autosummary/cyclass.rst

   GraphicsResource


Tensor Memory Accelerator (TMA)
-------------------------------

.. autosummary::
   :toctree: generated/

   :template: autosummary/cyclass.rst

   TensorMapDescriptor

   :template: dataclass.rst

   TensorMapDescriptorOptions


CUDA compilation toolchain
--------------------------

.. autosummary::
   :toctree: generated/

   :template: autosummary/cyclass.rst

   Program
   Linker
   ObjectCode
   Kernel

   :template: dataclass.rst

   ProgramOptions
   LinkerOptions


CUDA process checkpointing
--------------------------

The :mod:`cuda.core.checkpoint` module wraps the CUDA driver process
checkpoint APIs. These APIs are intended for Linux process checkpoint and
restore workflows, and require a CUDA driver with checkpoint API support and
a ``cuda-bindings`` version that exposes those driver entry points.

Checkpointing is typically driven by a coordinator process acting on a target
CUDA process, similar to attaching a debugger or sending a signal. The target
process is identified by process ID. Linux and the CUDA driver enforce process
permissions; checkpointing another user's process may require elevated
permissions such as ``CAP_SYS_PTRACE`` or administrator privileges.

The CUDA checkpoint APIs prepare CUDA-managed GPU state for process-level
checkpoint and restore. They do not capture the CPU process image by
themselves; full process checkpoint workflows still need a CPU-side process
checkpointing tool such as CRIU. A minimal coordinator-side sequence looks like
this:

.. code-block:: python

   import os

   from cuda.core import checkpoint

   target_pid = os.getpid()  # or the PID of another CUDA process
   process = checkpoint.Process(target_pid)
   process.lock(timeout_ms=5000)
   process.checkpoint()

   # Capture or restore the CPU process image outside cuda.core.

   process.restore()
   process.unlock()

``Process.state`` returns one of ``"running"``, ``"locked"``,
``"checkpointed"``, or ``"failed"``. Restore may optionally remap GPUs by
passing ``gpu_mapping`` from each checkpointed GPU UUID to the GPU UUID that
should be used during restore. For migration workflows, provide mappings for
every CUDA-visible GPU. The mapping may use ``CUuuid`` objects or the UUID
strings returned by :attr:`Device.uuid`. A successful restore returns the
process to the locked state; call ``Process.unlock`` after restore to allow
CUDA API calls to resume.

The CUDA driver requires restore to run from the process restore thread.
Use ``Process.restore_thread_id`` to discover that thread before calling
``Process.restore`` from a checkpoint coordinator. Restore also requires
persistence mode to be enabled or ``cuInit`` to have been called before
execution.

.. autosummary::
   :toctree: generated/

   :template: class.rst

   checkpoint.Process


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
   system.ClockType
   system.CoolerControl
   system.CoolerTarget
   system.DeviceArch
   system.EventType
   system.FanControlPolicy
   system.FieldId
   system.InforomObject
   system.NvlinkVersion
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

.. module:: cuda.core.utils

Utility functions
-----------------

.. autosummary::
   :toctree: generated/

   args_viewable_as_strided_memory

   :template: autosummary/cyclass.rst

   StridedMemoryView
