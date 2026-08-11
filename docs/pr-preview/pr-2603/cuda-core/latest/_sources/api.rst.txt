.. SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. module:: cuda.core

``cuda.core`` API Reference
===========================

This is the main API reference for ``cuda.core``. As of version 1.0.0, all
APIs are considered stable and follow `Semantic Versioning <https://semver.org/>`_
with appropriate deprecation periods for breaking changes. See the
:doc:`support policy <support>` for details.


Devices and execution
---------------------

.. autosummary::
   :toctree: generated/

   Device
   Host
   launch

   :template: autosummary/cyclass.rst

   Stream
   Event
   Context
   SMResource
   WorkqueueResource

   :template: dataclass.rst

   StreamOptions
   EventOptions
   LaunchConfig
   ContextOptions
   SMResourceOptions
   WorkqueueResourceOptions

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
   ManagedBuffer
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


CUDA compilation toolchain
--------------------------

.. currentmodule:: cuda.core

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

Program caches
``````````````

``Program.compile`` accepts a ``cache=`` keyword argument that integrates
with any :class:`~cuda.core.utils.ProgramCacheResource`, so callers can
avoid recompiling identical source + options + target without writing the
:func:`~cuda.core.utils.make_program_cache_key` lookup by hand.

.. currentmodule:: cuda.core.utils

.. autosummary::
   :toctree: generated/

   ProgramCacheResource
   InMemoryProgramCache
   FileStreamProgramCache
   make_program_cache_key


CUDA graphs
-----------

A CUDA graph captures a set of GPU operations and their dependencies,
allowing them to be defined once and launched repeatedly with minimal
CPU overhead. Graphs can be constructed in two ways:
:class:`~graph.GraphBuilder` captures operations from a stream, while
:class:`~graph.GraphDefinition` builds a graph explicitly by adding nodes and
edges. Both produce an executable :class:`~graph.Graph` that can be
launched on a :class:`Stream`.

.. currentmodule:: cuda.core

.. autosummary::
   :toctree: generated/

   graph.Graph
   graph.GraphBuilder
   graph.GraphDefinition

   :template: autosummary/cyclass.rst

   graph.GraphNode
   graph.GraphCondition

   :template: dataclass.rst

   graph.GraphCompleteOptions
   graph.GraphDebugPrintOptions

Node types
``````````

Every graph node is a subclass of :class:`~graph.GraphNode`, which
provides the common interface (dependencies, successors, destruction).
Each subclass exposes attributes unique to its operation type.

Parameter-bearing definition nodes expose subclass-specific ``update()``
methods: :class:`~graph.KernelNode`, :class:`~graph.MemcpyNode`,
:class:`~graph.MemsetNode`, :class:`~graph.ChildGraphNode`,
:class:`~graph.EventRecordNode`, :class:`~graph.EventWaitNode`, and
:class:`~graph.HostCallbackNode`. These methods require CUDA driver and
``cuda.bindings`` versions 12.2 or newer. Updates affect future graph
instantiations; executable graphs that were already instantiated continue
using their previous parameters and retained resources. Omitted optional
arguments preserve their current values where supported.
On CUDA 12.2 through 13.1, the intended CUDA context must be current when
updating memcpy or memset nodes. CUDA driver and ``cuda.bindings`` versions
13.2 and newer preserve the recorded context automatically.
Multidimensional or array-backed memcpy nodes and clustered or cooperative
kernel nodes cannot currently be updated. Clustered and cooperative kernel
nodes also cannot currently be constructed explicitly.

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

Executable node views
`````````````````````

Index an executable :class:`~graph.Graph` with a definition node to update that
node in the executable, for example
``graph[kernel_node].update(config=config, kernel=kernel, args=args)``.
The returned view retains the executable and source node, while CUDA validates
that the node is associated with the executable.

Executable graphs do not support reading back current node parameters, so
updates take a complete replacement. Buffer operands, kernels, events, kernel
arguments, and callback bindings are retained for every future launch that may
use them. Superseded resources remain retained until a successful whole-graph
update or executable destruction. Raw integer addresses remain caller-owned.
Memcpy and memset updates use the current CUDA context, which must match the
original node context.

Kernel, memcpy, and memset views also provide ``is_enabled``, ``enable()``, and
``disable()``. Executable-node updates require CUDA driver and
``cuda.bindings`` versions 12.2 or newer.

.. autosummary::
   :toctree: generated/

   :template: autosummary/cyclass.rst

   graph.ExecutableGraphNode
   graph.ExecutableKernelNode
   graph.ExecutableMemcpyNode
   graph.ExecutableMemsetNode
   graph.ExecutableHostCallbackNode
   graph.ExecutableChildGraphNode
   graph.ExecutableEventRecordNode
   graph.ExecutableEventWaitNode


Graphics interoperability
-------------------------

.. currentmodule:: cuda.core

.. autosummary::
   :toctree: generated/

   :template: autosummary/cyclass.rst

   GraphicsResource


Tensor Memory Accelerator (TMA)
-------------------------------

.. currentmodule:: cuda.core

.. autosummary::
   :toctree: generated/

   :template: autosummary/cyclass.rst

   TensorMapDescriptor

   :template: dataclass.rst

   TensorMapDescriptorOptions


Textures and surfaces
---------------------

CUDA arrays back bindless texture and surface objects for kernel-side sampled
reads and typed load/store. These types live in the :mod:`cuda.core.texture`
namespace. :class:`OpaqueArray` is allocated through
:meth:`cuda.core.Device.create_opaque_array` and bound through a
:class:`ResourceDescriptor` factory; linear (1D) and row-pitched 2D
:class:`Buffer` views as well as mipmapped allocations (:class:`MipmappedArray`,
via :meth:`cuda.core.Device.create_mipmapped_array`) are also supported as
texture backings. Bindless handles are created with
:meth:`cuda.core.Device.create_texture_object` and
:meth:`cuda.core.Device.create_surface_object`.

A :class:`OpaqueArray` has an opaque, hardware-defined layout with no linear
device pointer, so it cannot participate in ``__cuda_array_interface__`` /
DLPack zero-copy interop. Data is moved in and out only by copying — use
:meth:`OpaqueArray.copy_from` / :meth:`OpaqueArray.copy_to` against a linear
:class:`Buffer` or a host buffer-protocol object.

.. currentmodule:: cuda.core.texture

.. autosummary::
   :toctree: generated/

   :template: autosummary/cyclass.rst

   OpaqueArray
   MipmappedArray
   ResourceDescriptor
   TextureObject
   SurfaceObject

   :template: dataclass.rst

   OpaqueArrayOptions
   MipmappedArrayOptions
   TextureObjectOptions

The associated enumerations —
:class:`~cuda.core.typing.ArrayFormatType`,
:class:`~cuda.core.typing.AddressModeType`,
:class:`~cuda.core.typing.FilterModeType`, and
:class:`~cuda.core.typing.ReadModeType` — live in :mod:`cuda.core.typing`
alongside the other ``cuda.core`` enumerations.


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
every GPU visible to the NVIDIA kernel-mode driver at checkpoint time.
User-space masking such as ``CUDA_VISIBLE_DEVICES`` does not reduce this
mapping requirement, so applications that rely on user-space GPU masking may
not be valid migration targets. The mapping may use ``CUuuid`` objects or the
UUID strings returned by :attr:`Device.uuid`. A successful restore returns the
process to the locked state; call ``Process.unlock`` after restore to allow
CUDA API calls to resume.

The CUDA driver requires restore to run from the process restore thread.
Use ``Process.restore_thread_id`` to discover that thread before calling
``Process.restore`` from a checkpoint coordinator. Restore also requires
persistence mode to be enabled or ``cuInit`` to have been called before
execution.

.. currentmodule:: cuda.core

.. autosummary::
   :toctree: generated/

   :template: class.rst

   checkpoint.Process


Utility functions
-----------------

.. currentmodule:: cuda.core

.. autosummary::
   :toctree: generated/

   utils.args_viewable_as_strided_memory
   utils.prefetch_batch
   utils.discard_batch
   utils.discard_prefetch_batch

   :template: autosummary/cyclass.rst

   utils.StridedMemoryView
