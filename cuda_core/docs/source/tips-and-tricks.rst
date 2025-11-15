.. SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.core.experimental

Tips and Tricks
===============

This page provides helpful tips and best practices for working with ``cuda.core``.

CUDA Context Requirements
--------------------------

Understanding when CUDA operations require an active context can help you write more efficient and flexible code.

Operations that don't need an active context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many ``cuda.core`` operations **do not require an active CUDA context** (i.e., a device set to current). These operations can be performed with minimal setup:

**Operations requiring only CUDA initialization** (``cuInit(0)`` called):

- Querying device properties via :class:`Device`
- Enumerating available devices
- Getting device attributes and capabilities
- Checking driver and runtime versions

Example:

.. code-block:: python

   from cuda.core.experimental import Device

   # These work without setting a device as current
   # Only requires CUDA to be initialized
   dev = Device(0)
   compute_capability = dev.arch
   device_name = dev.name
   total_memory = dev.memory_total

**Operations requiring no CUDA initialization**:

Some operations don't even require CUDA to be initialized or the driver/GPU to be present, if the options are passed correctly:

- JIT compilation of CUDA kernels via :class:`Program` (when targeting specific architectures)
- Code generation and optimization with Link-Time Optimization (LTO)
- Creating program objects from source code

Example:

.. code-block:: python

   from cuda.core.experimental import Program, ProgramOptions

   code = """
   __global__ void simple_kernel() {
       // kernel code
   }
   """

   # This works even without a GPU present!
   # Just specify the target architecture
   options = ProgramOptions(std="c++17", arch="sm_80")
   prog = Program(code, code_type="c++", options=options)

   # Compilation can proceed without an active context
   # (though loading/executing the result does require a device)

Operations that require an active context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following operations **do require** an active CUDA context (i.e., you must call :meth:`Device.set_current` first):

- Allocating device memory via :class:`Buffer` or :class:`MemoryResource`
- Launching kernels on a :class:`Stream`
- Creating and executing CUDA :class:`GraphBuilder` objects
- Recording and waiting on :class:`Event` objects
- Transferring data to/from device memory

Example:

.. code-block:: python

   from cuda.core.experimental import Device

   dev = Device()
   dev.set_current()  # Required for memory allocation and kernel launches

   s = dev.create_stream()
   # Now you can allocate memory, launch kernels, etc.

Context Stack Management
------------------------

``cuda.core`` maintains a simplified context stack:

- The context stack size is kept at **either 0 or 1**
- When you call :meth:`Device.set_current`, it becomes the active context
- There is no deep context stack nesting as in the raw CUDA driver API
- This simplification makes context management more predictable and Pythonic

This design choice means:

- You don't need to worry about balancing push/pop operations
- Context switches are explicit via :meth:`Device.set_current`
- Multi-device workflows require explicit device switching

Example of multi-device usage:

.. code-block:: python

   from cuda.core.experimental import Device

   dev0 = Device(0)
   dev1 = Device(1)

   # Work on device 0
   dev0.set_current()
   stream0 = dev0.create_stream()
   # ... operations on device 0 ...

   # Switch to device 1
   dev1.set_current()
   stream1 = dev1.create_stream()
   # ... operations on device 1 ...

   # Switch back to device 0
   dev0.set_current()
   # ... more operations on device 0 ...

Memory and Resource Management
-------------------------------

Best practices for managing CUDA resources:

Always use context managers when available
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While ``cuda.core`` provides explicit ``.close()`` methods for cleanup, using context managers (``with`` statements) ensures proper resource cleanup even when exceptions occur:

.. code-block:: python

   from cuda.core.experimental import Device

   dev = Device()
   dev.set_current()

   # Preferred: use context manager
   with dev.create_stream() as s:
       # stream operations
       pass
   # stream automatically cleaned up

Synchronization and timing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`Stream` and :class:`Event` for fine-grained control over execution ordering and timing:

.. code-block:: python

   from cuda.core.experimental import Device

   dev = Device()
   dev.set_current()

   stream = dev.create_stream()

   # Launch work on stream
   # ...

   # Wait for completion
   stream.sync()

Program compilation tips
------------------------

Target multiple architectures for portability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When distributing code, compile for multiple GPU architectures to ensure compatibility:

.. code-block:: python

   from cuda.core.experimental import Program, ProgramOptions

   # Compile for multiple architectures
   for arch in ["sm_70", "sm_80", "sm_90"]:
       options = ProgramOptions(std="c++17", arch=arch)
       prog = Program(code, code_type="c++", options=options)
       cubin = prog.compile("cubin")
       # Save cubin for later use

Use ``name_expressions`` for template instantiations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When working with templated CUDA kernels, use the ``name_expressions`` parameter to explicitly specify which template instantiations to compile:

.. code-block:: python

   from cuda.core.experimental import Program, ProgramOptions, Device

   dev = Device()

   code = """
   template<typename T>
   __global__ void my_kernel(T* data) {
       // kernel implementation
   }
   """

   options = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
   prog = Program(code, code_type="c++", options=options)

   # Compile specific template instantiations
   mod = prog.compile("cubin",
                      name_expressions=("my_kernel<float>", "my_kernel<double>"))
