.. SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. module:: cuda.core.experimental

``cuda.core.experimental`` API Reference
========================================

All of the APIs listed (or cross-referenced from) below are considered *experimental*
and subject to future changes without deprecation notice. Once stabilized they will be
moved out of the ``experimental`` namespace.


CUDA runtime
------------

.. autosummary::
   :toctree: generated/

   Device
   Graph
   GraphBuilder
   launch
   Buffer
   Stream
   Event
   MemoryResource
   DeviceMemoryResource
   LegacyPinnedMemoryResource
   VirtualMemoryResource

   :template: dataclass.rst

   DeviceMemoryResourceOptions
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


CUDA system information
-----------------------

.. automethod:: cuda.core.experimental._system.System.get_driver_version
.. automethod:: cuda.core.experimental._system.System.get_num_devices


.. module:: cuda.core.experimental.utils

Utility functions
-----------------

.. autosummary::
   :toctree: generated/

   args_viewable_as_strided_memory

   :template: autosummary/cyclass.rst

   StridedMemoryView
   StridedLayout
