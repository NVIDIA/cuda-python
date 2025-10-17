.. SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

:orphan:

.. This page is to generate documentation for private classes exposed to users,
   i.e., users cannot instantiate them but may use their properties or methods
   via returned values from public APIs. These classes must be referred in
   public APIs returning their instances.

.. currentmodule:: cuda.core.experimental

CUDA runtime
------------

.. autosummary::
   :toctree: generated/

   _memory.PyCapsule
   _memory.DevicePointerT
   _memory.VirtualMemoryAllocationTypeT
   _memory.VirtualMemoryLocationTypeT
   _memory.VirtualMemoryGranularityT
   _memory.VirtualMemoryAccessTypeT
   _memory.VirtualMemoryHandleTypeT
   _device.DeviceProperties
   _memory.IPCAllocationHandle
   _memory.IPCBufferDescriptor
   _module.KernelAttributes
   _module.KernelOccupancy
   _module.ParamInfo
   _module.MaxPotentialBlockSizeOccupancyResult


CUDA protocols
--------------

.. autosummary::
   :toctree: generated/
   :template: protocol.rst

   _stream.IsStreamT
