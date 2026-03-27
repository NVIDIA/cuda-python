.. SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
   typing.VirtualMemoryAllocationTypeT
   typing.VirtualMemoryLocationTypeT
   typing.VirtualMemoryGranularityT
   typing.VirtualMemoryAccessTypeT
   typing.VirtualMemoryHandleTypeT
   _module.KernelAttributes
   _module.KernelOccupancy
   _module.ParamInfo
   _module.MaxPotentialBlockSizeOccupancyResult

   :template: autosummary/cyclass.rst

   _device.DeviceProperties
   _memory._ipc.IPCAllocationHandle
   _memory._ipc.IPCBufferDescriptor


CUDA protocols
--------------

.. autosummary::
   :toctree: generated/
   :template: protocol.rst

   typing.IsStreamT
