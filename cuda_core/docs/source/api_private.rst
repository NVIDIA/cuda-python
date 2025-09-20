.. SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

:orphan:

.. This page is to generate documentation for private classes exposed to users,
   i.e., users cannot instantiate it by themselves but may use it's properties
   or methods via returned values from public APIs. These classes must be referred
   in public APIs returning their instances.

.. currentmodule:: cuda.core.experimental

CUDA runtime
------------

.. autosummary::
   :toctree: generated/

   _memory.PyCapsule
   _memory.DevicePointerT
   _memory.IPCBufferDescriptor
   _device.DeviceProperties
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
