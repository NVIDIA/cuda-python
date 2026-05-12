.. SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. module:: cuda.core.system

``cuda.core.system`` API Reference
==================================

This is the API reference for Pythonic access to CUDA system information,
through the NVIDIA Management Library (NVML).

.. note::
   ``cuda.core.system`` support requires ``cuda_bindings`` 12.9.6 or later, or 13.2.0 or later.

Basic functions
---------------

.. autosummary::
   :toctree: generated/

   get_user_mode_driver_version
   get_kernel_mode_driver_version
   get_driver_branch
   get_num_devices
   get_nvml_version
   get_process_name
   get_topology_common_ancestor
   get_p2p_status

Events
------

.. autosummary::
   :toctree: generated/

   register_events

Types
-----

.. autosummary::
   :toctree: generated/

   :template: autosummary/cyclass.rst

   Device
   NvlinkInfo
