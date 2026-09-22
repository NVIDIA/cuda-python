.. SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. module:: cuda.core.system

``cuda.core.system`` API Reference
==================================

This is the API reference for Pythonic access to CUDA system information,
through the NVIDIA Management Library (NVML).

.. note::
   ``cuda.core.system`` uses NVML through ``cuda-bindings``. It has no requirement beyond the
   ``cuda-bindings`` floor of the release (see :ref:`cuda-core-bindings-floor`); the NVML library
   itself is loaded on first use, so importing the module needs neither CUDA nor NVML installed.

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

Constants
---------

.. autosummary::
   :toctree: generated/

   CUDA_BINDINGS_NVML_IS_COMPATIBLE
