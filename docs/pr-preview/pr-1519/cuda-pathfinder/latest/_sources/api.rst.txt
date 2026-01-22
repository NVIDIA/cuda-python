.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. module:: cuda.pathfinder

``cuda.pathfinder`` API Reference
=================================

The ``cuda.pathfinder`` module provides utilities for loading NVIDIA dynamic libraries,
and experimental APIs for locating NVIDIA C/C++ header directories.

.. autosummary::
   :toctree: generated/

   SUPPORTED_NVIDIA_LIBNAMES
   load_nvidia_dynamic_lib
   LoadedDL
   DynamicLibNotFoundError

   SUPPORTED_HEADERS_CTK
   SUPPORTED_HEADERS_NON_CTK
   find_nvidia_header_directory

Environment Variable Utilities
-------------------------------

The ``cuda.pathfinder._utils.env_vars`` module provides centralized handling of CUDA
environment variables used across all cuda-python packages.

.. autosummary::
   :toctree: generated/

   _utils.env_vars.get_cuda_home_or_path
   _utils.env_vars.CUDA_ENV_VARS_ORDERED

.. autofunction:: cuda.pathfinder._utils.env_vars.get_cuda_home_or_path

.. autodata:: cuda.pathfinder._utils.env_vars.CUDA_ENV_VARS_ORDERED
