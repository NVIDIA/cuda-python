.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. module:: cuda.pathfinder

``cuda.pathfinder`` API Reference
=================================

The ``cuda.pathfinder`` module provides utilities for loading NVIDIA dynamic libraries,
locating NVIDIA C/C++ header directories, and finding CUDA binary utilities.

.. autosummary::
   :toctree: generated/

   SUPPORTED_NVIDIA_LIBNAMES
   load_nvidia_dynamic_lib
   LoadedDL
   DynamicLibNotFoundError
   DynamicLibUnknownError
   DynamicLibNotAvailableError

   SUPPORTED_HEADERS_CTK
   SUPPORTED_HEADERS_NON_CTK
   find_nvidia_header_directory

   SUPPORTED_BINARY_UTILITIES
   find_nvidia_binary_utility
