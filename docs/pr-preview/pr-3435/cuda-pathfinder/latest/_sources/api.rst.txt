.. SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. module:: cuda.pathfinder

``cuda.pathfinder`` API Reference
=================================

The ``cuda.pathfinder`` module provides utilities for loading NVIDIA dynamic libraries,
locating NVIDIA C/C++ header directories, finding CUDA binary utilities, and locating
CUDA bitcode and static libraries.

Header, binary-utility, bitcode-library, and static-library lookups return
resolved absolute paths. Symlinks and parent-directory components (``..``)
are resolved using the filesystem, so the returned path identifies the
selected component even when a configured root contains a symlink followed
by ``..``. Cached results retain the resolved path selected by the first
lookup. Public path values remain strings.

.. NOTE: The source of truth for public APIs is cuda_pathfinder/cuda/pathfinder/__init__.py.
.. Keep this documentation in sync when adding or removing exports.

.. autosummary::
   :toctree: generated/

   get_cuda_path_or_home

   SUPPORTED_NVIDIA_LIBNAMES
   load_nvidia_dynamic_lib
   LoadedDL
   DynamicLibNotFoundError
   DynamicLibUnknownError
   DynamicLibNotAvailableError
   UnsupportedArchError

   SUPPORTED_HEADERS_CTK
   find_nvidia_header_directory
   locate_nvidia_header_directory
   LocatedHeaderDir

   SUPPORTED_BINARY_UTILITIES
   find_nvidia_binary_utility

   SUPPORTED_BITCODE_LIBS
   find_bitcode_lib
   locate_bitcode_lib
   LocatedBitcodeLib
   BitcodeLibNotFoundError

   SUPPORTED_STATIC_LIBS
   find_static_lib
   locate_static_lib
   LocatedStaticLib
   StaticLibNotFoundError
