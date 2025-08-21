.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

API Reference
=============

cuda.pathfinder
---------------

.. automodule:: cuda.pathfinder
   :members:
   :undoc-members:
   :show-inheritance:

Public API
----------

The ``cuda.pathfinder`` module provides the following public API for loading NVIDIA dynamic libraries:

Constants
~~~~~~~~~

.. autodata:: cuda.pathfinder.SUPPORTED_NVIDIA_LIBNAMES
   :annotation: : tuple[str]

Functions
~~~~~~~~~

.. autofunction:: cuda.pathfinder.load_nvidia_dynamic_lib

Classes
~~~~~~~

.. autoclass:: cuda.pathfinder.LoadedDL
   :members:
   :undoc-members:

Exceptions
~~~~~~~~~~

.. autoexception:: cuda.pathfinder.DynamicLibNotFoundError
