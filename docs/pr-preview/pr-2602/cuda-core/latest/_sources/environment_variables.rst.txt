.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Environment Variables
=====================

.. note::

   The ``cuda-bindings`` runtime environment variables also affect ``cuda.core``.
   See the `cuda-bindings environment variables documentation
   <https://nvidia.github.io/cuda-python/cuda-bindings/latest/environment_variables.html>`_.

Runtime Environment Variables
-----------------------------

- ``CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM`` : When set to 1, the default
  stream is the per-thread default stream. When set to 0, the default stream
  is the legacy default stream. This defaults to 0, for the legacy default
  stream. See `Stream Synchronization Behavior
  <https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html>`_
  for an explanation of the legacy and per-thread default streams.

- ``CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING`` : When set to 1, suppresses
  warnings about CUDA major version mismatches between ``cuda-bindings`` and
  the installed driver. This warning occurs when ``cuda-bindings`` was built
  for a newer CUDA major version than the installed driver supports.
