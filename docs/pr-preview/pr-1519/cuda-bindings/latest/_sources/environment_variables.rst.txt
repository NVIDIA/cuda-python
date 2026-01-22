.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

Environment Variables
=====================

Runtime Environment Variables
-----------------------------

- ``CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM`` : When set to 1, the default stream is the per-thread default stream. When set to 0, the default stream is the legacy default stream. This defaults to 0, for the legacy default stream. See `Stream Synchronization Behavior <https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html>`_ for an explanation of the legacy and per-thread default streams.


Build-Time Environment Variables
--------------------------------

- ``CUDA_PATH`` or ``CUDA_HOME``: Specifies the location of the CUDA Toolkit. If both are set, ``CUDA_PATH`` takes precedence. This search order is defined in :py:data:`cuda.pathfinder._utils.env_vars.CUDA_ENV_VARS_ORDERED`.

  .. note::
     **Breaking Change in v1.4.0**: The priority order changed from ``CUDA_HOME`` > ``CUDA_PATH`` to ``CUDA_PATH`` > ``CUDA_HOME``.

     **Migration Guide**:

     - If you only set one variable, no changes are needed
     - If you set both variables to the same location, no changes are needed
     - If you set both variables to different locations and relied on ``CUDA_HOME`` taking precedence, you should either:

       - Switch to using only ``CUDA_PATH`` (recommended)
       - Ensure both variables point to the same CUDA Toolkit installation
       - Be aware that ``CUDA_PATH`` will now be used

     A warning will be issued if both variables are set but point to different locations.

- ``CUDA_PYTHON_PARSER_CACHING`` : bool, toggles the caching of parsed header files during the cuda-bindings build process. If caching is enabled (``CUDA_PYTHON_PARSER_CACHING`` is True), the cache path is set to ./cache_<library_name>, where <library_name> is derived from the cuda toolkit libraries used to build cuda-bindings.

- ``CUDA_PYTHON_PARALLEL_LEVEL`` (previously ``PARALLEL_LEVEL``) : int, sets the number of threads used in the compilation of extension modules. Not setting it or setting it to 0 would disable parallel builds.
