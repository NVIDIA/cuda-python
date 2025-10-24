.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

Environment Variables
=====================

Runtime Environment Variables
-----------------------------

- ``CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM`` : When set to 1, the default stream is the per-thread default stream. When set to 0, the default stream is the legacy default stream. This defaults to 0, for the legacy default stream. See `Stream Synchronization Behavior <https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html>`_ for an explanation of the legacy and per-thread default streams.


Build-Time Environment Variables
--------------------------------

- ``CUDA_HOME`` or ``CUDA_PATH``: Specifies the location of the CUDA Toolkit.

- ``CUDA_PYTHON_PARSER_CACHING`` : bool, toggles the caching of parsed header files during the cuda-bindings build process. If caching is enabled (``CUDA_PYTHON_PARSER_CACHING`` is True), the cache path is set to ./cache_<library_name>, where <library_name> is derived from the cuda toolkit libraries used to build cuda-bindings.

- ``CUDA_PYTHON_PARALLEL_LEVEL`` (previously ``PARALLEL_LEVEL``) : int, sets the number of threads used in the compilation of extension modules. Not setting it or setting it to 0 would disable parallel builds.
