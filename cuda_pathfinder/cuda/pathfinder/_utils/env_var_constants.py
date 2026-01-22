# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants for CUDA environment variable handling.

This module defines the canonical search order for CUDA Toolkit environment variables
without any dependencies. It can be safely imported during bootstrap scenarios.

Search Order Priority:
    1. CUDA_PATH (higher priority)
    2. CUDA_HOME (lower priority)

.. versionadded:: 1.4.0
   Added centralized environment variable handling.

.. versionchanged:: 1.4.0
   **Breaking Change**: Priority changed from CUDA_HOME > CUDA_PATH to CUDA_PATH > CUDA_HOME.
"""

#: Canonical search order for CUDA Toolkit environment variables.
#: 
#: This tuple defines the priority order used by :py:func:`~cuda.pathfinder._utils.env_vars.get_cuda_home_or_path`
#: and throughout cuda-python packages when determining which CUDA Toolkit to use.
#: 
#: The first variable in the tuple has the highest priority. If multiple variables are set
#: and point to different locations, the first one is used and a warning is issued.
#: 
#: .. note::
#:    **Breaking Change in v1.4.0**: The order changed from ``("CUDA_HOME", "CUDA_PATH")`` 
#:    to ``("CUDA_PATH", "CUDA_HOME")``, making ``CUDA_PATH`` the highest priority.
#:
#: :type: tuple[str, ...]
CUDA_ENV_VARS_ORDERED = ("CUDA_PATH", "CUDA_HOME")
