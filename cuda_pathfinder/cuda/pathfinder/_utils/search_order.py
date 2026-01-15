# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centralized search order constants for NVIDIA pathfinder utilities.

This module defines the standard search order used across all pathfinder
functions (headers, binaries, static libraries, and dynamic libraries).
"""

# Standard search order used by all pathfinder functions
SEARCH_ORDER_DESCRIPTION = """
Search order:
    1. **NVIDIA Python wheels**

       - Scan installed distributions (``site-packages``) for artifacts
         shipped in NVIDIA wheels (e.g., ``cuda-toolkit``, ``cuda-nvcc``).

    2. **Conda environments**

       - Check Conda-style installation prefixes (``$CONDA_PREFIX``), which
         use platform-specific directory layouts.

    3. **CUDA Toolkit environment variables**

       - Use ``CUDA_HOME`` or ``CUDA_PATH`` (in that order).
"""

# For programmatic access
SEARCH_ORDER = [
    "site-packages",  # NVIDIA Python wheels
    "conda",  # Conda environment
    "cuda_home_or_path",  # CUDA_HOME or CUDA_PATH environment variables
]
