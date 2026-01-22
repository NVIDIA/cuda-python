# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centralized CUDA environment variable handling.

This module defines the canonical search order for CUDA Toolkit environment variables
used throughout cuda-python packages (cuda.pathfinder, cuda.core, cuda.bindings).

Search Order Priority:
    1. CUDA_PATH (higher priority)
    2. CUDA_HOME (lower priority)

If both are set and differ, CUDA_PATH takes precedence and a warning is issued.

Important Note on Caching:
    The result of get_cuda_home_or_path() is cached for the process lifetime. The first
    call determines the CUDA Toolkit path, and all subsequent calls return the cached
    value, even if environment variables change later. This ensures consistent behavior
    throughout the application lifecycle.
"""

import functools
import os
import warnings

from .env_var_constants import CUDA_ENV_VARS_ORDERED


def _paths_differ(a: str, b: str) -> bool:
    """
    Return True if paths are observably different.

    Strategy:
    1) Compare os.path.normcase(os.path.normpath(...)) for quick, robust textual equality.
       - Handles trailing slashes and case-insensitivity on Windows.
    2) If still different AND both exist, use os.path.samefile to resolve symlinks/junctions.
    3) Otherwise (nonexistent paths or samefile unavailable), treat as different.
    """
    norm_a = os.path.normcase(os.path.normpath(a))
    norm_b = os.path.normcase(os.path.normpath(b))
    if norm_a == norm_b:
        return False

    try:
        if os.path.exists(a) and os.path.exists(b):
            # samefile raises on non-existent paths; only call when both exist.
            return not os.path.samefile(a, b)
    except OSError:
        # Fall through to "different" if samefile isn't applicable/available.
        pass

    # If normalized strings differ and we couldn't prove they're the same entry, treat as different.
    return True


@functools.cache
def get_cuda_home_or_path() -> str | None:
    """Get CUDA Toolkit path from environment variables.

    Returns the value of CUDA_PATH or CUDA_HOME following the canonical search order
    defined in CUDA_ENV_VARS_ORDERED. If both are set and differ, CUDA_PATH takes
    precedence and a warning is issued.

    The result is cached for the process lifetime. The first call determines the CUDA
    Toolkit path, and subsequent calls return the cached value.

    Returns:
        Path to CUDA Toolkit, or None if neither variable is set. Empty strings are
        preserved and returned as-is if explicitly set in the environment.

    Warnings:
        UserWarning: If multiple CUDA environment variables are set but point to
            different locations (only on the first call).

    See Also:
        CUDA_ENV_VARS_ORDERED: The canonical search order for CUDA environment variables.
    """
    # Collect all set environment variables in priority order
    # Note: We check 'is not None' to preserve empty strings (which are valid but unusual).
    # Empty strings are falsy in Python but may indicate an intentional "unset" by the user.
    set_vars = {}
    for var in CUDA_ENV_VARS_ORDERED:
        val = os.environ.get(var)
        if val is not None:
            set_vars[var] = val
    
    if not set_vars:
        return None
    
    # If multiple variables are set, check if they differ and warn
    if len(set_vars) > 1:
        # Check if any non-empty values actually differ
        non_empty_values = [(var, val) for var, val in set_vars.items() if val]
        
        if len(non_empty_values) > 1:
            # Check if any pair of non-empty values differs
            values_differ = False
            for i in range(len(non_empty_values) - 1):
                if _paths_differ(non_empty_values[i][1], non_empty_values[i + 1][1]):
                    values_differ = True
                    break
            
            if values_differ:
                # Build a generic warning message that works for any number of variables
                var_list = "\n".join(f"  {var}={val}" for var, val in set_vars.items())
                highest_priority = CUDA_ENV_VARS_ORDERED[0]
                warnings.warn(
                    f"Multiple CUDA environment variables are set but differ:\n"
                    f"{var_list}\n"
                    f"Using {highest_priority} (highest priority as defined in CUDA_ENV_VARS_ORDERED).",
                    UserWarning,
                    stacklevel=2,
                )
    
    # Return the first (highest priority) set variable
    for var in CUDA_ENV_VARS_ORDERED:
        if var in set_vars:
            return set_vars[var]
    
    return None
