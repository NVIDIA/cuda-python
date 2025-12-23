# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import os
import warnings

# Track whether we've already checked version compatibility
_version_compatibility_checked = False


def check_cuda_version_compatibility():
    """Check if the CUDA driver version is compatible with cuda-bindings compile-time version.

    This function compares the CUDA version that cuda-bindings was compiled against
    with the CUDA version supported by the installed driver. If the compile-time
    major version is greater than the driver's major version, a warning is issued.

    The check runs only once per process. Subsequent calls are no-ops.

    The warning can be suppressed by setting the environment variable
    ``CUDA_PYTHON_DISABLE_VERSION_CHECK=1``.

    Examples
    --------
    >>> from cuda.bindings.utils import check_cuda_version_compatibility
    >>> check_cuda_version_compatibility()  # Issues warning if version mismatch
    """
    global _version_compatibility_checked
    if _version_compatibility_checked:
        return
    _version_compatibility_checked = True

    # Allow users to suppress the warning
    if os.environ.get("CUDA_PYTHON_DISABLE_VERSION_CHECK"):
        return

    # Import here to avoid circular imports and allow lazy loading
    from cuda.bindings import driver

    # Get compile-time CUDA version from cuda-bindings
    try:
        compile_version = driver.CUDA_VERSION  # e.g., 13010
    except AttributeError:
        # Older cuda-bindings may not expose CUDA_VERSION
        return

    # Get runtime driver version
    err, runtime_version = driver.cuDriverGetVersion()
    if err != driver.CUresult.CUDA_SUCCESS:
        return  # Can't check, skip silently

    compile_major = compile_version // 1000
    runtime_major = runtime_version // 1000

    if compile_major > runtime_major:
        compile_minor = (compile_version % 1000) // 10
        runtime_minor = (runtime_version % 1000) // 10
        warnings.warn(
            f"cuda-bindings was built against CUDA {compile_major}.{compile_minor}, "
            f"but the installed driver only supports CUDA {runtime_major}.{runtime_minor}. "
            f"Some features may not work correctly. Consider updating your NVIDIA driver. "
            f"Set CUDA_PYTHON_DISABLE_VERSION_CHECK=1 to suppress this warning.",
            UserWarning,
            stacklevel=3,
        )


def _reset_version_compatibility_check():
    """Reset the version compatibility check flag for testing purposes.

    This function is intended for use in tests to allow multiple test runs
    to check the warning behavior.
    """
    global _version_compatibility_checked
    _version_compatibility_checked = False
