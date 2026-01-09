# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import os
import warnings

# Track whether we've already checked major version compatibility
_major_version_compatibility_checked = False


def warn_if_cuda_major_version_mismatch():
    """Warn if the CUDA driver major version is older than cuda-bindings compile-time version.

    This function compares the CUDA major version that cuda-bindings was compiled
    against with the CUDA major version supported by the installed driver. If the
    compile-time major version is greater than the driver's major version, a warning
    is issued.

    The check runs only once per process. Subsequent calls are no-ops.

    The warning can be suppressed by setting the environment variable
    ``CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING=1``.
    """
    global _major_version_compatibility_checked
    if _major_version_compatibility_checked:
        return
    _major_version_compatibility_checked = True

    # Allow users to suppress the warning
    if os.environ.get("CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING"):
        return

    # Import here to avoid circular imports and allow lazy loading
    from cuda.bindings import driver

    # Get compile-time CUDA version from cuda-bindings
    compile_version = driver.CUDA_VERSION  # e.g., 13010
    compile_major = compile_version // 1000

    # Get runtime driver version
    err, runtime_version = driver.cuDriverGetVersion()
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to query CUDA driver version: {err}")

    runtime_major = runtime_version // 1000

    if compile_major > runtime_major:
        warnings.warn(
            f"cuda-bindings was built for CUDA major version {compile_major}, but the "
            f"NVIDIA driver only supports up to CUDA {runtime_major}. Some cuda-bindings "
            f"features may not work correctly. Consider updating your NVIDIA driver, "
            f"or using a cuda-bindings version built for CUDA {runtime_major}. "
            f"(Set CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING=1 to suppress this warning.)",
            UserWarning,
            stacklevel=3,
        )
