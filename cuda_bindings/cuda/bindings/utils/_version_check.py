# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import threading
import warnings

# Track whether we've already checked major version compatibility
_major_version_compatibility_checked = False
_lock = threading.Lock()

_DISABLE_WARNING_ENV_VAR = "CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING"


def _warning_disabled() -> bool:
    """Whether the user asked to suppress the major-version warning.

    ``=0`` means "do not suppress". A bare truthiness test on the raw string
    made ``CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING=0`` suppress the warning
    -- the exact opposite of what the warning itself tells the user to type,
    and the opposite of the other boolean knobs in this repository
    (``CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM`` and
    ``CUDA_CORE_DONT_FIX_TAB_COMPLETION``), which both parse their value with
    ``int()``.

    Unset and empty still mean "not disabled". A value that is not an integer
    keeps the old set-means-disabled behaviour, so anyone currently relying on
    a spelling like ``=true`` does not silently start seeing the warning again.
    """
    raw = os.environ.get(_DISABLE_WARNING_ENV_VAR, "").strip()
    if not raw:
        return False
    try:
        return int(raw) != 0
    except ValueError:
        return True


def warn_if_cuda_major_version_mismatch():
    """Warn if the CUDA driver major version is older than cuda-bindings compile-time version.

    This function compares the CUDA major version that cuda-bindings was compiled
    against with the CUDA major version supported by the installed driver. If the
    compile-time major version is greater than the driver's major version, a warning
    is issued.

    The check runs only once per process. Subsequent calls are no-ops.

    The warning can be suppressed by setting the environment variable
    ``CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING=1``. Setting it to ``0`` (or
    leaving it unset or empty) keeps the warning enabled.
    """
    global _major_version_compatibility_checked
    if _major_version_compatibility_checked:
        return
    with _lock:
        if _major_version_compatibility_checked:
            return
        _major_version_compatibility_checked = True

    # Allow users to suppress the warning
    if _warning_disabled():
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
            f"(Set {_DISABLE_WARNING_ENV_VAR}=1 to suppress this warning.)",
            UserWarning,
            stacklevel=3,
        )
