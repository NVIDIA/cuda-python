# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import os
import warnings
from unittest import mock

from cuda.bindings import driver
from cuda.bindings.utils import check_cuda_version_compatibility
from cuda.bindings.utils._version_check import _reset_version_compatibility_check


class TestVersionCompatibilityCheck:
    """Tests for CUDA version compatibility check function."""

    def setup_method(self):
        """Reset the version compatibility check flag before each test."""
        _reset_version_compatibility_check()

    def teardown_method(self):
        """Reset the version compatibility check flag after each test."""
        _reset_version_compatibility_check()

    def test_no_warning_when_driver_newer(self):
        """No warning should be issued when driver version >= compile version."""
        # Mock compile version 12.9 and driver version 13.0
        with (
            mock.patch.object(driver, "CUDA_VERSION", 12090),
            mock.patch.object(driver, "cuDriverGetVersion", return_value=(driver.CUresult.CUDA_SUCCESS, 13000)),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            check_cuda_version_compatibility()
            assert len(w) == 0

    def test_no_warning_when_same_major_version(self):
        """No warning should be issued when major versions match."""
        # Mock compile version 12.9 and driver version 12.8
        with (
            mock.patch.object(driver, "CUDA_VERSION", 12090),
            mock.patch.object(driver, "cuDriverGetVersion", return_value=(driver.CUresult.CUDA_SUCCESS, 12080)),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            check_cuda_version_compatibility()
            assert len(w) == 0

    def test_warning_when_compile_major_newer(self):
        """Warning should be issued when compile major version > driver major version."""
        # Mock compile version 13.0 and driver version 12.8
        with (
            mock.patch.object(driver, "CUDA_VERSION", 13000),
            mock.patch.object(driver, "cuDriverGetVersion", return_value=(driver.CUresult.CUDA_SUCCESS, 12080)),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            check_cuda_version_compatibility()
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "cuda-bindings was built against CUDA 13.0" in str(w[0].message)
            assert "driver only supports CUDA 12.8" in str(w[0].message)

    def test_warning_only_issued_once(self):
        """Warning should only be issued once per process."""
        with (
            mock.patch.object(driver, "CUDA_VERSION", 13000),
            mock.patch.object(driver, "cuDriverGetVersion", return_value=(driver.CUresult.CUDA_SUCCESS, 12080)),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            check_cuda_version_compatibility()
            check_cuda_version_compatibility()
            check_cuda_version_compatibility()
            # Only one warning despite multiple calls
            assert len(w) == 1

    def test_warning_suppressed_by_env_var(self):
        """Warning should be suppressed when CUDA_PYTHON_DISABLE_VERSION_CHECK is set."""
        with (
            mock.patch.object(driver, "CUDA_VERSION", 13000),
            mock.patch.object(driver, "cuDriverGetVersion", return_value=(driver.CUresult.CUDA_SUCCESS, 12080)),
            mock.patch.dict(os.environ, {"CUDA_PYTHON_DISABLE_VERSION_CHECK": "1"}),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            check_cuda_version_compatibility()
            assert len(w) == 0

    def test_silent_when_driver_version_fails(self):
        """Should silently skip if cuDriverGetVersion fails."""
        with (
            mock.patch.object(driver, "CUDA_VERSION", 13000),
            mock.patch.object(
                driver, "cuDriverGetVersion", return_value=(driver.CUresult.CUDA_ERROR_NOT_INITIALIZED, 0)
            ),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            check_cuda_version_compatibility()
            assert len(w) == 0

    def test_silent_when_cuda_version_not_available(self):
        """Should silently skip if CUDA_VERSION attribute is not available."""
        # Simulate older cuda-bindings without CUDA_VERSION
        original = driver.CUDA_VERSION
        try:
            del driver.CUDA_VERSION
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                check_cuda_version_compatibility()
                assert len(w) == 0
        finally:
            driver.CUDA_VERSION = original
