# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import os
import warnings
from unittest import mock

import pytest
from cuda.bindings import driver
from cuda.bindings.utils import _version_check, warn_if_cuda_major_version_mismatch


class TestVersionCompatibilityCheck:
    """Tests for CUDA major version mismatch warning function."""

    def setup_method(self):
        """Reset the version compatibility check flag before each test."""
        _version_check._major_version_compatibility_checked = False

    def teardown_method(self):
        """Reset the version compatibility check flag after each test."""
        _version_check._major_version_compatibility_checked = False

    def test_no_warning_when_driver_newer(self):
        """No warning should be issued when driver version >= compile version."""
        # Mock compile version 12.9 and driver version 13.0
        with (
            mock.patch.object(driver, "CUDA_VERSION", 12090),
            mock.patch.object(driver, "cuDriverGetVersion", return_value=(driver.CUresult.CUDA_SUCCESS, 13000)),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            warn_if_cuda_major_version_mismatch()
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
            warn_if_cuda_major_version_mismatch()
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
            warn_if_cuda_major_version_mismatch()
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "cuda-bindings was built for CUDA major version 13" in str(w[0].message)
            assert "only supports up to CUDA 12" in str(w[0].message)

    def test_warning_only_issued_once(self):
        """Warning should only be issued once per process."""
        with (
            mock.patch.object(driver, "CUDA_VERSION", 13000),
            mock.patch.object(driver, "cuDriverGetVersion", return_value=(driver.CUresult.CUDA_SUCCESS, 12080)),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            warn_if_cuda_major_version_mismatch()
            warn_if_cuda_major_version_mismatch()
            warn_if_cuda_major_version_mismatch()
            # Only one warning despite multiple calls
            assert len(w) == 1

    def test_warning_suppressed_by_env_var(self):
        """Warning should be suppressed when CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING is set."""
        with (
            mock.patch.object(driver, "CUDA_VERSION", 13000),
            mock.patch.object(driver, "cuDriverGetVersion", return_value=(driver.CUresult.CUDA_SUCCESS, 12080)),
            mock.patch.dict(os.environ, {"CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING": "1"}),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            warn_if_cuda_major_version_mismatch()
            assert len(w) == 0

    def test_error_when_driver_version_fails(self):
        """Should raise RuntimeError if cuDriverGetVersion fails."""
        with (
            mock.patch.object(driver, "CUDA_VERSION", 13000),
            mock.patch.object(
                driver, "cuDriverGetVersion", return_value=(driver.CUresult.CUDA_ERROR_NOT_INITIALIZED, 0)
            ),
            pytest.raises(RuntimeError, match="Failed to query CUDA driver version"),
        ):
            warn_if_cuda_major_version_mismatch()
