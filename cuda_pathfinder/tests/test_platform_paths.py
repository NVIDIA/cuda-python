# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for platform path configurations."""

from cuda.pathfinder._utils.platform_aware import IS_WINDOWS
from cuda.pathfinder._utils.platform_paths import CUDA_TARGETS_LIB_SUBDIRS, PLATFORM


def test_platform_paths_structure():
    """Test that PLATFORM has all expected attributes."""
    assert hasattr(PLATFORM, "conda_bin_subdirs")
    assert hasattr(PLATFORM, "conda_lib_subdirs")
    assert hasattr(PLATFORM, "conda_nvvm_subdirs")
    assert hasattr(PLATFORM, "cuda_home_bin_subdirs")
    assert hasattr(PLATFORM, "cuda_home_lib_subdirs")
    assert hasattr(PLATFORM, "cuda_home_nvvm_subdirs")


def test_platform_paths_are_tuples():
    """Test that all paths are tuples (immutable)."""
    assert isinstance(PLATFORM.conda_bin_subdirs, tuple)
    assert isinstance(PLATFORM.conda_lib_subdirs, tuple)
    assert isinstance(PLATFORM.conda_nvvm_subdirs, tuple)
    assert isinstance(PLATFORM.cuda_home_bin_subdirs, tuple)
    assert isinstance(PLATFORM.cuda_home_lib_subdirs, tuple)
    assert isinstance(PLATFORM.cuda_home_nvvm_subdirs, tuple)


def test_platform_paths_windows_specific():
    """Test Windows-specific paths."""
    if IS_WINDOWS:
        assert "Library/bin" in PLATFORM.conda_bin_subdirs
        assert "Library/lib" in PLATFORM.conda_lib_subdirs
        assert "lib/x64" in PLATFORM.cuda_home_lib_subdirs
    else:
        # Unix should have simpler paths
        assert "bin" in PLATFORM.conda_bin_subdirs
        assert "lib" in PLATFORM.conda_lib_subdirs
        assert "lib64" in PLATFORM.cuda_home_lib_subdirs or "lib" in PLATFORM.cuda_home_lib_subdirs


def test_cuda_targets_constant():
    """Test CUDA targets constant."""
    assert isinstance(CUDA_TARGETS_LIB_SUBDIRS, tuple)
    assert len(CUDA_TARGETS_LIB_SUBDIRS) == 2
    assert "lib64" in CUDA_TARGETS_LIB_SUBDIRS
    assert "lib" in CUDA_TARGETS_LIB_SUBDIRS


def test_nvvm_subdirs_present():
    """Test that NVVM subdirs are configured."""
    assert len(PLATFORM.conda_nvvm_subdirs) > 0
    assert len(PLATFORM.cuda_home_nvvm_subdirs) > 0
    assert "libdevice" in str(PLATFORM.conda_nvvm_subdirs)
    assert "libdevice" in str(PLATFORM.cuda_home_nvvm_subdirs)
