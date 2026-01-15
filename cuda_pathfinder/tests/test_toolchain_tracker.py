# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for toolchain consistency tracking."""

import pytest

from cuda.pathfinder._utils.toolchain_tracker import (
    SearchContext,
    SearchLocation,
    ToolchainMismatchError,
    ToolchainSource,
    get_default_context,
    reset_default_context,
    search_location,
)


def test_context_initial_state():
    """Test that a new context starts with no preferred source."""
    ctx = SearchContext()
    assert ctx.preferred_source is None


def test_context_first_record_sets_preference():
    """Test that first recorded artifact sets the preferred source."""
    ctx = SearchContext()
    ctx.record("nvcc", "/conda/bin/nvcc", ToolchainSource.CONDA)
    assert ctx.preferred_source == ToolchainSource.CONDA


def test_context_allows_same_source():
    """Test that context allows multiple artifacts from same source."""
    ctx = SearchContext()
    ctx.record("nvcc", "/conda/bin/nvcc", ToolchainSource.CONDA)
    ctx.record("nvdisasm", "/conda/bin/nvdisasm", ToolchainSource.CONDA)
    assert ctx.preferred_source == ToolchainSource.CONDA


def test_context_rejects_different_source():
    """Test that context raises exception for mixed sources."""
    ctx = SearchContext()
    ctx.record("nvcc", "/conda/bin/nvcc", ToolchainSource.CONDA)

    with pytest.raises(ToolchainMismatchError) as exc_info:
        ctx.record("nvdisasm", "/cuda_home/bin/nvdisasm", ToolchainSource.CUDA_HOME)

    assert exc_info.value.artifact_name == "nvdisasm"
    assert exc_info.value.attempted_source == ToolchainSource.CUDA_HOME
    assert exc_info.value.preferred_source == ToolchainSource.CONDA
    # Check that the improved error message includes helpful text
    assert "reset_default_context()" in str(exc_info.value)
    assert "explicit SearchContext" in str(exc_info.value)


def test_find_prefers_established_source(tmp_path):
    """Test that find searches preferred source first."""
    ctx = SearchContext()

    # Create test directories
    conda_dir = tmp_path / "conda" / "bin"
    conda_dir.mkdir(parents=True)
    (conda_dir / "nvcc").touch()
    (conda_dir / "nvdisasm").touch()  # Also add nvdisasm in conda

    cuda_home_dir = tmp_path / "cuda_home" / "bin"
    cuda_home_dir.mkdir(parents=True)
    (cuda_home_dir / "nvcc").touch()

    site_packages_dir = tmp_path / "site_packages" / "bin"
    site_packages_dir.mkdir(parents=True)
    (site_packages_dir / "nvdisasm").touch()

    # Define locations
    locations = [
        SearchLocation(
            source=ToolchainSource.SITE_PACKAGES,
            base_dir_func=lambda: str(site_packages_dir.parent),
            subdirs=["bin"],
            filename_variants=lambda n: (n,),
        ),
        SearchLocation(
            source=ToolchainSource.CONDA,
            base_dir_func=lambda: str(conda_dir.parent),
            subdirs=["bin"],
            filename_variants=lambda n: (n,),
        ),
        SearchLocation(
            source=ToolchainSource.CUDA_HOME,
            base_dir_func=lambda: str(cuda_home_dir.parent),
            subdirs=["bin"],
            filename_variants=lambda n: (n,),
        ),
    ]

    # First find establishes CONDA preference
    result = ctx.find("nvcc", locations)
    assert result == str(conda_dir / "nvcc")
    assert ctx.preferred_source == ToolchainSource.CONDA

    # Second find should prefer CONDA over SITE_PACKAGES (finds in conda)
    result2 = ctx.find("nvdisasm", locations)
    assert result2 == str(conda_dir / "nvdisasm")  # Should find in conda, not site_packages


def test_search_location_basic(tmp_path):
    """Test basic search_location functionality."""
    # Create test directory
    test_dir = tmp_path / "test" / "bin"
    test_dir.mkdir(parents=True)
    (test_dir / "nvcc").touch()

    location = SearchLocation(
        source=ToolchainSource.CONDA,
        base_dir_func=lambda: str(test_dir.parent),
        subdirs=["bin"],
        filename_variants=lambda n: (n,),
    )

    result = search_location(location, "nvcc")
    assert result == str(test_dir / "nvcc")


def test_search_location_not_found(tmp_path):
    """Test search_location returns None when file not found."""
    location = SearchLocation(
        source=ToolchainSource.CONDA,
        base_dir_func=lambda: str(tmp_path),
        subdirs=["bin"],
        filename_variants=lambda n: (n,),
    )

    result = search_location(location, "nvcc")
    assert result is None


def test_reset_default_context():
    """Test that reset creates a new default context."""
    ctx1 = get_default_context()
    ctx1.record("nvcc", "/conda/bin/nvcc", ToolchainSource.CONDA)

    reset_default_context()

    ctx2 = get_default_context()
    assert ctx2.preferred_source is None
    # Should be a fresh instance
    assert len(ctx2._artifacts) == 0
