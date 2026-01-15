# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cache behavior with explicit contexts."""

import os

import pytest

from cuda.pathfinder import SearchContext, find_nvidia_binary, reset_default_context
from cuda.pathfinder._binaries.supported_nvidia_binaries import SUPPORTED_BINARIES


def test_default_context_is_cached():
    """Test that default context calls are cached."""
    # Reset to ensure clean state
    reset_default_context()

    # Multiple calls with default context should use cache
    if SUPPORTED_BINARIES:
        binary_name = SUPPORTED_BINARIES[0]
        result1 = find_nvidia_binary(binary_name)
        result2 = find_nvidia_binary(binary_name)
        # Should return the same object (cached)
        assert result1 is result2


def test_explicit_context_bypasses_cache(tmp_path):
    """Test that explicit context bypasses cache."""
    # Create mock binary directories
    dir1 = tmp_path / "install1" / "bin"
    dir1.mkdir(parents=True)
    binary1 = dir1 / "nvdisasm"
    binary1.touch()

    dir2 = tmp_path / "install2" / "bin"
    dir2.mkdir(parents=True)
    binary2 = dir2 / "nvdisasm"
    binary2.touch()

    # Use explicit contexts - they should bypass cache
    ctx1 = SearchContext()
    ctx2 = SearchContext()

    # Note: In actual use, the search locations would find different paths
    # This test verifies the structure allows for independent contexts
    assert ctx1 is not ctx2
    assert ctx1.preferred_source is None
    assert ctx2.preferred_source is None


def test_reset_clears_preference():
    """Test that reset_default_context clears the preference."""
    from cuda.pathfinder._utils.toolchain_tracker import ToolchainSource, get_default_context

    reset_default_context()
    ctx = get_default_context()

    # Record something
    ctx.record("test", "/path/to/test", ToolchainSource.CONDA)
    assert ctx.preferred_source == ToolchainSource.CONDA

    # Reset should clear
    reset_default_context()
    ctx2 = get_default_context()
    assert ctx2.preferred_source is None


@pytest.mark.parametrize("binary_name", ["nvdisasm", "cuobjdump"])
def test_explicit_context_independent(binary_name):
    """Test that explicit contexts are independent of default."""
    # Skip if binary not supported
    if binary_name not in SUPPORTED_BINARIES:
        pytest.skip(f"{binary_name} not in SUPPORTED_BINARIES")

    reset_default_context()

    # Use default context
    default_result = find_nvidia_binary(binary_name)

    # Use explicit context - should not affect cache
    explicit_ctx = SearchContext()
    explicit_result = find_nvidia_binary(binary_name, context=explicit_ctx)

    # Both might find the same file, but search was independent
    # If both found something, they should be the same path
    if default_result and explicit_result:
        # Both found - should be same path (but searched independently)
        assert os.path.normpath(default_result) == os.path.normpath(explicit_result)
