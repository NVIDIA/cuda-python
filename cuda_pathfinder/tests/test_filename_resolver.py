# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for filename resolution utilities."""

import pytest

from cuda.pathfinder._utils.filename_resolver import FilenameResolver


def test_binary_resolution():
    """Test binary filename resolution."""
    result = FilenameResolver.for_binary("nvcc")
    assert result == ("nvcc", "nvcc.exe")
    assert len(result) == 2


def test_binary_resolution_preserves_order():
    """Test that exact name comes first."""
    result = FilenameResolver.for_binary("nvdisasm")
    assert result[0] == "nvdisasm"
    assert result[1] == "nvdisasm.exe"


@pytest.mark.skipif(
    not hasattr(__import__("cuda.pathfinder._utils.platform_aware"), "IS_WINDOWS"),
    reason="Platform detection not available",
)
def test_static_lib_resolution_platform_specific():
    """Test static library resolution is platform-specific."""
    from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

    result = FilenameResolver.for_static_lib("cudadevrt")

    if IS_WINDOWS:
        assert result == ("cudadevrt.lib",)
    else:
        assert result == ("libcudadevrt.a",)


def test_static_lib_preserves_extensions():
    """Test that files with extensions are preserved."""
    result = FilenameResolver.for_static_lib("libdevice.10.bc")
    assert result == ("libdevice.10.bc",)


def test_static_lib_bitcode_files():
    """Test bitcode file resolution."""
    # Bitcode files should be platform-independent
    result = FilenameResolver.for_static_lib("libdevice.10.bc")
    assert result == ("libdevice.10.bc",)

    result = FilenameResolver.for_static_lib("some.other.bc")
    assert result == ("some.other.bc",)
