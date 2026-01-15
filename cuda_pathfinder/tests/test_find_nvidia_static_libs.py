# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from cuda.pathfinder import find_nvidia_static_lib
from cuda.pathfinder._static_libs.supported_nvidia_static_libs import SUPPORTED_STATIC_LIBS


def test_unknown_artifact():
    with pytest.raises(ValueError, match=r"Unknown artifact: 'unknown-artifact'"):
        find_nvidia_static_lib("unknown-artifact")


@pytest.mark.parametrize("artifact_name", SUPPORTED_STATIC_LIBS)
def test_find_static_libs(info_summary_append, artifact_name):
    artifact_path = find_nvidia_static_lib(artifact_name)
    info_summary_append(f"{artifact_path=!r}")
    if artifact_path:
        assert os.path.isfile(artifact_path)
        # Verify the artifact name (or its base) is in the path
        base_name = artifact_name.replace(".10", "")  # Handle libdevice.10.bc -> libdevice
        assert base_name.split(".")[0] in artifact_path.lower()


def test_libdevice_specific(info_summary_append):
    """Specific test for libdevice.10.bc to ensure it's working."""
    artifact_path = find_nvidia_static_lib("libdevice.10.bc")
    info_summary_append(f"libdevice.10.bc path: {artifact_path!r}")
    if artifact_path:
        assert os.path.isfile(artifact_path)
        assert "libdevice" in artifact_path
        # Should end with .bc
        assert artifact_path.endswith(".bc")


def test_libcudadevrt_specific(info_summary_append):
    """Specific test for cudadevrt to ensure it's working."""
    artifact_path = find_nvidia_static_lib("cudadevrt")
    info_summary_append(f"cudadevrt path: {artifact_path!r}")
    if artifact_path:
        assert os.path.isfile(artifact_path)
        # On Linux it should be .a, on Windows it might be .lib
        assert artifact_path.endswith((".a", ".lib"))
        assert "cudadevrt" in artifact_path.lower()


def test_caching():
    """Test that the find functions are properly cached."""
    # Call twice and ensure we get the same object (due to functools.cache)
    path1 = find_nvidia_static_lib("libdevice.10.bc")
    path2 = find_nvidia_static_lib("libdevice.10.bc")
    assert path1 is path2  # Should be the exact same object due to caching
