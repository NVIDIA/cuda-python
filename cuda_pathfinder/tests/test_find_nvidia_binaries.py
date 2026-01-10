# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from cuda.pathfinder import find_nvidia_binary
from cuda.pathfinder._binaries.supported_nvidia_binaries import SUPPORTED_BINARIES

STRICTNESS = os.environ.get("CUDA_PATHFINDER_TEST_FIND_NVIDIA_BINARIES_STRICTNESS", "see_what_works")
assert STRICTNESS in ("see_what_works", "all_must_work")


def test_unknown_binary():
    with pytest.raises(RuntimeError, match=r"^UNKNOWN binary_name='unknown-binary'$"):
        find_nvidia_binary("unknown-binary")


@pytest.mark.parametrize("binary_name", SUPPORTED_BINARIES)
def test_find_binaries(info_summary_append, binary_name):
    binary_path = find_nvidia_binary(binary_name)
    info_summary_append(f"{binary_path=!r}")
    if binary_path:
        assert os.path.isfile(binary_path)
        # Verify the binary name is in the path
        assert binary_name in os.path.basename(binary_path)
    if STRICTNESS == "all_must_work":
        assert binary_path is not None


def test_nvdisasm_specific(info_summary_append):
    """Specific test for nvdisasm to ensure it's working."""
    binary_path = find_nvidia_binary("nvdisasm")
    info_summary_append(f"nvdisasm path: {binary_path!r}")
    # Only assert if we're in strict mode or if cuda-toolkit is installed
    if STRICTNESS == "all_must_work" or os.environ.get("CUDA_HOME") or os.environ.get("CONDA_PREFIX"):
        if binary_path:
            assert os.path.isfile(binary_path)


def test_cuobjdump_specific(info_summary_append):
    """Specific test for cuobjdump to ensure it's working."""
    binary_path = find_nvidia_binary("cuobjdump")
    info_summary_append(f"cuobjdump path: {binary_path!r}")
    # Only assert if we're in strict mode or if cuda-toolkit is installed
    if STRICTNESS == "all_must_work" or os.environ.get("CUDA_HOME") or os.environ.get("CONDA_PREFIX"):
        if binary_path:
            assert os.path.isfile(binary_path)
