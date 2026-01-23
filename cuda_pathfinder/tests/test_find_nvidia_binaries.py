# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from cuda.pathfinder import find_nvidia_binary_utility
from cuda.pathfinder._binaries.supported_nvidia_binaries import (
    SUPPORTED_BINARIES,
    SUPPORTED_BINARIES_ALL,
)

STRICTNESS = os.environ.get("CUDA_PATHFINDER_TEST_FIND_NVIDIA_BINARIES_STRICTNESS", "see_what_works")
assert STRICTNESS in ("see_what_works", "all_must_work")


def test_unknown_utility_name():
    with pytest.raises(RuntimeError, match=r"^UNKNOWN utility_name='unknown-utility'$"):
        find_nvidia_binary_utility("unknown-utility")


@pytest.mark.parametrize("utility_name", SUPPORTED_BINARIES)
def test_find_binary_utilities(info_summary_append, utility_name):
    bin_path = find_nvidia_binary_utility(utility_name)
    info_summary_append(f"{bin_path=!r}")

    if bin_path:
        assert os.path.isfile(bin_path), f"Path exists but is not a file: {bin_path}"
        # Note: We verify the file exists but don't check executability here because
        # permissions may vary in test environments (e.g., mounted filesystems, CI
        # containers). The _is_executable() check is tested separately in unit tests.

    if STRICTNESS == "all_must_work":
        assert bin_path is not None, f"Could not find {utility_name}"


def test_supported_binaries_consistency():
    # Ensure SUPPORTED_BINARIES is a subset of SUPPORTED_BINARIES_ALL
    assert set(SUPPORTED_BINARIES).issubset(set(SUPPORTED_BINARIES_ALL))


def test_caching_behavior():
    # Call twice with same utility name to test @functools.cache
    result1 = find_nvidia_binary_utility("nvdisasm")
    result2 = find_nvidia_binary_utility("nvdisasm")
    assert result1 is result2  # Should be the exact same object due to caching


def test_site_packages_bindirs_consistency():
    """Verify SITE_PACKAGES_BINDIRS keys are in SUPPORTED_BINARIES_ALL."""
    from cuda.pathfinder._binaries.supported_nvidia_binaries import SITE_PACKAGES_BINDIRS

    for utility_name in SITE_PACKAGES_BINDIRS:
        assert utility_name in SUPPORTED_BINARIES_ALL, (
            f"Utility '{utility_name}' in SITE_PACKAGES_BINDIRS but not in SUPPORTED_BINARIES_ALL"
        )


def test_caching_per_utility():
    """Verify that different utilities have independent cache entries."""
    nvdisasm1 = find_nvidia_binary_utility("nvdisasm")
    nvcc1 = find_nvidia_binary_utility("nvcc")
    nvdisasm2 = find_nvidia_binary_utility("nvdisasm")
    nvcc2 = find_nvidia_binary_utility("nvcc")

    # Same utility should return cached result
    assert nvdisasm1 is nvdisasm2
    assert nvcc1 is nvcc2

    # Different utilities should have different results (unless both None)
    if nvdisasm1 is not None and nvcc1 is not None:
        assert nvdisasm1 != nvcc1
