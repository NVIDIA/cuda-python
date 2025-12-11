# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import sys

import pytest

from cuda.bindings import _nvml as nvml


def assert_nvml_is_initialized():
    assert nvml.device_get_count_v2() > 0


def assert_nvml_is_uninitialized():
    with pytest.raises(nvml.UninitializedError):
        nvml.device_get_count_v2()


@pytest.mark.skipif(sys.platform == "win32", reason="Test not supported on Windows")
def test_init_ref_count():
    """
    Verifies that we can call NVML shutdown and init(2) multiple times, and that ref counting works
    """
    with pytest.raises(nvml.UninitializedError):
        nvml.shutdown()

    assert_nvml_is_uninitialized()

    for i in range(3):
        # Init 5 times
        for j in range(5):
            nvml.init_v2()
            assert_nvml_is_initialized()

        # Shutdown 4 times, NVML should remain initailized
        for j in range(4):
            nvml.shutdown()
            assert_nvml_is_initialized()

        # Shutdown the final time
        nvml.shutdown()
        assert_nvml_is_uninitialized()


def test_init_check_index(nvml_init):
    """
    Verifies that the index from nvmlDeviceGetIndex is correct
    """
    dev_count = nvml.device_get_count_v2()
    for idx in range(dev_count):
        handle = nvml.device_get_handle_by_index_v2(idx)
        # Verify that the index matches
        assert idx == nvml.device_get_index(handle)
